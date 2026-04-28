"""
fit_calibrator.py
=================
Fits an isotonic regression calibrator on your historical CCK predictions
to correct the model's dampening/under-confidence.

The calibrator maps raw model probabilities to empirically-observed win rates.
Example: model says 62% -> empirically wins 80% -> calibrated output: ~80%

Run from your project root:
    python fit_calibrator.py

Outputs:
    ./models/prob_calibrator.joblib  -- the fitted calibrator
    Prints calibration curve and improvement stats
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

REPORTS_DIR = "./reports"
MODEL_DIR   = "./models"
OUT_PATH    = f"{MODEL_DIR}/prob_calibrator.joblib"


# -- LOAD DATA -------------------------------------------------

def load_predictions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    if not files:
        raise FileNotFoundError(f"No CCK complete files in {REPORTS_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        slug = os.path.basename(f).split("_predictions")[0]
        df["_slug"] = slug
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Need: model prob (prob_player_a_win) and outcome (correct_prediction)
    df = df[df["correct_prediction"].notna() & df["prob_player_a_win"].notna()].copy()
    df["prob_player_a_win"] = pd.to_numeric(df["prob_player_a_win"], errors="coerce")
    df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
    df = df.dropna(subset=["prob_player_a_win", "correct_prediction"])

    # The outcome here: correct_prediction=1 means pred_winner won
    # pred_winner = player_a when prob_player_a_win >= 0.5
    # So we need to reconstruct: did player_a actually win?
    # correct_prediction=1 and pred=a -> a won -> y=1
    # correct_prediction=0 and pred=a -> b won -> y=0
    # correct_prediction=1 and pred=b -> b won -> y=0
    # correct_prediction=0 and pred=b -> a won -> y=1
    pred_a = df["prob_player_a_win"] >= 0.5
    df["y"] = np.where(
        pred_a,
        df["correct_prediction"],        # pred was a, correct=1 -> a won
        1 - df["correct_prediction"]     # pred was b, correct=1 -> b won -> a lost
    )

    print(f"Loaded {len(df):,} scored predictions")
    print(f"  Player A win rate: {df['y'].mean():.3f} (should be ~0.5 if balanced)")
    print(f"  Prob range: {df['prob_player_a_win'].min():.3f} – {df['prob_player_a_win'].max():.3f}")

    return df


# -- FIT CALIBRATOR --------------------------------------------

def fit_calibrator(df: pd.DataFrame):
    """
    Fit isotonic regression calibrator.
    Input:  raw model probability for player A
    Output: empirical probability that player A wins
    """
    X = df["prob_player_a_win"].values
    y = df["y"].values

    # Isotonic regression -- monotonically increasing mapping
    # out_of_bounds='clip' handles probabilities outside training range
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(X, y)

    return calibrator


# -- EVALUATE --------------------------------------------------

def evaluate(df: pd.DataFrame, calibrator) -> None:
    X = df["prob_player_a_win"].values
    y = df["y"].values

    p_raw  = X
    p_cal  = calibrator.predict(X)

    # Brier scores (lower = better)
    brier_raw = np.mean((p_raw - y) ** 2)
    brier_cal = np.mean((p_cal - y) ** 2)

    print(f"\n  Calibration improvement:")
    print(f"    Brier score (raw):        {brier_raw:.4f}")
    print(f"    Brier score (calibrated): {brier_cal:.4f}")
    print(f"    Improvement:              {(brier_raw-brier_cal)*100:+.2f}% {'OK' if brier_cal < brier_raw else 'FAIL'}")

    # Show the calibration curve
    print(f"\n  Calibration curve (raw model -> empirical win rate):")
    print(f"  {'Raw Prob':<12} {'Calibrated':<12} {'Empirical':<12} {'n':>5}")
    print(f"  {'-'*44}")

    bins = np.arange(0.45, 1.01, 0.05)
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (p_raw >= lo) & (p_raw < hi)
        n = mask.sum()
        if n < 5:
            continue
        emp  = y[mask].mean()
        cal  = p_cal[mask].mean()
        raw  = p_raw[mask].mean()
        print(f"  {raw:.2f}-{hi:.2f}     {cal:.3f}        {emp:.3f}        {n:>5}")

    # Accuracy comparison
    acc_raw = ((p_raw >= 0.5) == y.astype(bool)).mean()
    acc_cal = ((p_cal >= 0.5) == y.astype(bool)).mean()
    print(f"\n  Directional accuracy (raw):        {acc_raw:.1%}")
    print(f"  Directional accuracy (calibrated): {acc_cal:.1%}")
    print(f"  Note: accuracy should be identical -- calibration doesn't change which")
    print(f"        player is favoured, only the stated probability magnitude")

    # Show what the calibrator does to confidence levels
    print(f"\n  Effect on stated probabilities:")
    print(f"  {'Raw prob':<12} {'-> Calibrated':<14}")
    print(f"  {'-'*28}")
    for raw_p in [0.51, 0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.66, 0.70, 0.75, 0.80]:
        cal_p = float(calibrator.predict([raw_p])[0])
        print(f"  {raw_p:.2f}         ->  {cal_p:.3f}")


# -- MAIN ------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  CourtIQ Probability Calibrator")
    print("="*60)

    df = load_predictions()

    print(f"\n  Fitting isotonic regression calibrator...")
    calibrator = fit_calibrator(df)

    evaluate(df, calibrator)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(calibrator, OUT_PATH)
    print(f"\n  Calibrator saved: {OUT_PATH}")

    # Quick sanity check -- reload and verify
    cal_check = load(OUT_PATH)
    test_prob = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
    print(f"\n  Sanity check (reload + predict):")
    for p in test_prob:
        c = float(cal_check.predict([p])[0])
        print(f"    {p:.2f} -> {c:.3f}")

    print(f"\n{'='*60}")
    print(f"  Done. Load in your scripts with:")
    print(f"  from joblib import load")
    print(f"  calibrator = load('./models/prob_calibrator.joblib')")
    print(f"  p_calibrated = float(calibrator.predict([p_raw])[0])")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
