"""
backtest_signals.py
===================
Analyses your historical prediction data to answer:

  1. Does ELO divergence from the book predict outcomes?
  2. Does model (p_temp_a) divergence from the book predict outcomes?
  3. At what threshold does any signal become meaningful?
  4. How does the model/ELO/book perform across different surface types?
  5. Where does the model genuinely add value vs just track the market?

Run from your project root:
    python backtest_signals.py

Reads all *_predictions_cck_complete.csv files from ./reports/
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPORTS_DIR = "./reports"

# -- LOAD ALL CCK COMPLETE FILES -------------------------------

def load_all_predictions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    if not files:
        raise FileNotFoundError(f"No CCK complete files found in {REPORTS_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        slug = os.path.basename(f).split("_predictions")[0]
        parts = slug.rsplit("_", 1)
        df["tourney_slug"] = parts[0] if len(parts) == 2 else slug
        df["round_code"]   = parts[1] if len(parts) == 2 else "?"
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only rows with actual results
    combined = combined[combined["correct_prediction"].notna()].copy()

    # Standardise types
    combined["correct_prediction"]      = pd.to_numeric(combined["correct_prediction"], errors="coerce")
    combined["correct_prediction_book"] = pd.to_numeric(combined["correct_prediction_book"], errors="coerce")
    combined["prob_player_a_win"]       = pd.to_numeric(combined["prob_player_a_win"], errors="coerce")
    combined["book_fair_prob_a"]        = pd.to_numeric(combined["book_fair_prob_a"], errors="coerce")
    combined["p_elo_a"]                 = pd.to_numeric(combined["p_elo_a"], errors="coerce")
    combined["p_temp_a"]                = pd.to_numeric(combined["p_temp_a"], errors="coerce")

    return combined


# -- SIGNAL ANALYSIS -------------------------------------------

def analyse_divergence(df: pd.DataFrame, signal_col: str,
                       signal_name: str, book_col: str = "book_fair_prob_a") -> None:
    """
    For a given signal (ELO or model prob), analyse:
    - Does higher divergence from book correlate with worse/better accuracy?
    - What does the book get right that the signal misses?
    """
    sub = df[df[signal_col].notna() & df[book_col].notna() &
             df["correct_prediction"].notna()].copy()

    if len(sub) < 50:
        print(f"  Insufficient data for {signal_name} divergence analysis ({len(sub)} rows)")
        return

    # Divergence = signal - book (for player_a)
    # Positive = signal more bullish on A than book
    sub["divergence"] = sub[signal_col] - sub[book_col]
    sub["abs_div"]    = sub["divergence"].abs()

    # Signal prediction: who does this signal favour?
    sub["signal_pred_correct"] = (
        ((sub[signal_col] >= 0.5) & (sub["correct_prediction"] == 1)) |
        ((sub[signal_col] <  0.5) & (sub["correct_prediction"] == 0))
    ).astype(int)
    sub["book_pred_correct"] = (
        ((sub[book_col] >= 0.5) & (sub["correct_prediction"] == 1)) |
        ((sub[book_col] <  0.5) & (sub["correct_prediction"] == 0))
    ).astype(int)

    print(f"\n{'-'*60}")
    print(f"  {signal_name} vs Book Analysis  (n={len(sub)})")
    print(f"{'-'*60}")

    overall_signal = sub["signal_pred_correct"].mean()
    overall_book   = sub["book_pred_correct"].mean()
    print(f"  Overall {signal_name} accuracy: {overall_signal:.1%}")
    print(f"  Overall book accuracy:          {overall_book:.1%}")
    print(f"  Gap (signal - book):            {(overall_signal-overall_book)*100:+.1f}pp")

    # Pearson correlation between divergence and correct_prediction
    corr = sub["divergence"].corr(sub["correct_prediction"])
    print(f"\n  Corr(divergence, outcome):  {corr:+.3f}")
    print(f"  (positive = signal adds value when it disagrees with book)")

    # Bucket analysis by divergence threshold
    print(f"\n  Accuracy by |divergence| bucket:")
    print(f"  {'Threshold':<18} {'n':>5} {'Signal':>8} {'Book':>8} {'Gap':>8}")
    print(f"  {'-'*50}")

    thresholds = [0, 3, 5, 8, 10, 15, 20]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i+1]
        bucket = sub[(sub["abs_div"] * 100 >= lo) & (sub["abs_div"] * 100 < hi)]
        if len(bucket) < 5:
            continue
        sa = bucket["signal_pred_correct"].mean()
        ba = bucket["book_pred_correct"].mean()
        label = f"{lo}-{hi}pp"
        print(f"  {label:<18} {len(bucket):>5} {sa:>7.1%} {ba:>8.1%} {(sa-ba)*100:>+7.1f}pp")

    # Large divergence bucket
    big = sub[sub["abs_div"] * 100 >= 15]
    if len(big) >= 5:
        sa = big["signal_pred_correct"].mean()
        ba = big["book_pred_correct"].mean()
        print(f"  {'>=15pp':<18} {len(big):>5} {sa:>7.1%} {ba:>8.1%} {(sa-ba)*100:>+7.1f}pp")

    # Direction analysis: when signal and book DISAGREE, who wins?
    disagree = sub[sub["abs_div"] * 100 >= 5]
    if len(disagree) >= 10:
        print(f"\n  When |divergence| >= 5pp (n={len(disagree)}):")
        print(f"    Signal correct:  {disagree['signal_pred_correct'].mean():.1%}")
        print(f"    Book correct:    {disagree['book_pred_correct'].mean():.1%}")

        # Signal goes against book -- does the signal's pick win?
        sig_fav_a = disagree[signal_col] >= 0.5
        book_fav_a = disagree[book_col] >= 0.5
        disagrees_on_winner = disagree[sig_fav_a != book_fav_a]
        if len(disagrees_on_winner) >= 5:
            sig_right = disagrees_on_winner["signal_pred_correct"].mean()
            bk_right  = disagrees_on_winner["book_pred_correct"].mean()
            print(f"\n  When signal and book pick DIFFERENT winner (n={len(disagrees_on_winner)}):")
            print(f"    Signal correct:  {sig_right:.1%}  <- {'SIGNAL WINS' if sig_right>bk_right else 'BOOK WINS'}")
            print(f"    Book correct:    {bk_right:.1%}")


def analyse_by_surface(df: pd.DataFrame) -> None:
    """Model vs book accuracy broken down by surface."""
    sub = df[df["correct_prediction"].notna() &
             df["correct_prediction_book"].notna() &
             df["surface"].notna()].copy()

    if len(sub) < 50:
        return

    print(f"\n{'-'*60}")
    print(f"  Model vs Book by Surface  (n={len(sub)})")
    print(f"{'-'*60}")
    print(f"  {'Surface':<12} {'n':>5} {'Model':>8} {'Book':>8} {'Gap':>8}")
    print(f"  {'-'*45}")

    for surf in ["Clay", "Hard", "Grass"]:
        s = sub[sub["surface"].str.contains(surf, case=False, na=False)]
        if len(s) < 10:
            continue
        ma = s["correct_prediction"].mean()
        ba = s["correct_prediction_book"].mean()
        print(f"  {surf:<12} {len(s):>5} {ma:>7.1%} {ba:>8.1%} {(ma-ba)*100:>+7.1f}pp")


def analyse_by_round(df: pd.DataFrame) -> None:
    """Model vs book by round depth."""
    sub = df[df["correct_prediction"].notna() &
             df["correct_prediction_book"].notna() &
             df["round"].notna()].copy()

    if len(sub) < 50:
        return

    round_order = {"R128":1,"R64":2,"R32":3,"R16":4,"QF":5,"SF":6,"F":7,
                   "RR1":8,"RR2":9,"RR3":10}

    print(f"\n{'-'*60}")
    print(f"  Model vs Book by Round  (n={len(sub)})")
    print(f"{'-'*60}")
    print(f"  {'Round':<8} {'n':>5} {'Model':>8} {'Book':>8} {'Gap':>8}")
    print(f"  {'-'*40}")

    rounds = sorted(sub["round"].unique(), key=lambda r: round_order.get(r.upper(), 99))
    for rnd in rounds:
        s = sub[sub["round"] == rnd]
        if len(s) < 5:
            continue
        ma = s["correct_prediction"].mean()
        ba = s["correct_prediction_book"].mean()
        print(f"  {rnd:<8} {len(s):>5} {ma:>7.1%} {ba:>8.1%} {(ma-ba)*100:>+7.1f}pp")


def analyse_confidence_calibration(df: pd.DataFrame) -> None:
    """
    Calibration: when the model says 70%, does it win 70% of the time?
    Compares model calibration vs book calibration.
    """
    sub = df[df["prob_player_a_win"].notna() &
             df["book_fair_prob_a"].notna() &
             df["correct_prediction"].notna()].copy()

    if len(sub) < 50:
        return

    # Use predicted winner's probability
    sub["model_conf"] = sub.apply(
        lambda r: r["prob_player_a_win"] if r["prob_player_a_win"] >= 0.5
                  else 1 - r["prob_player_a_win"], axis=1)
    sub["book_conf"]  = sub.apply(
        lambda r: r["book_fair_prob_a"] if r["book_fair_prob_a"] >= 0.5
                  else 1 - r["book_fair_prob_a"], axis=1)

    print(f"\n{'-'*60}")
    print(f"  Calibration: Stated Confidence vs Actual Win Rate")
    print(f"{'-'*60}")
    print(f"  {'Model Conf':<14} {'n':>5} {'Actual WR':>10}  {'Book Conf':<14} {'n':>5} {'Actual WR':>10}")
    print(f"  {'-'*60}")

    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%",">90%"]

    for i, lbl in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        m = sub[(sub["model_conf"] >= lo) & (sub["model_conf"] < hi)]
        b = sub[(sub["book_conf"]  >= lo) & (sub["book_conf"]  < hi)]
        m_wr = m["correct_prediction"].mean() if len(m) >= 5 else None
        b_wr = b["correct_prediction"].mean() if len(b) >= 5 else None
        m_str = f"{m_wr:.1%}" if m_wr else "--"
        b_str = f"{b_wr:.1%}" if b_wr else "--"
        m_n = len(m) if len(m) >= 5 else "--"
        b_n = len(b) if len(b) >= 5 else "--"
        print(f"  {lbl:<14} {str(m_n):>5} {m_str:>10}  {lbl:<14} {str(b_n):>5} {b_str:>10}")

    # Brier score (lower = better calibrated)
    m_brier = ((sub["model_conf"] - sub["correct_prediction"]) ** 2).mean()
    b_brier = ((sub["book_conf"]  - sub["correct_prediction"]) ** 2).mean()
    print(f"\n  Brier score (lower = better calibrated):")
    print(f"    Model: {m_brier:.4f}")
    print(f"    Book:  {b_brier:.4f}")
    print(f"    Gap:   {(m_brier-b_brier):+.4f}  ({'model worse' if m_brier>b_brier else 'model better'})")


def model_dampening_analysis(df: pd.DataFrame) -> None:
    """
    Quantify how 'dampened' the model is compared to the book.
    Shows the distribution of model probs vs book probs.
    """
    sub = df[df["prob_player_a_win"].notna() &
             df["book_fair_prob_a"].notna()].copy()

    if len(sub) < 50:
        return

    # Convert to favourite probability (always >= 0.5)
    sub["model_fav"] = sub["prob_player_a_win"].apply(lambda p: max(p, 1-p))
    sub["book_fav"]  = sub["book_fair_prob_a"].apply(lambda p: max(p, 1-p))

    print(f"\n{'-'*60}")
    print(f"  Model Dampening Analysis  (n={len(sub)})")
    print(f"{'-'*60}")
    print(f"  Metric                    Model      Book")
    print(f"  {'-'*45}")
    print(f"  Mean favourite prob:       {sub['model_fav'].mean():.3f}     {sub['book_fav'].mean():.3f}")
    print(f"  Median favourite prob:     {sub['model_fav'].median():.3f}     {sub['book_fav'].median():.3f}")
    print(f"  Std deviation:             {sub['model_fav'].std():.3f}     {sub['book_fav'].std():.3f}")
    print(f"  % with >70% confidence:   {(sub['model_fav']>0.70).mean():.1%}    {(sub['book_fav']>0.70).mean():.1%}")
    print(f"  % with >80% confidence:   {(sub['model_fav']>0.80).mean():.1%}    {(sub['book_fav']>0.80).mean():.1%}")
    print(f"  % with >90% confidence:   {(sub['model_fav']>0.90).mean():.1%}    {(sub['book_fav']>0.90).mean():.1%}")

    # Mean absolute difference
    sub["abs_diff"] = (sub["model_fav"] - sub["book_fav"]).abs()
    print(f"\n  Mean |model - book| prob:  {sub['abs_diff'].mean():.3f}  ({sub['abs_diff'].mean()*100:.1f}pp)")
    print(f"  Correlation (model, book): {sub['model_fav'].corr(sub['book_fav']):.3f}")

    # How often does the model's extra dampening cost it accuracy?
    # Cases where book is very confident (>70%) but model is less so (55-65%)
    damp_cases = sub[(sub["book_fav"] > 0.70) & (sub["model_fav"] < 0.65)]
    if len(damp_cases) >= 5:
        print(f"\n  Cases where book >70% but model <65% (n={len(damp_cases)}):")
        cp = df.loc[damp_cases.index, "correct_prediction"].dropna()
        cpb = df.loc[damp_cases.index, "correct_prediction_book"].dropna()
        if len(cp) >= 5:
            print(f"    Model accuracy:  {cp.mean():.1%}")
            print(f"    Book accuracy:   {cpb.mean():.1%}")
            print(f"    -> Model loses {(cpb.mean()-cp.mean())*100:.1f}pp by being under-confident")


# -- MAIN ------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  CourtIQ Signal Backtest")
    print("="*60)

    df = load_all_predictions()

    print(f"\n  Loaded {len(df):,} scored predictions")
    print(f"  Tournaments: {df['tourney_slug'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    scored = df["correct_prediction"].notna().sum()
    book_scored = df["correct_prediction_book"].notna().sum()
    elo_avail = df["p_elo_a"].notna().sum()
    temp_avail = df["p_temp_a"].notna().sum()
    book_avail = df["book_fair_prob_a"].notna().sum()

    print(f"\n  Data availability:")
    print(f"    Scored predictions:  {scored:,}")
    print(f"    With book data:      {book_scored:,}")
    print(f"    With ELO prob:       {elo_avail:,}")
    print(f"    With temp prob:      {temp_avail:,}")
    print(f"    With book fair prob: {book_avail:,}")

    # Overall accuracy
    model_acc = df["correct_prediction"].mean()
    book_acc  = df["correct_prediction_book"].dropna().mean()
    print(f"\n  Overall model accuracy: {model_acc:.1%}")
    print(f"  Overall book accuracy:  {book_acc:.1%}")
    print(f"  Gap: {(model_acc-book_acc)*100:+.1f}pp")

    # Run analyses
    model_dampening_analysis(df)
    analyse_confidence_calibration(df)
    analyse_by_surface(df)
    analyse_by_round(df)

    if elo_avail >= 50:
        analyse_divergence(df, "p_elo_a", "ELO")
    else:
        print(f"\n  Skipping ELO divergence -- only {elo_avail} rows with ELO data")

    if temp_avail >= 50:
        analyse_divergence(df, "p_temp_a", "Model (temp-scaled)")
    else:
        print(f"\n  Skipping model divergence -- only {temp_avail} rows with temp prob")

    # Final verdict
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Model accuracy:  {model_acc:.1%}")
    print(f"  Book accuracy:   {book_acc:.1%}")
    print(f"  Gap:             {(model_acc-book_acc)*100:+.1f}pp")
    print()
    print(f"  Key questions answered:")
    print(f"  -> Is the model dampened vs the book?  (see dampening analysis)")
    print(f"  -> Does ELO divergence predict anything?  (see ELO analysis)")
    print(f"  -> Where does the model outperform the book?  (see surface/round)")
    print(f"  -> Is the model well-calibrated?  (see calibration + Brier score)")
    print()


if __name__ == "__main__":
    main()
