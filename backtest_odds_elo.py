"""
backtest_odds_elo.py
=====================
Analyses:
  1. Book accuracy grouped by implied probability (odds buckets)
  2. Model accuracy using pure ELO only (p_elo_a)
  3. Model accuracy grouped by ELO difference between players

Run from your project root:
    python backtest_odds_elo.py
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        slug = os.path.basename(f).split("_predictions")[0]
        parts = slug.rsplit("_", 1)
        df["tourney_slug"] = parts[0] if len(parts) == 2 else slug
        df["round_code"]   = parts[1] if len(parts) == 2 else "?"
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["correct_prediction"].notna()].copy()

    for col in ["correct_prediction","correct_prediction_book","prob_player_a_win",
                "book_fair_prob_a","p_elo_a","odds_player_a","odds_player_b",
                "delta_elo","delta_surface_elo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -- 1. BOOK ACCURACY BY ODDS BUCKET ---------------------------

def book_accuracy_by_odds(df: pd.DataFrame) -> None:
    """
    Group by the book's implied probability for the favourite
    and show how accurate the book is in each bucket.
    """
    sub = df[df["book_fair_prob_a"].notna() &
             df["correct_prediction_book"].notna()].copy()

    if len(sub) < 50:
        print(f"  Insufficient data ({len(sub)} rows)")
        return

    # Book's confidence = favourite's implied prob
    sub["book_fav_prob"] = sub["book_fair_prob_a"].apply(lambda p: max(p, 1-p))

    # Convert to approximate American odds for labelling
    def prob_to_odds_label(p):
        if p >= 0.95: return ">95% (-1900+)"
        if p >= 0.90: return "90-95% (-900 to -1900)"
        if p >= 0.80: return "80-90% (-400 to -900)"
        if p >= 0.70: return "70-80% (-233 to -400)"
        if p >= 0.60: return "60-70% (-150 to -233)"
        if p >= 0.55: return "55-60% (-122 to -150)"
        return "50-55% (near pick'em)"

    bins  = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]
    labels = [
        "50-55%  (near pick'em,  ~EVEN to -122)",
        "55-60%  (slight fav,    -122 to -150)",
        "60-70%  (moderate fav,  -150 to -233)",
        "70-80%  (clear fav,     -233 to -400)",
        "80-90%  (heavy fav,     -400 to -900)",
        "90-95%  (dominant fav,  -900 to -1900)",
        "95%+    (massive fav,   -1900+)",
    ]

    print(f"\n{'-'*70}")
    print(f"  Book Accuracy by Implied Probability Bucket  (n={len(sub)})")
    print(f"{'-'*70}")
    print(f"  {'Bucket':<45} {'n':>5} {'Book Acc':>9} {'Expected':>9} {'Edge':>7}")
    print(f"  {'-'*70}")

    total_correct = 0; total_n = 0
    for i, lbl in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        bucket = sub[(sub["book_fav_prob"] >= lo) & (sub["book_fav_prob"] < hi)]
        if len(bucket) < 3:
            continue
        acc      = bucket["correct_prediction_book"].mean()
        expected = bucket["book_fav_prob"].mean()  # what the odds imply
        edge     = acc - expected
        total_correct += bucket["correct_prediction_book"].sum()
        total_n += len(bucket)
        flag = " <-" if abs(edge) > 0.05 else ""
        print(f"  {lbl:<45} {len(bucket):>5} {acc:>8.1%} {expected:>9.1%} {edge:>+6.1%}{flag}")

    print(f"  {'-'*70}")
    if total_n > 0:
        print(f"  {'TOTAL':<45} {total_n:>5} {total_correct/total_n:>8.1%}")

    print(f"""
  Reading: 'Edge' = actual accuracy minus what the odds implied.
  Positive edge = book was right MORE often than odds suggested (well calibrated or lucky).
  Negative edge = book was right LESS often than odds suggested (underdog upsets > expected).
  Near zero = book probabilities closely matched real outcomes.""")


# -- 2. ELO-ONLY ACCURACY --------------------------------------

def elo_only_accuracy(df: pd.DataFrame) -> None:
    """
    Using pure ELO probability (p_elo_a), how accurate is ELO alone?
    Compares ELO vs model vs book.
    """
    sub = df[df["p_elo_a"].notna() &
             df["correct_prediction"].notna()].copy()

    if len(sub) < 50:
        print(f"  Insufficient ELO data ({len(sub)} rows)")
        return

    # ELO prediction: whoever ELO favours
    sub["elo_pred_correct"] = (
        ((sub["p_elo_a"] >= 0.5) & (sub["correct_prediction"] == 1)) |
        ((sub["p_elo_a"] <  0.5) & (sub["correct_prediction"] == 0))
    ).astype(int)

    elo_acc   = sub["elo_pred_correct"].mean()
    model_acc = sub["correct_prediction"].mean()

    bk = sub[sub["correct_prediction_book"].notna()]
    book_acc = bk["correct_prediction_book"].mean() if len(bk) > 0 else None

    print(f"\n{'-'*55}")
    print(f"  Pure ELO Accuracy  (n={len(sub)})")
    print(f"{'-'*55}")
    print(f"  ELO only:        {elo_acc:.1%}")
    print(f"  Full model (CCK):{model_acc:.1%}  (delta {(model_acc-elo_acc)*100:+.1f}pp)")
    if book_acc:
        print(f"  Book:            {book_acc:.1%}  (delta {(book_acc-elo_acc)*100:+.1f}pp vs ELO)")

    # ELO calibration -- when ELO says 70%, does it win 70%?
    sub["elo_fav_prob"] = sub["p_elo_a"].apply(lambda p: max(p, 1-p))

    print(f"\n  ELO calibration (stated confidence vs actual win rate):")
    print(f"  {'ELO Conf':<12} {'n':>5} {'Actual WR':>10} {'Expected':>10} {'Gap':>8}")
    print(f"  {'-'*48}")

    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%","90%+"]

    for i, lbl in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        bucket = sub[(sub["elo_fav_prob"] >= lo) & (sub["elo_fav_prob"] < hi)]
        if len(bucket) < 5:
            continue
        actual   = bucket["elo_pred_correct"].mean()
        expected = bucket["elo_fav_prob"].mean()
        gap      = actual - expected
        print(f"  {lbl:<12} {len(bucket):>5} {actual:>9.1%} {expected:>10.1%} {gap:>+7.1%}")

    # ELO Brier score
    p_elo_for_brier = sub.apply(
        lambda r: r["p_elo_a"] if r["p_elo_a"] >= 0.5 else 1-r["p_elo_a"], axis=1)
    brier_elo = ((p_elo_for_brier - sub["elo_pred_correct"])**2).mean()
    print(f"\n  ELO Brier score: {brier_elo:.4f}")


# -- 3. ACCURACY BY ELO DIFFERENCE ----------------------------

def accuracy_by_elo_delta(df: pd.DataFrame) -> None:
    """
    Group matches by the ELO difference between the two players
    and show how model, ELO, and book perform at each gap level.
    """
    sub = df[df["delta_elo"].notna() &
             df["correct_prediction"].notna()].copy()

    if len(sub) < 50:
        # Try delta_current_elo as fallback
        sub = df[df["delta_current_elo"].notna() &
                 df["correct_prediction"].notna()].copy()
        if len(sub) < 50:
            print(f"  Insufficient delta_elo data")
            return
        sub["delta_elo"] = sub["delta_current_elo"].abs()
    else:
        sub["delta_elo"] = sub["delta_elo"].abs()

    has_elo = df["p_elo_a"].notna().sum() > 50
    if has_elo:
        sub = sub[sub["p_elo_a"].notna()].copy()
        sub["elo_pred_correct"] = (
            ((sub["p_elo_a"] >= 0.5) & (sub["correct_prediction"] == 1)) |
            ((sub["p_elo_a"] <  0.5) & (sub["correct_prediction"] == 0))
        ).astype(int)

    print(f"\n{'-'*70}")
    print(f"  Accuracy by ELO Difference  (n={len(sub)})")
    print(f"{'-'*70}")
    print(f"  {'ELO Gap':<20} {'n':>5} {'Model':>8} {'ELO':>8} {'Book':>8} {'Upsets':>8}")
    print(f"  {'-'*60}")

    buckets = [
        (0,   25,  "0-25   (near equal)"),
        (25,  50,  "25-50  (slight edge)"),
        (50,  100, "50-100 (clear edge)"),
        (100, 150, "100-150 (big edge)"),
        (150, 250, "150-250 (dominant)"),
        (250, 999, "250+   (massive)"),
    ]

    for lo, hi, lbl in buckets:
        bucket = sub[(sub["delta_elo"] >= lo) & (sub["delta_elo"] < hi)]
        if len(bucket) < 5:
            continue

        model_a = bucket["correct_prediction"].mean()
        elo_a   = bucket["elo_pred_correct"].mean() if has_elo else None
        bk      = bucket[bucket["correct_prediction_book"].notna()]
        book_a  = bk["correct_prediction_book"].mean() if len(bk) >= 5 else None

        # Upset rate = how often the ELO underdog wins
        upset_rate = 1 - model_a  # proxy -- wrong predictions

        model_str = f"{model_a:.1%}"
        elo_str   = f"{elo_a:.1%}" if elo_a else "--"
        book_str  = f"{book_a:.1%}" if book_a else "--"
        upset_str = f"{upset_rate:.1%}"

        print(f"  {lbl:<20} {len(bucket):>5} {model_str:>8} {elo_str:>8} {book_str:>8} {upset_str:>8}")

    print(f"""
  Reading:
  - Small ELO gaps (0-50): high upset risk, all signals converge toward 50/50
  - Large ELO gaps (150+): heavy favourites, upsets rare but impactful
  - 'Upsets' = fraction of matches where lower-ELO player won""")

    # Key insight: at what ELO gap does the model become reliable?
    reliable = sub[sub["delta_elo"] >= 100]
    if len(reliable) >= 20:
        ra = reliable["correct_prediction"].mean()
        print(f"\n  Matches with ELO gap >= 100 ({len(reliable)} matches):")
        print(f"    Model accuracy: {ra:.1%}")
        if has_elo:
            print(f"    ELO accuracy:   {reliable['elo_pred_correct'].mean():.1%}")

    # ELO gap distribution
    print(f"\n  ELO gap distribution:")
    print(f"    Mean gap:   {sub['delta_elo'].mean():.0f} ELO points")
    print(f"    Median gap: {sub['delta_elo'].median():.0f} ELO points")
    print(f"    % with gap <50:  {(sub['delta_elo']<50).mean():.1%}")
    print(f"    % with gap >100: {(sub['delta_elo']>100).mean():.1%}")
    print(f"    % with gap >150: {(sub['delta_elo']>150).mean():.1%}")


# -- 4. SURFACE ELO -------------------------------------------

def surface_elo_analysis(df: pd.DataFrame) -> None:
    """Does surface-specific ELO (delta_surface_elo) predict better than overall ELO?"""
    sub = df[df["delta_surface_elo"].notna() &
             df["delta_elo"].notna() &
             df["correct_prediction"].notna()].copy()

    if len(sub) < 50:
        return

    sub["overall_elo_correct"] = (sub["delta_elo"] > 0) == (sub["correct_prediction"] == 1)
    sub["surface_elo_correct"] = (sub["delta_surface_elo"] > 0) == (sub["correct_prediction"] == 1)

    oa = sub["overall_elo_correct"].mean()
    sa = sub["surface_elo_correct"].mean()

    print(f"\n{'-'*55}")
    print(f"  Surface ELO vs Overall ELO  (n={len(sub)})")
    print(f"{'-'*55}")
    print(f"  Overall ELO accuracy: {oa:.1%}")
    print(f"  Surface ELO accuracy: {sa:.1%}  (delta {(sa-oa)*100:+.1f}pp)")

    if abs(sa-oa)*100 < 1:
        print(f"\n  -> Minimal difference. Surface ELO adds little beyond overall ELO.")
    elif sa > oa:
        print(f"\n  -> Surface ELO is better. Surface-specific history matters.")
    else:
        print(f"\n  -> Overall ELO is better. Surface ELO may be overfitting.")


# -- MAIN -----------------------------------------------------

def main():
    print("\n" + "="*70)
    print("  CourtIQ Odds & ELO Backtest")
    print("="*70)

    df = load_data()
    print(f"\n  Loaded {len(df):,} scored predictions across {df['tourney_slug'].nunique()} tournaments")

    overall_model = df["correct_prediction"].mean()
    overall_book  = df["correct_prediction_book"].dropna().mean()
    print(f"  Overall -- Model: {overall_model:.1%}  Book: {overall_book:.1%}")

    book_accuracy_by_odds(df)
    elo_only_accuracy(df)
    accuracy_by_elo_delta(df)
    surface_elo_analysis(df)

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
