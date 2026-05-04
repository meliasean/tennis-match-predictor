"""
update_madrid_sf_f.py
=====================
Scores the Madrid 2026 SF and F results:

SF: Sinner def. Fils 2-0  (-625/+500)   01 May 2026
SF: Zverev def. Blockx 2-0 (-435/+333)  01 May 2026
F:  Sinner def. Zverev 2-0 (-588/+450)  03 May 2026

Run from your project root:
    python update_madrid_sf_f.py
"""

import unicodedata, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"

def norm(n):
    if not n or pd.isna(n): return ""
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
    return n.lower().strip().replace("-","").replace("'","").replace(" ","")

def ap(o):
    if o is None or pd.isna(o): return np.nan
    try: o = float(o)
    except: return np.nan
    return (-o)/((-o)+100) if o < 0 else 100/(o+100)

def devig(a, b):
    if any(np.isnan(v) for v in [a,b]) or a<=0 or b<=0: return np.nan,np.nan
    s=a+b; return a/s,b/s

# (player_a_approx, player_b_approx, actual_winner, odds_a, odds_b)
RESULTS = [
    ("SF",  "Jannik Sinner", "Arthur Fils",      "Jannik Sinner", -625,  500),
    ("SF",  "Alexander Blockx", "Alexander Zverev", "Alexander Zverev", 333, -435),
    ("F",   "Jannik Sinner", "Alexander Zverev", "Jannik Sinner", -588,  450),
]

def score_row(df, pa_approx, pb_approx, winner_approx, odds_a, odds_b):
    """Find the matching row by fuzzy name, fill correct_prediction."""
    all_names = list(set(df["player_a"].dropna().tolist() + df["player_b"].dropna().tolist()))

    # Find pa and pb in the file
    pa_m = next((n for n in all_names if norm(pa_approx) in norm(n) or norm(n) in norm(pa_approx)), None)
    pb_m = next((n for n in all_names if norm(pb_approx) in norm(n) or norm(n) in norm(pb_approx)), None)

    if not pa_m or not pb_m:
        print(f"    NAME NOT FOUND: '{pa_approx}'->{pa_m}  '{pb_approx}'->{pb_m}")
        print(f"    Available: {all_names}")
        return df, False

    aw_m = pa_m if (norm(winner_approx) in norm(pa_m) or norm(pa_m) in norm(winner_approx)) else pb_m

    mask = (
        ((df["player_a"]==pa_m) & (df["player_b"]==pb_m)) |
        ((df["player_a"]==pb_m) & (df["player_b"]==pa_m))
    )
    if not mask.any():
        print(f"    ROW NOT FOUND for: {pa_m} vs {pb_m}")
        return df, False

    idx = df[mask].index[0]

    # Determine correct_prediction
    pred = str(df.at[idx, "pred_winner"])
    correct = 1 if (norm(pred) in norm(aw_m) or norm(aw_m) in norm(pred)) else 0

    # Book prediction
    paf, pbf = devig(ap(odds_a), ap(odds_b))
    if not np.isnan(paf):
        bk = df.at[idx,"player_a"] if paf >= 0.5 else df.at[idx,"player_b"]
        bk_correct = 1 if (norm(bk) in norm(aw_m) or norm(aw_m) in norm(bk)) else 0
    else:
        bk_correct = np.nan

    old_cp = df.at[idx,"correct_prediction"]
    df.at[idx,"correct_prediction"] = correct
    df.at[idx,"correct_prediction_book"] = bk_correct
    df.at[idx,"odds_player_a"] = float(odds_a)
    df.at[idx,"odds_player_b"] = float(odds_b)

    # Also fill book fair probs
    if not np.isnan(paf):
        df.at[idx,"book_fair_prob_a"] = round(float(paf), 6)
        df.at[idx,"book_fair_prob_b"] = round(float(pbf), 6)

    changed = pd.isna(old_cp) or int(old_cp) != correct
    result_str = "correct OK" if correct else "wrong FAIL"
    flag = " (was different)" if changed and pd.notna(old_cp) else ""
    print(f"    {pa_m} vs {pb_m} -> {aw_m} [{result_str}]{flag}")
    return df, True


def main():
    print("\n=== Madrid 2026 SF + F Results ===\n")

    total = 0
    for rnd, pa, pb, winner, oa, ob in RESULTS:
        print(f"{rnd}: {pa} vs {pb} -> {winner}")
        for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
            fpath = Path(REPORTS_DIR)/f"madrid2026_{rnd}{suffix}"
            if not fpath.exists():
                print(f"  NOT FOUND: {fpath.name}")
                continue
            df = pd.read_csv(fpath)
            df, changed = score_row(df, pa, pb, winner, oa, ob)
            if changed:
                df.to_csv(fpath, index=False)
                print(f"  Saved: {fpath.name}")
                total += 1
        print()

    print(f"Done. {total} files updated.")
    print(f"\nNext:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/madrid2026_*.csv docs/index.html")
    print(f"  git commit -m 'results: Madrid 2026 SF + F -- Sinner champion'")
    print(f"  git push")

if __name__ == "__main__":
    main()
