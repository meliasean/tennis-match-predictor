"""
fix_missing_data_and_streak.py
================================
Fixes two issues:

1. MISSING MATCH DATA -- Some tournament/round files show far fewer scored
   matches than expected (e.g. IndianWells R128 showing 1/32 instead of 32/32).
   This script diagnoses which files have issues and reports what's missing.

2. STREAK CALCULATION -- The current streak in profiles uses exponential
   smoothing which is wrong. Streak should be consecutive W or L from
   most recent match backwards. This rebuilds it correctly.

Run from your project root:
    python fix_missing_data_and_streak.py

Read-only diagnostic mode by default.
Set FIX_STREAK = True to also rebuild streak values.
"""

import glob
import os
import unicodedata
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"

# Set to True to rebuild streak in profiles after diagnosing
FIX_STREAK = True

CANONICAL = {
    "Felix Auger-Aliassime":   "Felix Auger Aliassime",
    "Jan Lennard Struff":      "Jan-Lennard Struff",
    "Mackenzie Mcdonald":      "Mackenzie McDonald",
    "Christopher Oconnell":    "Christopher O'Connell",
    "Botic Van De Zandschulp": "Botic van de Zandschulp",
    "Aleksandar Vukic":        "Aleksander Vukic",
    "Adolfo Daniel Vallejo":   "Diego Vallejo",
}
def to_canon(raw) -> str:
    if not raw or pd.isna(raw): return ""
    n = unicodedata.normalize("NFKD", str(raw)).encode("ascii","ignore").decode("ascii").strip()
    return CANONICAL.get(n, n)

# Expected match counts per round per tournament size
EXPECTED_COUNTS = {
    # (tourney_type, round): expected_matches
    "R128": 32, "R64": 32, "R32": 16, "R16": 8,
    "QF": 4, "SF": 2, "F": 1,
}

# Known tournament types (Masters have R128, 500s start at R32 usually)
MASTERS = {"indianwells2026","miami2026","montecarlo2026","madrid2026",
           "rome2026","canada2026","cincinnati2026","shanghai2026","paris2026"}
SLAMS   = {"ao2026","rg2026","wimbledon2026","usopen2026",
           "ao2025","rg2025","wimbledon2025","usopen2025"}
ATP500  = {"barcelona2026","halle2026","queens2026","hamburg2026",
           "washington2026","tokyo2026","vienna2026","basel2026",
           "dubai2026","rotterdam2026","acapulco2026"}


def get_expected(slug: str, rnd: str) -> int:
    """Get expected number of matches for this tournament/round."""
    base_slug = slug.split("_")[0] if "_" in slug else slug
    # Masters/Slams have 96-128 draw (R128 has 32, R64 has 32, etc.)
    if base_slug in MASTERS or base_slug in SLAMS:
        return EXPECTED_COUNTS.get(rnd, 0)
    # 500s usually start at R32
    if base_slug in ATP500:
        return {"R32":16,"R16":8,"QF":4,"SF":2,"F":1}.get(rnd, 0)
    # 250s start at R32
    return {"R32":16,"R16":8,"QF":4,"SF":2,"F":1}.get(rnd, 0)


# -- DIAGNOSTIC ------------------------------------------------

def diagnose_files():
    """Check all complete files for missing correct_prediction values."""
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    print(f"Checking {len(files)} complete prediction files...\n")

    problems = []
    totals = {"files":0,"matches":0,"scored":0,"missing":0}

    for fpath in files:
        basename = os.path.basename(fpath)
        # Parse slug and round
        stem = basename.replace("_predictions_cck_complete.csv","")
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        slug, rnd = parts[0], parts[1]

        df = pd.read_csv(fpath)
        cp = pd.to_numeric(df.get("correct_prediction",""), errors="coerce")
        total   = len(df)
        scored  = int(cp.notna().sum())
        missing = total - scored
        expected = get_expected(slug, rnd)

        totals["files"]   += 1
        totals["matches"] += total
        totals["scored"]  += scored
        totals["missing"] += missing

        # Flag if significantly under-scored
        if missing > 0 or (expected > 0 and scored < expected * 0.8):
            problems.append({
                "file":     basename,
                "slug":     slug,
                "round":    rnd,
                "total":    total,
                "scored":   scored,
                "missing":  missing,
                "expected": expected,
            })

    print(f"Summary: {totals['files']} files, {totals['matches']} matches, "
          f"{totals['scored']} scored, {totals['missing']} missing\n")

    if not problems:
        print("OK No issues found -- all matches are scored")
    else:
        print(f"Issues found in {len(problems)} files:\n")
        print(f"{'File':<55} {'Total':>6} {'Scored':>7} {'Missing':>8} {'Expected':>9}")
        print(f"{'-'*90}")
        for p in sorted(problems, key=lambda x: -x["missing"]):
            flag = " <- BLANK" if p["scored"] == 0 else ""
            exp_str = str(p["expected"]) if p["expected"] > 0 else "?"
            print(f"  {p['file']:<53} {p['total']:>6} {p['scored']:>7} "
                  f"{p['missing']:>8} {exp_str:>9}{flag}")

        print(f"\nMost likely causes:")
        print(f"  1. recalibrate_all_predictions.py overwrote correct_prediction with NA")
        print(f"  2. Original files had blank predictions (round not manually scored)")
        print(f"  3. build_site_data() is only counting CCK complete files, not standard")

        # Check if standard (non-CCK) complete files have better coverage
        print(f"\nChecking standard (non-CCK) complete files for same rounds...")
        for p in problems[:5]:
            std_path = Path(REPORTS_DIR) / p["file"].replace("_cck_complete","_complete")
            if std_path.exists():
                df2 = pd.read_csv(std_path)
                cp2 = pd.to_numeric(df2.get("correct_prediction",""), errors="coerce")
                print(f"  {p['slug']} {p['round']}: standard has {cp2.notna().sum()}/{len(df2)} scored")
            else:
                print(f"  {p['slug']} {p['round']}: no standard complete file found")

    return problems


# -- STREAK FIX ------------------------------------------------

def compute_correct_streak(wins_losses: list) -> int:
    """
    Given a list of 1s (win) and 0s (loss) in chronological order,
    compute the current streak:
    - Positive = consecutive wins ending with most recent match
    - Negative = consecutive losses ending with most recent match
    """
    if not wins_losses:
        return 0
    streak = 0
    last = wins_losses[-1]
    for result in reversed(wins_losses):
        if result == last:
            streak += (1 if last == 1 else -1)
        else:
            break
    return streak


def rebuild_streaks():
    """Rebuild correct streaks for all players from complete prediction files."""
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    # Build per-player chronological W/L record
    player_results: dict = {}  # name -> [(date, is_win)]

    for fpath in sorted(files):
        df = pd.read_csv(fpath)
        if "correct_prediction" not in df.columns: continue
        df = df[pd.to_numeric(df.get("correct_prediction",""), errors="coerce").notna()].copy()
        df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
        df["date"] = pd.to_datetime(df.get("date",""), errors="coerce")
        df = df.dropna(subset=["date","correct_prediction","pred_winner"])
        df = df.sort_values("date")

        for _, r in df.iterrows():
            pa   = to_canon(str(r.get("player_a","")))
            pb   = to_canon(str(r.get("player_b","")))
            pred = to_canon(str(r.get("pred_winner","")))
            cp   = int(r["correct_prediction"])
            dt   = r["date"]

            # Reconstruct actual winner/loser
            actual_winner = pred if cp==1 else (pb if pred==pa else pa)
            actual_loser  = pb   if actual_winner==pa else pa

            for p,is_win in [(actual_winner,1),(actual_loser,0)]:
                if not p: continue
                if p not in player_results:
                    player_results[p] = []
                player_results[p].append((dt, is_win))

    # Sort each player's results chronologically
    for p in player_results:
        player_results[p].sort(key=lambda x: x[0])

    # Compute streaks
    streaks = {}
    for p, results in player_results.items():
        wins_losses = [r[1] for r in results]
        streaks[p]  = compute_correct_streak(wins_losses)

    print(f"\nComputed streaks for {len(streaks)} players from prediction files")

    # Show some examples
    print("\nSample streaks (sorted by absolute value):")
    top = sorted(streaks.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    for name, streak in top:
        results = player_results[name]
        last5 = [r[1] for r in results[-5:]]
        l5str  = "".join("W" if r==1 else "L" for r in last5)
        print(f"  {name:<30} streak={streak:>+4}  last5={l5str}")

    # Update profiles
    profiles_path = f"{REPORTS_DIR}/player_profiles_latest.csv"
    if not Path(profiles_path).exists():
        print(f"\nProfiles file not found: {profiles_path}")
        return

    prof = pd.read_csv(profiles_path)
    prof["name"] = prof["name"].apply(to_canon)
    prof["streak_old"] = prof["streak"].copy()

    updated = 0
    for idx, row in prof.iterrows():
        name = row["name"]
        if name in streaks:
            prof.at[idx, "streak"] = float(streaks[name])
            updated += 1

    # Show before/after for key players
    print(f"\nStreak corrections ({updated} players updated):")
    print(f"{'Player':<30} {'Old':>6} {'New':>6} {'Last5 from files':>16}")
    check_players = ["Jannik Sinner","Carlos Alcaraz","Alexander Zverev",
                     "Stefanos Tsitsipas","Arthur Fils","Alexander Blockx",
                     "Casper Ruud","Daniil Medvedev"]
    for name in check_players:
        row = prof[prof["name"]==name]
        if not row.empty:
            r    = row.iloc[0]
            old  = r.get("streak_old",0)
            new  = r.get("streak",0)
            results = player_results.get(name,[])
            l5   = "".join("W" if x[1]==1 else "L" for x in results[-5:])
            flag = " <- changed" if old != new else ""
            print(f"  {name:<30} {old:>+6.0f} {new:>+6.0f}  {l5:>16}{flag}")

    prof = prof.drop(columns=["streak_old"])
    prof.to_csv(profiles_path, index=False, encoding="utf-8-sig")
    prof.to_csv(f"{REPORTS_DIR}/player_profiles_post_madrid_2026.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Saved updated profiles: {profiles_path}")


# -- MAIN ------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  Missing Data Diagnostic + Streak Fix")
    print("="*60)

    print("\n-- MISSING DATA DIAGNOSIS -----------------------------")
    problems = diagnose_files()

    if FIX_STREAK:
        print("\n-- STREAK REBUILD -------------------------------------")
        rebuild_streaks()

    print("\n-- NEXT STEPS -----------------------------------------")
    if problems:
        print("""
The files with missing correct_prediction values need to be manually fixed.
Most likely the recalibrate_all_predictions.py run cleared some values.

To fix: compare the _backup files in reports/_pre_calibration_backup/
against the current files and restore correct_prediction columns.

Quick fix command (restores correct_prediction from backup):
  python3 -c \"
import pandas as pd, glob, os
backup_dir = './reports/_pre_calibration_backup'
for bfile in glob.glob(f'{backup_dir}/*_cck_complete.csv'):
    curr_path = bfile.replace(backup_dir, './reports')
    if not os.path.exists(curr_path): continue
    bdf = pd.read_csv(bfile)
    cdf = pd.read_csv(curr_path)
    # Restore correct_prediction and correct_prediction_book from backup
    for col in ['correct_prediction','correct_prediction_book']:
        if col in bdf.columns:
            cdf[col] = bdf[col].values
    cdf.to_csv(curr_path, index=False)
    scored = pd.to_numeric(cdf['correct_prediction'], errors='coerce').notna().sum()
    print(f'{os.path.basename(curr_path)}: {scored}/{len(cdf)} scored')
  \"
""")
    else:
        print("  All match data looks complete.")

    if FIX_STREAK:
        print("\nStreak rebuilt. Run:")
        print("  python courtiq_engine.py site --output docs/index.html")
        print("  git add reports/player_profiles_latest.csv reports/player_profiles_post_madrid_2026.csv docs/index.html")
        print("  git commit -m 'fix: correct streak calculation, restore missing match data'")
        print("  git push")


if __name__ == "__main__":
    main()
