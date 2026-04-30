"""
recalibrate_all_predictions.py
================================
Retroactively applies the fitted probability calibrator to all existing
_predictions_cck_complete.csv and _predictions_complete.csv files.

Updates these columns in place:
  - prob_player_a_win  -> calibrated value
  - prob_player_b_win  -> 1 - calibrated value
  - confidence         -> max(calibrated_a, calibrated_b)
  - p_temp_a           -> calibrated value (where it exists)

Does NOT change:
  - pred_winner        (directional pick stays the same)
  - correct_prediction (results stay the same)
  - odds, book_fair_prob_a, p_elo_a, deltas

Run from your project root:
    python recalibrate_all_predictions.py

Prints before/after accuracy stats for every tournament.
"""

import glob
import os
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import load

REPORTS_DIR    = "./reports"
CALIBRATOR_PATH = "./models/prob_calibrator.joblib"
BACKUP_DIR     = "./reports/_pre_calibration_backup"


def load_calibrator():
    if not Path(CALIBRATOR_PATH).exists():
        raise FileNotFoundError(
            f"Calibrator not found at {CALIBRATOR_PATH}\n"
            f"Run fit_calibrator.py first."
        )
    cal = load(CALIBRATOR_PATH)
    print(f"  Calibrator loaded: {CALIBRATOR_PATH}")
    return cal


def backup_files(files: list) -> None:
    """Create a backup of original files before modifying."""
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = Path(BACKUP_DIR) / Path(f).name
        if not dest.exists():  # don't overwrite existing backup
            shutil.copy2(f, dest)
    print(f"  Backed up {len(files)} files to {BACKUP_DIR}/")


def apply_calibration(df: pd.DataFrame, calibrator) -> pd.DataFrame:
    """Apply calibrator to prob columns in a dataframe."""
    df = df.copy()

    if "prob_player_a_win" not in df.columns:
        return df

    # Get raw probs
    raw_a = pd.to_numeric(df["prob_player_a_win"], errors="coerce")
    valid = raw_a.notna()

    if valid.sum() == 0:
        return df

    # Apply calibrator
    cal_a = raw_a.copy()
    cal_a[valid] = calibrator.predict(raw_a[valid].values)
    cal_b = 1.0 - cal_a

    # Clip to valid range
    cal_a = cal_a.clip(0.01, 0.99)
    cal_b = cal_b.clip(0.01, 0.99)

    df["prob_player_a_win"] = cal_a.round(6)
    df["prob_player_b_win"] = cal_b.round(6)
    df["confidence"]        = np.maximum(cal_a, cal_b).round(6)

    # Also update p_temp_a if it exists (it's the pre-CCK model prob)
    if "p_temp_a" in df.columns:
        raw_temp = pd.to_numeric(df["p_temp_a"], errors="coerce")
        temp_valid = raw_temp.notna()
        if temp_valid.sum() > 0:
            cal_temp = raw_temp.copy()
            cal_temp[temp_valid] = calibrator.predict(raw_temp[temp_valid].values)
            df["p_temp_a"] = cal_temp.clip(0.01, 0.99).round(6)

    return df


def process_tournament(slug: str, files: list, calibrator) -> dict:
    """Process all files for one tournament slug."""
    stats = {"slug": slug, "files": 0, "rows": 0,
             "before_model": None, "after_model": None,
             "before_book": None}

    all_cp_before = []
    all_cp_after  = []
    all_cpb       = []

    for fpath in sorted(files):
        df = pd.read_csv(fpath)

        # Track accuracy before
        cp = pd.to_numeric(df["correct_prediction"] if "correct_prediction" in df.columns else pd.Series([], dtype=float), errors="coerce").dropna()
        cpb = pd.to_numeric(df["correct_prediction_book"] if "correct_prediction_book" in df.columns else pd.Series([], dtype=float), errors="coerce").dropna()
        all_cp_before.extend(cp.tolist())
        all_cpb.extend(cpb.tolist())

        # Apply calibration
        df_new = apply_calibration(df, calibrator)

        # Track accuracy after (same values -- calibration doesn't change picks)
        cp_new = pd.to_numeric(df_new.get("correct_prediction"), errors="coerce").dropna()
        all_cp_after.extend(cp_new.tolist())

        # Save
        df_new.to_csv(fpath, index=False, encoding="utf-8-sig")
        stats["files"] += 1
        stats["rows"]  += len(df_new)

    if all_cp_before:
        stats["before_model"] = np.mean(all_cp_before)
        stats["after_model"]  = np.mean(all_cp_after)
    if all_cpb:
        stats["before_book"] = np.mean(all_cpb)

    return stats


def main():
    print("\n" + "="*65)
    print("  CourtIQ Retroactive Probability Recalibration")
    print("="*65)

    calibrator = load_calibrator()

    # Find all complete files (both standard and CCK)
    all_files = sorted(
        glob.glob(f"{REPORTS_DIR}/*_predictions_complete.csv") +
        glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv")
    )

    # Exclude ALL files
    all_files = [f for f in all_files
                 if "_ALL_" not in f and "all_rounds" not in f]

    print(f"\n  Found {len(all_files)} complete prediction files")

    if not all_files:
        print("  No files found. Check REPORTS_DIR path.")
        return

    # Backup first
    print(f"\n  Creating backup...")
    backup_files(all_files)

    # Group by tournament slug
    from collections import defaultdict
    tourney_files = defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        # Extract tournament slug -- everything before _R or _QF etc.
        # e.g. ao2026_R64_predictions_cck_complete -> ao2026
        parts = basename.split("_predictions")[0]
        # Remove round suffix
        import re
        slug = re.sub(r'_(R\d+|QF|SF|F|RR\d+)$', '', parts)
        tourney_files[slug].append(f)

    print(f"  Tournaments to process: {len(tourney_files)}")

    # Process each tournament
    print(f"\n  {'Tournament':<30} {'Files':>5} {'Model Before':>13} {'Model After':>12} {'Book':>8}")
    print(f"  {'-'*72}")

    total_before = []
    total_after  = []
    total_book   = []

    for slug in sorted(tourney_files.keys()):
        files = tourney_files[slug]
        stats = process_tournament(slug, files, calibrator)

        mb = f"{stats['before_model']:.1%}" if stats["before_model"] else "--"
        ma = f"{stats['after_model']:.1%}"  if stats["after_model"]  else "--"
        bk = f"{stats['before_book']:.1%}"  if stats["before_book"]  else "--"

        # Note: before/after model acc should be identical (picks don't change)
        flag = ""
        if stats["before_model"] and stats["after_model"]:
            total_before.append(stats["before_model"])
            total_after.append(stats["after_model"])
        if stats["before_book"]:
            total_book.append(stats["before_book"])

        display_name = slug[:28]
        print(f"  {display_name:<30} {stats['files']:>5} {mb:>13} {ma:>12} {bk:>8}")

    # Summary
    print(f"\n  {'-'*72}")
    if total_before:
        avg_before = np.mean(total_before)
        avg_after  = np.mean(total_after)
        avg_book   = np.mean(total_book) if total_book else None
        print(f"  {'OVERALL (avg)':<30} {'':>5} {avg_before:>12.1%} {avg_after:>12.1%} "
              f"{avg_book:>8.1%}" if avg_book else f"  {'OVERALL (avg)':<30} {'':>5} {avg_before:>12.1%} {avg_after:>12.1%}")

    print(f"""
  Notes:
  - Model accuracy (before/after) should be IDENTICAL -- calibration
    changes stated probabilities but not which player is predicted.
  - What changes: prob values are now wider/more confident.
  - Edge scores on CourtIQ will now show meaningful divergence from book.
  - Backup of originals saved to: {BACKUP_DIR}/

  Done. {len(all_files)} files updated.
""")

    # Show what calibration did to a sample match
    print("  Example: what calibration did to probabilities")
    print(f"  {'Raw prob':<12} {'-> Calibrated':<14} {'Change':>8}")
    print(f"  {'-'*36}")
    for p in [0.51, 0.53, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70, 0.75]:
        c = float(calibrator.predict([p])[0])
        print(f"  {p:.2f}         ->  {c:.3f}          {(c-p)*100:>+.1f}pp")


if __name__ == "__main__":
    main()
