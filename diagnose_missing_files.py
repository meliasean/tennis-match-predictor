"""
diagnose_missing_files.py
=========================
Prints the actual player names and match structure from every file
that still has unscored correct_prediction rows.

Run from your project root:
    python diagnose_missing_files.py

This tells you exactly what names are in each file so backfill_missing_data.py
can be updated with the correct pairings.
"""

import glob, os
import pandas as pd
from pathlib import Path

REPORTS_DIR = "./reports"

CHECK = [
    ("acapulco2026",     "R32"),
    ("acapulco2026",     "R16"),
    ("acapulco2026",     "QF"),
    ("indianwells2026",  "R128"),
    ("canada2025",       "R32"),
    ("canada2025",       "R16"),
    ("montecarlo2026",   "R32"),
    ("montecarlo2026",   "R64"),
]

for slug, rnd in CHECK:
    fpath = Path(REPORTS_DIR) / f"{slug}_{rnd}_predictions_cck_complete.csv"
    if not fpath.exists():
        print(f"\n{slug} {rnd}: FILE NOT FOUND")
        continue

    df = pd.read_csv(fpath)
    cp = pd.to_numeric(df.get("correct_prediction",""), errors="coerce")
    missing = df[cp.isna()].copy()
    scored  = df[cp.notna()].copy()

    print(f"\n{'='*60}")
    print(f"{slug} {rnd}: {len(df)} total rows, {len(scored)} scored, {len(missing)} unscored")

    if len(missing) > 0:
        print(f"\n  UNSCORED rows (need correct_prediction filled):")
        for _, r in missing.iterrows():
            pa   = r.get("player_a","?")
            pb   = r.get("player_b","?")
            pred = r.get("pred_winner","?")
            oa   = r.get("odds_player_a","")
            ob   = r.get("odds_player_b","")
            print(f"    {pa}  vs  {pb}  |  pred={pred}  odds={oa}/{ob}")

    if len(scored) > 0 and len(missing) > 0:
        print(f"\n  Already scored (for reference):")
        for _, r in scored.head(3).iterrows():
            print(f"    {r.get('player_a','?')} vs {r.get('player_b','?')} cp={r.get('correct_prediction','?')}")
