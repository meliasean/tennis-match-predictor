"""
test_elo_recalibration.py
=========================
Tests an improved ELO system against your current one using the full
match dataset from 2021 through Madrid 2026.

Improvements tested:
  1. Starting ELO: 1500 -> 1200 (new players start lower, more realistic)
  2. Fixed K -> Diminishing K: starts high (fast learning) and decreases
     as matches accumulate, converging on a player's true level

K formula: K(n) = K_floor + (K_start - K_floor) * exp(-n / K_decay)
  Where n = number of matches played so far
  K starts high (~32) for unknown players and decays toward a floor (~8)
  K_decay controls how quickly it converges (50 = halves at ~35 matches)

Run from your project root:
    python test_elo_recalibration.py

Reads all *_predictions_cck_complete.csv files from ./reports/
to extract the full match history and test both ELO systems.
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
import unicodedata

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"


# -- ELO SYSTEM CONFIGS ----------------------------------------

@dataclass
class EloConfig:
    name: str
    base: float          # starting ELO for new players
    k_start: float       # initial K factor (high = fast learning)
    k_floor: float       # minimum K factor (converges to this)
    k_decay: float       # matches until K halves from (k_start - k_floor)
    k_surface: float     # surface ELO K factor (same diminishing logic)
    k_surf_floor: float  # surface K floor

    def k(self, n_matches: int) -> float:
        """Compute K factor given number of prior matches."""
        return self.k_floor + (self.k_start - self.k_floor) * np.exp(-n_matches / self.k_decay)

    def k_surf(self, n_matches: int) -> float:
        return self.k_surf_floor + (self.k_surface - self.k_surf_floor) * np.exp(-n_matches / self.k_decay)


CONFIGS = [
    EloConfig(
        name       = "Current (base=1500, fixed K=24)",
        base       = 1500.0,
        k_start    = 24.0,
        k_floor    = 24.0,   # fixed = no decay
        k_decay    = 999.0,  # effectively infinite = no decay
        k_surface  = 18.0,
        k_surf_floor = 18.0,
    ),
    EloConfig(
        name       = "New (base=1200, diminishing K)",
        base       = 1200.0,
        k_start    = 40.0,   # fast learning for unknown players
        k_floor    = 8.0,    # converges to 8 for established players
        k_decay    = 60.0,   # K halves after ~42 matches
        k_surface  = 28.0,
        k_surf_floor = 6.0,
    ),
    EloConfig(
        name       = "Alt (base=1200, moderate decay)",
        base       = 1200.0,
        k_start    = 32.0,
        k_floor    = 10.0,
        k_decay    = 80.0,   # slower decay
        k_surface  = 22.0,
        k_surf_floor = 7.0,
    ),
]


# -- LOAD MATCH HISTORY ----------------------------------------

ALIASES = {
    "Felix Auger-Aliassime": "Felix Auger Aliassime",
    "Botic Van De Zandschulp": "Botic van de Zandschulp",
}
def alias(n):
    if not n or pd.isna(n): return str(n)
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
    return ALIASES.get(n, n)

SURFACE_MAP = {"Clay":"Clay","Hard":"Hard","Grass":"Grass","Carpet":"Hard","Acrylic":"Hard"}

def norm_surface(s):
    if pd.isna(s): return "Hard"
    for k,v in SURFACE_MAP.items():
        if k.lower() in str(s).lower(): return v
    return "Hard"


def load_match_history() -> pd.DataFrame:
    """
    Load all scored matches from complete prediction files.
    Extracts: date, winner_name, loser_name, surface, tourney_slug
    """
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    if not files:
        raise FileNotFoundError(f"No CCK complete files in {REPORTS_DIR}")

    rows = []
    for fpath in files:
        df = pd.read_csv(fpath)
        slug = os.path.basename(fpath).split("_predictions")[0]
        parts = slug.rsplit("_", 1)
        tourney = parts[0] if len(parts) == 2 else slug

        req = ["player_a","player_b","pred_winner","correct_prediction","date"]
        if not all(c in df.columns for c in req):
            continue

        df = df[df["correct_prediction"].notna()].copy()
        df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
        df = df.dropna(subset=["correct_prediction"])

        surface = "Clay"
        if "surface" in df.columns and len(df) > 0:
            surface = norm_surface(df["surface"].iloc[0])

        for _, r in df.iterrows():
            pa   = alias(r["player_a"])
            pb   = alias(r["player_b"])
            pred = alias(r["pred_winner"])
            cp   = int(r["correct_prediction"])

            # Reconstruct actual winner
            actual_winner = pred if cp == 1 else (pb if pred == pa else pa)
            actual_loser  = pb if actual_winner == pa else pa

            rows.append({
                "date":    r["date"],
                "winner":  actual_winner,
                "loser":   actual_loser,
                "surface": surface,
                "tourney": tourney,
            })

    df_all = pd.DataFrame(rows)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all = df_all.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"  Loaded {len(df_all):,} matches from {df_all['tourney'].nunique()} tournaments")
    print(f"  Date range: {df_all['date'].min().date()} -> {df_all['date'].max().date()}")
    print(f"  Unique players: {pd.concat([df_all['winner'],df_all['loser']]).nunique()}")
    return df_all


# -- ELO ENGINE ------------------------------------------------

def run_elo_system(matches: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """
    Run ELO through all matches. Returns a dataframe with pre-match ELO
    for each match (winner and loser) plus prediction accuracy.
    """
    elo_overall:  Dict[str, float] = {}
    elo_surface:  Dict[Tuple[str,str], float] = {}
    n_matches:    Dict[str, int] = {}    # match count per player

    results = []

    for _, r in matches.iterrows():
        w = r["winner"]; l = r["loser"]; s = r["surface"]

        # Initialise if new player
        if w not in elo_overall:
            elo_overall[w] = cfg.base; n_matches[w] = 0
        if l not in elo_overall:
            elo_overall[l] = cfg.base; n_matches[l] = 0
        if (w, s) not in elo_surface:
            elo_surface[(w,s)] = cfg.base
        if (l, s) not in elo_surface:
            elo_surface[(l,s)] = cfg.base

        # Pre-match ELO
        ew_pre = elo_overall[w]; el_pre = elo_overall[l]
        es_w   = elo_surface[(w,s)]; es_l = elo_surface[(l,s)]

        # Expected scores
        exp_w   = 1.0 / (1.0 + 10 ** ((el_pre - ew_pre) / 400))
        exp_w_s = 1.0 / (1.0 + 10 ** ((es_l   - es_w)   / 400))

        # ELO prediction: did the higher-ELO player win?
        elo_pred_correct = 1 if ew_pre >= el_pre else 0

        # K factors (diminishing)
        kw = cfg.k(n_matches[w]); kl = cfg.k(n_matches[l])
        k_avg = (kw + kl) / 2  # use average for this match

        ks_w = cfg.k_surf(n_matches[w]); ks_l = cfg.k_surf(n_matches[l])
        ks_avg = (ks_w + ks_l) / 2

        results.append({
            "winner":         w,
            "loser":          l,
            "surface":        s,
            "elo_w_pre":      ew_pre,
            "elo_l_pre":      el_pre,
            "elo_delta":      abs(ew_pre - el_pre),
            "elo_pred_correct": elo_pred_correct,
            "exp_w":          exp_w,
            "k_used":         k_avg,
            "n_w":            n_matches[w],
            "n_l":            n_matches[l],
        })

        # Update ELO
        new_ew = ew_pre + k_avg * (1 - exp_w)
        new_el = el_pre - k_avg * (1 - exp_w)
        new_es_w = es_w + ks_avg * (1 - exp_w_s)
        new_es_l = es_l - ks_avg * (1 - exp_w_s)

        elo_overall[w] = new_ew; elo_overall[l] = new_el
        elo_surface[(w,s)] = new_es_w; elo_surface[(l,s)] = new_es_l
        n_matches[w] += 1; n_matches[l] += 1

    return pd.DataFrame(results), elo_overall, n_matches


# -- COMPARE SYSTEMS -------------------------------------------

def evaluate_system(df: pd.DataFrame, cfg: EloConfig, elo_final: dict,
                    n_matches: dict) -> None:
    """Print evaluation metrics for one ELO system."""

    print(f"\n  {'-'*60}")
    print(f"  {cfg.name}")
    print(f"  {'-'*60}")

    # Overall accuracy
    acc = df["elo_pred_correct"].mean()
    print(f"  Overall ELO accuracy: {acc:.1%}  (n={len(df):,})")

    # Brier score vs expected
    brier = ((df["exp_w"] - 1.0)**2).mean()  # winner always scores 1
    print(f"  Brier score: {brier:.4f}")

    # Calibration
    print(f"\n  Calibration (ELO win probability vs actual):")
    print(f"  {'ELO Conf':<12} {'n':>6} {'Actual WR':>10} {'Expected':>10} {'Gap':>8} {'K avg':>8}")
    print(f"  {'-'*56}")

    df["elo_fav"] = df.apply(lambda r: max(r["exp_w"], 1-r["exp_w"]), axis=1)
    df["elo_fav_correct"] = df["elo_pred_correct"]

    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%","90%+"]
    for i, lbl in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        b = df[(df["elo_fav"] >= lo) & (df["elo_fav"] < hi)]
        if len(b) < 5: continue
        actual   = b["elo_fav_correct"].mean()
        expected = b["elo_fav"].mean()
        gap      = actual - expected
        k_avg    = b["k_used"].mean()
        flag     = " <-" if gap > 0.05 else ""
        print(f"  {lbl:<12} {len(b):>6} {actual:>9.1%} {expected:>10.1%} {gap:>+7.1%} {k_avg:>8.1f}{flag}")

    # By ELO gap
    print(f"\n  Accuracy by ELO gap:")
    print(f"  {'Gap':<22} {'n':>5} {'ELO Acc':>9} {'Upset rate':>11}")
    print(f"  {'-'*50}")
    for lo, hi, lbl in [(0,25,"0-25"),(25,50,"25-50"),(50,100,"50-100"),
                         (100,150,"100-150"),(150,250,"150-250"),(250,999,"250+")]:
        b = df[(df["elo_delta"] >= lo) & (df["elo_delta"] < hi)]
        if len(b) < 5: continue
        a = b["elo_pred_correct"].mean()
        print(f"  {lbl:<22} {len(b):>5} {a:>8.1%} {1-a:>11.1%}")

    # K factor distribution (for diminishing K configs)
    if cfg.k_floor < cfg.k_start:
        print(f"\n  K factor distribution (shows learning curve):")
        print(f"    Mean K used: {df['k_used'].mean():.1f}")
        print(f"    K range: {df['k_used'].min():.1f} – {df['k_used'].max():.1f}")

        # Show theoretical K curve
        print(f"\n  K by match count:")
        for n in [0, 10, 25, 50, 100, 200, 500]:
            print(f"    n={n:>4}  K_overall={cfg.k(n):.1f}  K_surface={cfg.k_surf(n):.1f}")

    # ELO distribution at end
    elos = list(elo_final.values())
    print(f"\n  Final ELO distribution ({len(elos)} players):")
    print(f"    Mean: {np.mean(elos):.0f}")
    print(f"    Median: {np.median(elos):.0f}")
    print(f"    Std: {np.std(elos):.0f}")
    print(f"    Top 10: {sorted(elos, reverse=True)[:5]}")
    print(f"    Bottom 5: {sorted(elos)[:5]}")


# -- MAIN ------------------------------------------------------

def main():
    print("\n" + "="*65)
    print("  ELO Recalibration Test")
    print("="*65)
    print("\nLoading match history...")

    matches = load_match_history()

    print(f"\nRunning {len(CONFIGS)} ELO configurations...")

    results_summary = []
    all_dfs = []

    for cfg in CONFIGS:
        print(f"\n  Running: {cfg.name}...")
        df, elo_final, n_matches = run_elo_system(matches, cfg)
        all_dfs.append(df)
        evaluate_system(df, cfg, elo_final, n_matches)
        results_summary.append({
            "config":   cfg.name,
            "accuracy": df["elo_pred_correct"].mean(),
            "brier":    ((df["exp_w"] - 1.0)**2).mean(),
            "n":        len(df),
        })

    # Head-to-head comparison
    print(f"\n{'='*65}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Config':<45} {'Accuracy':>9} {'Brier':>8}")
    print(f"  {'-'*63}")
    best_acc = max(r["accuracy"] for r in results_summary)
    for r in results_summary:
        flag = " <- best" if r["accuracy"] == best_acc else ""
        print(f"  {r['config']:<45} {r['accuracy']:>8.1%} {r['brier']:>8.4f}{flag}")

    # Anti-calibration check: does the new system fix it?
    print(f"\n  Anti-calibration check (80-90% confidence bucket):")
    print(f"  {'-'*50}")
    for cfg, df in zip(CONFIGS, all_dfs):
        df["elo_fav"] = df.apply(lambda r: max(r["exp_w"], 1-r["exp_w"]), axis=1)
        b = df[(df["elo_fav"] >= 0.80) & (df["elo_fav"] < 0.90)]
        if len(b) >= 5:
            actual = b["elo_pred_correct"].mean()
            expected = b["elo_fav"].mean()
            print(f"  {cfg.name[:44]:<44}  {actual:.1%} actual vs {expected:.1%} expected")
        else:
            print(f"  {cfg.name[:44]:<44}  insufficient data in 80-90% bucket")

    print(f"""
  Key questions answered by this test:
  1. Does base=1200 improve calibration vs base=1500?
  2. Does diminishing K fix the anti-calibration at high confidence?
  3. Which K decay rate works best?
  4. Does the ELO distribution look more realistic at end?

  If the new system shows improved calibration (actual ~ expected)
  and higher accuracy, it's worth integrating into the main pipeline.
""")


if __name__ == "__main__":
    main()
