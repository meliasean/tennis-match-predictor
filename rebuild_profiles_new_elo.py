"""
rebuild_profiles_new_elo.py
============================
Rebuilds ALL player profiles from scratch using the new ELO system:
  - Base ELO: 1200 (was 1500)
  - Diminishing K: starts at 40, decays to floor of 8 over ~60 matches
  - Surface K: starts at 28, decays to 6

Reads full match history from all *_predictions_cck_complete.csv files
in ./reports/ (Wimbledon 2025 through Madrid 2026) plus the raw 2021-2024
ATP match data used to train the RF model.

Run from your project root:
    python rebuild_profiles_new_elo.py

Outputs:
    ./reports/player_profiles_latest.csv          (updated)
    ./reports/player_profiles_post_madrid_2026.csv
"""

from __future__ import annotations

import glob
import os
import unicodedata
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPORTS_DIR  = "./reports"
DATA_DIR     = "./data"
OUT_LATEST   = f"{REPORTS_DIR}/player_profiles_latest.csv"
OUT_MADRID   = f"{REPORTS_DIR}/player_profiles_post_madrid_2026.csv"

# -- NEW ELO CONFIG ---------------------------------------------
@dataclass
class EloConfig:
    base:          float = 1200.0
    k_start:       float = 40.0
    k_floor:       float = 8.0
    k_decay:       float = 60.0
    k_surf_start:  float = 28.0
    k_surf_floor:  float = 6.0

    def k(self, n: int) -> float:
        return self.k_floor + (self.k_start - self.k_floor) * np.exp(-n / self.k_decay)

    def k_surf(self, n: int) -> float:
        return self.k_surf_floor + (self.k_surf_start - self.k_surf_floor) * np.exp(-n / self.k_decay)

CFG = EloConfig(
    base=1500.0,
    k_start=24.0,
    k_floor=24.0,
    k_decay=999.0,
    k_surf_start=18.0,
    k_surf_floor=18.0,
)

# -- ALIASES ----------------------------------------------------
ALIASES = {
    "Felix Auger-Aliassime":        "Felix Auger Aliassime",
    "Botic Van De Zandschulp":      "Botic van de Zandschulp",
    "Adolfo Daniel Vallejo":        "Diego Vallejo",
    "Jan Lennard Struff":           "Jan-Lennard Struff",
    "Otto Virtanen":                "Oscar Virtanen",
}
def alias(n: str) -> str:
    if not n or pd.isna(n): return str(n)
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
    return ALIASES.get(n.strip(), n.strip())

SURFACE_MAP = {"Clay":"Clay","Hard":"Hard","Grass":"Grass","Carpet":"Hard","Acrylic":"Hard"}
def norm_surf(s) -> str:
    if pd.isna(s): return "Hard"
    for k,v in SURFACE_MAP.items():
        if k.lower() in str(s).lower(): return v
    return "Hard"

# -- LOAD MATCH HISTORY -----------------------------------------

def load_from_predictions() -> pd.DataFrame:
    """Load match history from scored prediction files (Wimbledon 2025+)."""
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    rows = []
    for fpath in files:
        df = pd.read_csv(fpath)
        slug = os.path.basename(fpath).split("_predictions")[0]
        parts = slug.rsplit("_",1)
        tourney = parts[0] if len(parts)==2 else slug

        req = ["player_a","player_b","pred_winner","correct_prediction","date"]
        if not all(c in df.columns for c in req): continue

        df = df[pd.to_numeric(df["correct_prediction"], errors="coerce").notna()].copy()
        df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")

        surface = "Hard"
        if "surface" in df.columns and len(df):
            surface = norm_surf(df["surface"].iloc[0])

        for _, r in df.iterrows():
            pa   = alias(r["player_a"]); pb = alias(r["player_b"])
            pred = alias(r["pred_winner"]); cp = int(r["correct_prediction"])
            winner = pred if cp==1 else (pb if pred==pa else pa)
            loser  = pb   if winner==pa else pa
            rows.append({"date":r["date"],"winner":winner,"loser":loser,
                          "surface":surface,"tourney":tourney,"match_num":r.get("match_no",0)})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_from_raw_dataset() -> pd.DataFrame:
    """Load 2021-2024 match history from raw data files."""
    candidates = sorted(glob.glob(f"{DATA_DIR}/match_dataset*.csv"))
    if not candidates:
        candidates = sorted(glob.glob(f"{REPORTS_DIR}/match_dataset*.csv"))
    if not candidates:
        print("  WARNING: No raw match dataset found -- using prediction files only")
        return pd.DataFrame()

    # Use the most recent dataset
    fpath = candidates[-1]
    print(f"  Loading raw dataset: {fpath}")
    df = pd.read_csv(fpath)
    print(f"  Raw dataset: {len(df):,} rows, cols: {list(df.columns)[:8]}")

    # Standardise columns
    date_col   = next((c for c in ["date","tourney_date"] if c in df.columns), None)
    winner_col = next((c for c in ["winner_name","p1_name"] if c in df.columns), None)
    loser_col  = next((c for c in ["loser_name","p2_name"] if c in df.columns), None)
    surf_col   = next((c for c in ["surface"] if c in df.columns), None)
    tour_col   = next((c for c in ["tourney_name","tourney_slug"] if c in df.columns), None)

    if not all([date_col, winner_col, loser_col]):
        print(f"  WARNING: Raw dataset missing required columns")
        return pd.DataFrame()

    out = pd.DataFrame({
        "date":    pd.to_datetime(df[date_col], errors="coerce"),
        "winner":  df[winner_col].apply(alias),
        "loser":   df[loser_col].apply(alias),
        "surface": df[surf_col].apply(norm_surf) if surf_col else "Hard",
        "tourney": df[tour_col] if tour_col else "historical",
        "match_num": df.get("match_num", np.arange(len(df))),
    })
    # For p1_name/p2_name format, y=1 means p1 won
    if winner_col == "p1_name" and "y" in df.columns:
        y = pd.to_numeric(df["y"], errors="coerce")
        out["winner"] = np.where(y==1, df["p1_name"].apply(alias), df["p2_name"].apply(alias))
        out["loser"]  = np.where(y==1, df["p2_name"].apply(alias), df["p1_name"].apply(alias))

    return out.dropna(subset=["date","winner","loser"]).sort_values("date").reset_index(drop=True)


def build_combined_history() -> pd.DataFrame:
    """Combine raw 2021-2024 data with 2025-2026 prediction data."""
    raw  = load_from_raw_dataset()
    pred = load_from_predictions()

    if raw.empty and pred.empty:
        raise ValueError("No match data found")

    all_matches = pd.concat([raw, pred], ignore_index=True) if not raw.empty else pred
    all_matches = all_matches.drop_duplicates(subset=["date","winner","loser"])
    all_matches = all_matches.sort_values(["date","tourney","match_num"], kind="mergesort").reset_index(drop=True)

    print(f"\n  Combined match history: {len(all_matches):,} matches")
    print(f"  Date range: {all_matches['date'].min().date()} -> {all_matches['date'].max().date()}")
    print(f"  Players: {pd.concat([all_matches['winner'],all_matches['loser']]).nunique():,} unique")
    return all_matches


# -- NEW ELO ENGINE ---------------------------------------------

def run_new_elo(matches: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Run new ELO system (base=1200, diminishing K) through all matches.
    Returns per-match log, final ELO dict, final surface ELO dict.
    """
    elo:   Dict[str, float] = {}
    selo:  Dict[Tuple[str,str], float] = {}
    n_m:   Dict[str, int] = {}  # match count per player
    rows:  List[Dict] = []

    for _, r in matches.iterrows():
        w = str(r["winner"]); l = str(r["loser"]); s = norm_surf(r["surface"])

        # Init new players at base=1200
        for p in [w, l]:
            if p not in elo:
                elo[p] = CFG.base; n_m[p] = 0
            for surf in ["Clay","Hard","Grass"]:
                if (p, surf) not in selo:
                    selo[(p, surf)] = CFG.base

        # Pre-match values
        ew = elo[w]; el = elo[l]
        es_w = selo[(w,s)]; es_l = selo[(l,s)]

        # Expected score
        exp_w   = 1.0 / (1.0 + 10**((el  - ew ) / 400))
        exp_w_s = 1.0 / (1.0 + 10**((es_l - es_w) / 400))

        # Diminishing K
        kw = CFG.k(n_m[w]); kl = CFG.k(n_m[l])
        k  = (kw + kl) / 2
        ks_w = CFG.k_surf(n_m[w]); ks_l = CFG.k_surf(n_m[l])
        ks = (ks_w + ks_l) / 2

        rows.append({
            "player": w, "opp": l, "date": r["date"], "surface": s,
            "is_win": 1, "pre_elo": ew, "post_elo": ew + k*(1-exp_w),
            "pre_selo": es_w, "post_selo": es_w + ks*(1-exp_w_s),
            "n_matches": n_m[w], "k_used": k,
        })
        rows.append({
            "player": l, "opp": w, "date": r["date"], "surface": s,
            "is_win": 0, "pre_elo": el, "post_elo": el - k*(1-exp_w),
            "pre_selo": es_l, "post_selo": es_l - ks*(1-exp_w_s),
            "n_matches": n_m[l], "k_used": k,
        })

        # Update
        elo[w] = ew + k*(1-exp_w);   elo[l] = el - k*(1-exp_w)
        selo[(w,s)] = es_w + ks*(1-exp_w_s)
        selo[(l,s)] = es_l - ks*(1-exp_w_s)
        n_m[w] += 1; n_m[l] += 1

    log = pd.DataFrame(rows).sort_values(["player","date"], kind="mergesort").reset_index(drop=True)
    return log, elo, selo, n_m


# -- PROFILE BUILDER --------------------------------------------

def _smooth(old, x, alpha):
    try: old = float(old)
    except: old = 0.5
    if pd.isna(old): old = 0.5
    return float((1-alpha)*old + alpha*x)


def build_profiles(matches: pd.DataFrame, elo_log: pd.DataFrame,
                   elo_final: Dict, selo_final: Dict, n_matches: Dict) -> pd.DataFrame:
    """Build player profiles from scratch using the new ELO system."""
    players = pd.concat([matches["winner"], matches["loser"]]).unique()
    print(f"\n  Building profiles for {len(players):,} players...")

    profiles = {}
    for p in players:
        profiles[p] = {
            "name": p,
            "current_elo": elo_final.get(p, CFG.base),
            "peak_elo": CFG.base,
            "selo_Clay":  selo_final.get((p,"Clay"),  CFG.base),
            "selo_Grass": selo_final.get((p,"Grass"), CFG.base),
            "selo_Hard":  selo_final.get((p,"Hard"),  CFG.base),
            "wr_Clay": 0.5, "wr_Grass": 0.5, "wr_Hard": 0.5,
            "overall_wr": 0.5, "form10_wr": 0.5, "form5_wr": 0.5,
            "streak": 0.0, "avg_rest_days": 20.0,
            "matches_28d": 0.0, "last_match_date": pd.NaT,
            "n_matches": n_matches.get(p, 0),
        }

    # Walk through all matches chronologically to compute rolling stats
    per_player: List[Tuple] = []
    for _, r in matches.iterrows():
        per_player.append((r["winner"], r["date"], 1, norm_surf(r["surface"])))
        per_player.append((r["loser"],  r["date"], 0, norm_surf(r["surface"])))

    pl = (pd.DataFrame(per_player, columns=["player","date","is_win","surface"])
            .sort_values(["player","date"], kind="mergesort")
            .reset_index(drop=True))

    for p, sub in pl.groupby("player", sort=False):
        if p not in profiles: continue
        prof = profiles[p]
        cur = 0.0
        last_dt = None
        avg_rest = 20.0; matches_28d = 0.0
        wr = 0.5; f10 = 0.5; f5 = 0.5
        peak = prof["current_elo"]

        for _, rr in sub.iterrows():
            dt = rr["date"]; iw = int(rr["is_win"]); s = rr["surface"]

            # Streak
            cur = (cur+1) if (iw==1 and cur>=0) else (1 if iw==1 else (cur-1 if cur<=0 else -1))

            # Rest days + 28d match count
            if last_dt is not None and pd.notna(last_dt):
                gap = float(np.clip((dt - last_dt).days, 0, 60))
                avg_rest = 0.90*avg_rest + 0.10*gap
                decay = float(np.exp(-gap/28.0))
                matches_28d = matches_28d*decay + 1.0
            else:
                matches_28d += 1.0

            # Win rates
            wr  = _smooth(wr,  iw, 0.06)
            f10 = _smooth(f10, iw, 0.12)
            f5  = _smooth(f5,  iw, 0.20)

            # Surface WR
            key = f"wr_{s}"
            if key in prof: prof[key] = _smooth(prof[key], iw, 0.15)

            last_dt = dt

        prof["streak"]        = float(cur)
        prof["avg_rest_days"] = float(np.clip(avg_rest, 0, 60))
        prof["matches_28d"]   = float(np.clip(matches_28d, 0, 60))
        prof["overall_wr"]    = float(np.clip(wr,  0, 1))
        prof["form10_wr"]     = float(np.clip(f10, 0, 1))
        prof["form5_wr"]      = float(np.clip(f5,  0, 1))
        if last_dt is not None: prof["last_match_date"] = pd.to_datetime(last_dt)

    # Peak ELO from log
    for p, sub in elo_log[elo_log["is_win"]==1].groupby("player"):
        if p in profiles:
            profiles[p]["peak_elo"] = float(sub["post_elo"].max())

    # Build dataframe
    cols = ["name","last_match_date","current_elo","peak_elo",
            "selo_Clay","selo_Grass","selo_Hard",
            "wr_Clay","wr_Grass","wr_Hard",
            "overall_wr","form10_wr","form5_wr",
            "streak","avg_rest_days","matches_28d","n_matches"]
    df = pd.DataFrame(list(profiles.values()))
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    df = df[cols].sort_values("current_elo", ascending=False).reset_index(drop=True)
    df["last_match_date"] = pd.to_datetime(df["last_match_date"], errors="coerce")
    return df


# -- MAIN -------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  Player Profile Rebuild -- New ELO System")
    print(f"  Base: {CFG.base}  K_start: {CFG.k_start}  K_floor: {CFG.k_floor}  K_decay: {CFG.k_decay}")
    print("="*60)

    # Load full match history
    print("\nLoading match history...")
    matches = build_combined_history()

    # Run new ELO
    print("\nRunning new ELO system...")
    elo_log, elo_final, selo_final, n_matches = run_new_elo(matches)
    print(f"  Processed {len(elo_log)//2:,} matches, {len(elo_final):,} players")

    # ELO distribution sanity check
    elos = list(elo_final.values())
    print(f"\n  ELO distribution (should be centred ~1200):")
    print(f"    Mean:   {np.mean(elos):.0f}")
    print(f"    Median: {np.median(elos):.0f}")
    print(f"    Std:    {np.std(elos):.0f}")
    print(f"    Max:    {np.max(elos):.0f} ({max(elo_final, key=elo_final.get)})")
    print(f"    Top 10: {sorted([(k,v) for k,v in elo_final.items()],key=lambda x:-x[1])[:10]}")

    # Build profiles
    profiles = build_profiles(matches, elo_log, elo_final, selo_final, n_matches)

    # Spot checks
    print(f"\n  Spot checks (top players):")
    for name in ["Jannik Sinner","Carlos Alcaraz","Alexander Zverev","Novak Djokovic",
                 "Daniil Medvedev","Arthur Fils","Alexander Blockx","Casper Ruud"]:
        row = profiles[profiles["name"]==name]
        if not row.empty:
            r = row.iloc[0]
            print(f"    {name:<28} ELO={r['current_elo']:>7.0f}  Clay={r['selo_Clay']:>7.0f}  "
                  f"form={r['form10_wr']:.2f}  streak={int(r['streak']):>+3}  n={int(r['n_matches'])}")
        else:
            print(f"    {name:<28} NOT FOUND")

    # Save
    os.makedirs(REPORTS_DIR, exist_ok=True)
    profiles.drop(columns=["n_matches"], errors="ignore").to_csv(OUT_LATEST, index=False, encoding="utf-8-sig")
    profiles.drop(columns=["n_matches"], errors="ignore").to_csv(OUT_MADRID,  index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {OUT_LATEST} ({len(profiles)} players)")
    print(f"  Saved: {OUT_MADRID}")
    print(f"\n  Done. Commit with:")
    print(f"  git add reports/player_profiles_latest.csv reports/player_profiles_post_madrid_2026.csv")
    print(f"  git commit -m 'rebuild: player profiles with new ELO (base=1200, diminishing K)'")


if __name__ == "__main__":
    main()
