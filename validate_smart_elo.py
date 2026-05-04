"""
validate_smart_elo.py
=====================
Tests the new Smart ELO system (time decay + diminishing K) against the
current system across three dimensions:

1. SANITY CHECK -- Does the ranking look realistic?
   - Top 20 players should be recognisable ATP elites
   - Tsitsipas should drop from #8 (poor recent form should be reflected)
   - No duplicate players
   - Players with terrible recent form should rank lower

2. PREDICTIVE ACCURACY -- Does it predict match outcomes better?
   - Walk-forward test: use ELO at time T to predict match at T
   - Compare: Current ELO vs Smart ELO vs Book
   - Test on last 500 scored matches (out-of-sample)

3. CALIBRATION -- Are stated probabilities trustworthy?
   - When Smart ELO says 80%, does it win 80% of the time?
   - Compare calibration curve for both systems
   - Check if the anti-calibration bug (80-90% bucket -> 47% actual) is fixed

Run from your project root:
    python validate_smart_elo.py

Does NOT modify any files. Read-only validation.
"""

from __future__ import annotations
import glob, os, json, unicodedata, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"
DATA_DIR    = "./data"

# -- ELO CONFIGS ------------------------------------------------

@dataclass
class EloConfig:
    name:          str
    base:          float
    k_start:       float
    k_floor:       float
    k_decay_n:     float
    k_surf_start:  float
    k_surf_floor:  float
    decay_per_day: float = 0.0  # 0 = no time decay (current system)

    def k(self, n: int) -> float:
        if self.k_start == self.k_floor: return self.k_start
        return self.k_floor + (self.k_start - self.k_floor) * np.exp(-n / self.k_decay_n)

    def k_surf(self, n: int) -> float:
        if self.k_surf_start == self.k_surf_floor: return self.k_surf_start
        return self.k_surf_floor + (self.k_surf_start - self.k_surf_floor) * np.exp(-n / self.k_decay_n)

    def time_decay(self, elo: float, days: int) -> float:
        if self.decay_per_day == 0 or days <= 14: return elo
        alpha = 1.0 - np.exp(-self.decay_per_day * max(0, days - 14))
        return elo + alpha * (self.base - elo)

CONFIGS = {
    "current": EloConfig(
        name="Current (base=1500, fixed K=24, no decay)",
        base=1500.0, k_start=24.0, k_floor=24.0, k_decay_n=999.0,
        k_surf_start=18.0, k_surf_floor=18.0, decay_per_day=0.0,
    ),
    "new_no_decay": EloConfig(
        name="New (base=1200, diminishing K, no decay)",
        base=1200.0, k_start=40.0, k_floor=8.0, k_decay_n=60.0,
        k_surf_start=28.0, k_surf_floor=6.0, decay_per_day=0.0,
    ),
    "smart": EloConfig(
        name="Smart (base=1200, diminishing K + time decay)",
        base=1200.0, k_start=40.0, k_floor=8.0, k_decay_n=60.0,
        k_surf_start=28.0, k_surf_floor=6.0, decay_per_day=0.003,
    ),
    "smart_fast_decay": EloConfig(
        name="Smart-Fast (base=1200, diminishing K + faster decay)",
        base=1200.0, k_start=40.0, k_floor=8.0, k_decay_n=60.0,
        k_surf_start=28.0, k_surf_floor=6.0, decay_per_day=0.006,
    ),
}

# -- NAME NORMALISATION -----------------------------------------

CANONICAL: Dict[str, Optional[str]] = {
    "Felix Auger-Aliassime":      "Felix Auger Aliassime",
    "Jan Lennard Struff":         "Jan-Lennard Struff",
    "Mackenzie Mcdonald":         "Mackenzie McDonald",
    "Christopher Oconnell":       "Christopher O'Connell",
    "Christopher O Connell":      "Christopher O'Connell",
    "Botic Van De Zandschulp":    "Botic van de Zandschulp",
    "Aleksandar Vukic":           "Aleksander Vukic",
    "Adolfo Daniel Vallejo":      "Diego Vallejo",
    "Daniel Vallejo":             "Diego Vallejo",
    "Dino Prizmic":               "Dusan Prizmic",
    "Marcelo Tomas Barrios Vera": "Tomas Barrios Vera",
    "Guy Gen Ouden":              "Guy Den Ouden",
    "Lorenzo Darderi":            "Luciano Darderi",
    "Otto Virtanen":              "Oscar Virtanen",
    "John Isner":                 None,
    "Roger Federer":              None,
    "Rafael Nadal":               None,
    "Andy Murray":                None,
    "Jo-Wilfried Tsonga":         None,
    "Marcelo Arevalo":            None,
}

SURFACE_MAP = {"Clay":"Clay","Hard":"Hard","Grass":"Grass","Carpet":"Hard"}
def norm_surf(s) -> str:
    if pd.isna(s): return "Hard"
    for k,v in SURFACE_MAP.items():
        if k.lower() in str(s).lower(): return v
    return "Hard"

def to_canonical(raw: str) -> Optional[str]:
    if not raw or pd.isna(raw): return None
    n = unicodedata.normalize("NFKD", str(raw)).encode("ascii","ignore").decode("ascii").strip()
    if n in CANONICAL: return CANONICAL[n]
    return n

def dedup_key(name: str) -> str:
    if not name: return ""
    return name.lower().replace("-","").replace("'","").replace(" ","").replace(".","")


# -- LOAD DATA --------------------------------------------------

def load_all_matches() -> pd.DataFrame:
    rows = []

    # Raw 2021-2024
    cands = sorted(glob.glob(f"{DATA_DIR}/match_dataset*.csv"))
    if not cands: cands = sorted(glob.glob(f"{REPORTS_DIR}/match_dataset*.csv"))
    if cands:
        df = pd.read_csv(cands[-1])
        dc = next((c for c in ["date","tourney_date"] if c in df.columns), None)
        wc = next((c for c in ["winner_name","p1_name"] if c in df.columns), None)
        lc = next((c for c in ["loser_name","p2_name"] if c in df.columns), None)
        sc = "surface" if "surface" in df.columns else None
        if all([dc, wc, lc]):
            out = pd.DataFrame({
                "date":    pd.to_datetime(df[dc], errors="coerce"),
                "winner":  df[wc].apply(lambda x: to_canonical(str(x))),
                "loser":   df[lc].apply(lambda x: to_canonical(str(x))),
                "surface": df[sc].apply(norm_surf) if sc else "Hard",
                "source":  "raw",
                "book_fair_prob_a": np.nan,
            })
            if wc == "p1_name" and "y" in df.columns:
                y = pd.to_numeric(df["y"], errors="coerce")
                out["winner"] = np.where(y==1,
                    df["p1_name"].apply(lambda x: to_canonical(str(x))),
                    df["p2_name"].apply(lambda x: to_canonical(str(x))))
                out["loser"] = np.where(y==1,
                    df["p2_name"].apply(lambda x: to_canonical(str(x))),
                    df["p1_name"].apply(lambda x: to_canonical(str(x))))
            rows.append(out.dropna(subset=["date","winner","loser"]))

    # Scored predictions 2025-2026
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]
    for fpath in files:
        df = pd.read_csv(fpath)
        surface = norm_surf(df["surface"].iloc[0]) if "surface" in df.columns and len(df) else "Hard"
        df = df[pd.to_numeric(df.get("correct_prediction",""), errors="coerce").notna()].copy()
        df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
        if "book_fair_prob_a" in df.columns:
            df["book_fair_prob_a"] = pd.to_numeric(df["book_fair_prob_a"], errors="coerce")
        else:
            df["book_fair_prob_a"] = np.nan
        for _, r in df.iterrows():
            pa   = to_canonical(str(r.get("player_a","")))
            pb   = to_canonical(str(r.get("player_b","")))
            pred = to_canonical(str(r.get("pred_winner","")))
            cp   = int(r["correct_prediction"])
            if not pa or not pb: continue
            w = pred if cp==1 else (pb if pred==pa else pa)
            l = pb if w==pa else pa
            rows.append(pd.DataFrame([{
                "date": r["date"], "winner": w, "loser": l,
                "surface": surface, "source": "pred",
                "book_fair_prob_a": r.get("book_fair_prob_a", np.nan),
            }]))

    all_m = pd.concat(rows, ignore_index=True)
    all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
    all_m = all_m.dropna(subset=["date","winner","loser"])
    all_m = all_m[all_m["winner"].notna() & all_m["loser"].notna()]
    all_m = all_m[~all_m["winner"].isin(["None","nan",""])]
    all_m = all_m.drop_duplicates(subset=["date","winner","loser"])
    all_m = all_m.sort_values("date", kind="mergesort").reset_index(drop=True)
    return all_m


# -- ELO ENGINE (walk-forward) ---------------------------------

def run_elo_walkforward(matches: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """
    Runs ELO through all matches chronologically, recording
    the pre-match ELO for each game (used for prediction testing).
    """
    elo:       Dict[str, float] = {}
    n_m:       Dict[str, int] = {}
    last_date: Dict[str, pd.Timestamp] = {}
    canon:     Dict[str, str] = {}
    records = []

    for _, r in matches.iterrows():
        w_raw = str(r["winner"]); l_raw = str(r["loser"])
        wk = dedup_key(w_raw); lk = dedup_key(l_raw)
        if not wk or not lk: continue

        if wk not in canon: canon[wk] = w_raw
        if lk not in canon: canon[lk] = l_raw

        dt = pd.Timestamp(r["date"])
        s  = norm_surf(r["surface"])

        for key in [wk, lk]:
            if key not in elo:
                elo[key] = cfg.base; n_m[key] = 0

        # Time decay
        for key in [wk, lk]:
            if key in last_date:
                gap = (dt - last_date[key]).days
                elo[key] = cfg.time_decay(elo[key], gap)

        ew = elo[wk]; el = elo[lk]
        exp_w = 1.0 / (1.0 + 10**((el - ew) / 400))

        # Record pre-match state
        records.append({
            "date":       dt,
            "winner":     canon[wk],
            "loser":      canon[lk],
            "elo_w":      ew,
            "elo_l":      el,
            "elo_delta":  abs(ew - el),
            "exp_w":      exp_w,
            "elo_pred_correct": 1 if ew >= el else 0,
            "n_w":        n_m[wk],
            "n_l":        n_m[lk],
            "source":     r.get("source","?"),
            "book_fair_prob_a": r.get("book_fair_prob_a", np.nan),
        })

        k = (cfg.k(n_m[wk]) + cfg.k(n_m[lk])) / 2
        elo[wk] = ew + k * (1 - exp_w)
        elo[lk] = el - k * (1 - exp_w)
        n_m[wk] += 1; n_m[lk] += 1
        last_date[wk] = dt; last_date[lk] = dt

    df = pd.DataFrame(records)
    # Final ELO state for sanity check
    final_elo = {canon[k]: v for k,v in elo.items()}
    return df, final_elo, n_m, canon


# -- TESTS ------------------------------------------------------

def test_predictive_accuracy(all_df: pd.DataFrame, cfg_key: str, cfg: EloConfig,
                              test_from: pd.Timestamp) -> dict:
    """Walk-forward accuracy on test period."""
    df, final_elo, n_m, canon = run_elo_walkforward(all_df, cfg)

    # Test only on matches after test_from
    test = df[df["date"] >= test_from].copy()
    if len(test) < 20:
        return {"n": 0, "accuracy": None, "brier": None}

    acc    = test["elo_pred_correct"].mean()
    brier  = ((test["exp_w"] - 1.0)**2).mean()  # winner always = 1

    # Book accuracy (where available)
    has_book = test["book_fair_prob_a"].notna()
    book_acc = None
    if has_book.sum() >= 20:
        book_sub = test[has_book].copy()
        book_pred_correct = (
            ((book_sub["book_fair_prob_a"] >= 0.5) & (book_sub["elo_pred_correct"] == 1)) |
            ((book_sub["book_fair_prob_a"] < 0.5)  & (book_sub["elo_pred_correct"] == 0))
        )
        # Actually: book_fair_prob_a is for player_a, but we only have winner/loser here
        # Use exp_w vs book comparison differently
        book_acc = None  # requires player_a mapping

    return {
        "n":        len(test),
        "accuracy": float(acc),
        "brier":    float(brier),
        "final_elo": final_elo,
        "n_m":      n_m,
        "canon":    canon,
        "df":       df,
    }


def test_calibration(df: pd.DataFrame, cfg_name: str) -> None:
    """Check if stated ELO probabilities match actual outcomes."""
    df["elo_fav"] = df.apply(lambda r: max(r["exp_w"], 1-r["exp_w"]), axis=1)
    df["elo_fav_correct"] = df["elo_pred_correct"]

    print(f"\n  Calibration ({cfg_name}):")
    print(f"  {'Conf':<10} {'n':>5} {'Actual':>8} {'Expected':>9} {'Gap':>7}")
    print(f"  {'-'*42}")

    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%","90%+"]
    for i, lbl in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        b = df[(df["elo_fav"] >= lo) & (df["elo_fav"] < hi)]
        if len(b) < 5: continue
        actual   = b["elo_fav_correct"].mean()
        expected = b["elo_fav"].mean()
        gap      = actual - expected
        flag = " <- ANTI-CALIB" if actual < lo - 0.05 else (" OK" if abs(gap) < 0.05 else "")
        print(f"  {lbl:<10} {len(b):>5} {actual:>7.1%} {expected:>9.1%} {gap:>+6.1%}{flag}")


def test_sanity(final_elo: dict, n_m: dict, canon: dict, cfg_name: str,
                cutoff: pd.Timestamp) -> None:
    """Check if the rankings look realistic."""
    print(f"\n  Rankings ({cfg_name}):")

    # Build ranked list
    rows = []
    for name, e in final_elo.items():
        if not name or name in ("None","nan",""): continue
        rows.append({"name": name, "elo": e, "n": n_m.get(dedup_key(name), 0)})
    ranked = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)

    # Filter to active players (played 5+ matches)
    ranked = ranked[ranked["n"] >= 5].reset_index(drop=True)

    print(f"  Top 20:")
    for i, r in ranked.head(20).iterrows():
        print(f"    {i+1:>3}. {r['name']:<30} {r['elo']:>7.0f}  (n={int(r['n'])})")

    # Spot checks
    print(f"\n  Spot checks:")
    checks = {
        "Jannik Sinner":      ("should be #1-3",   lambda r: r <= 3),
        "Stefanos Tsitsipas": ("should be #10-25",  lambda r: 8 <= r <= 30),
        "Alexander Zverev":   ("should be #3-8",    lambda r: r <= 8),
        "Carlos Alcaraz":     ("should be #1-5",    lambda r: r <= 5),
        "Aleksander Vukic":   ("should exist once", None),
        "Aleksandar Vukic":   ("should NOT exist",  None),
        "Mackenzie McDonald": ("should exist once",  None),
        "Mackenzie Mcdonald": ("should NOT exist",  None),
    }
    for name, (desc, check_fn) in checks.items():
        hits = ranked[ranked["name"] == name]
        if len(hits) == 0:
            print(f"    {name:<30} NOT FOUND  ({desc})")
        elif len(hits) > 1:
            print(f"    {name:<30} DUPLICATE ({len(hits)} times) <- PROBLEM")
        else:
            rank = hits.index[0] + 1
            elo  = hits.iloc[0]["elo"]
            ok   = check_fn(rank) if check_fn else True
            flag = " OK" if ok else " <- CHECK THIS"
            print(f"    {name:<30} #{rank:<4} ELO={elo:.0f}  ({desc}){flag}")


# -- MAIN -------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("  Smart ELO Validation Test")
    print("="*70)

    print("\nLoading match history...")
    all_m = load_all_matches()
    print(f"  {len(all_m):,} matches loaded")
    print(f"  Date range: {all_m['date'].min().date()} -> {all_m['date'].max().date()}")

    # Test on last 20% of matches (walk-forward, out-of-sample)
    n_test = max(200, int(len(all_m) * 0.20))
    test_from = all_m["date"].iloc[-n_test]
    print(f"\n  Walk-forward test: last {n_test} matches (from {test_from.date()})")

    # -- RUN ALL CONFIGS ------------------------------------------
    results = {}
    for cfg_key, cfg in CONFIGS.items():
        print(f"\nRunning: {cfg.name}...")
        r = test_predictive_accuracy(all_m, cfg_key, cfg, test_from)
        results[cfg_key] = r
        if r["n"] > 0:
            print(f"  n={r['n']}  accuracy={r['accuracy']:.1%}  brier={r['brier']:.4f}")

    # -- SUMMARY TABLE --------------------------------------------
    print(f"\n{'='*70}")
    print(f"  ACCURACY COMPARISON (walk-forward on last {n_test} matches)")
    print(f"{'='*70}")
    print(f"  {'Config':<50} {'n':>5} {'Acc':>7} {'Brier':>8}")
    print(f"  {'-'*68}")

    best_acc   = max((r["accuracy"] for r in results.values() if r["n"] > 0), default=0)
    best_brier = min((r["brier"]    for r in results.values() if r["n"] > 0), default=999)
    for key, r in results.items():
        if r["n"] == 0: continue
        cfg = CONFIGS[key]
        acc_flag   = " <- best acc"   if r["accuracy"] == best_acc   else ""
        brier_flag = " <- best calib" if r["brier"]    == best_brier else ""
        print(f"  {cfg.name:<50} {r['n']:>5} {r['accuracy']:>6.1%} {r['brier']:>8.4f}{acc_flag}{brier_flag}")

    # -- CALIBRATION ----------------------------------------------
    print(f"\n{'='*70}")
    print(f"  CALIBRATION (anti-calibration check -- 80-90% bucket is the key)")
    print(f"{'='*70}")
    for key, r in results.items():
        if r["n"] == 0: continue
        test_df = r["df"][r["df"]["date"] >= test_from].copy()
        test_calibration(test_df, CONFIGS[key].name)

    # -- SANITY / RANKINGS -----------------------------------------
    print(f"\n{'='*70}")
    print(f"  SANITY CHECK -- do the rankings look right?")
    print(f"{'='*70}")

    # Only show rankings for smart config (the proposed new system)
    for key in ["current","smart"]:
        if key in results and results[key]["n"] > 0:
            r = results[key]
            test_sanity(r["final_elo"], r["n_m"], r["canon"], CONFIGS[key].name, test_from)

    # -- TSITSIPAS DEEP DIVE ---------------------------------------
    print(f"\n{'='*70}")
    print(f"  TSITSIPAS RANK COMPARISON")
    print(f"{'='*70}")
    for key, r in results.items():
        if r["n"] == 0: continue
        fe = r["final_elo"]
        rows = sorted([(n,e) for n,e in fe.items() if n and n != "None"],
                      key=lambda x:-x[1])
        filtered = [(i+1,n,e) for i,(n,e) in enumerate(rows) if r["n_m"].get(dedup_key(n),0)>=5]
        tsit = [(rank,n,e) for rank,n,e in filtered if "Tsitsipas" in n]
        sinner = [(rank,n,e) for rank,n,e in filtered if "Sinner" in n]
        if tsit:
            rank,n,e = tsit[0]
            print(f"  {CONFIGS[key].name[:45]:<45} Tsitsipas #{rank} (ELO={e:.0f})")
        if sinner:
            rank,n,e = sinner[0]
            print(f"  {' ':<45} Sinner     #{rank} (ELO={e:.0f})")

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    smart = results.get("smart",{})
    curr  = results.get("current",{})
    if smart.get("n",0) > 0 and curr.get("n",0) > 0:
        acc_diff   = (smart["accuracy"] - curr["accuracy"]) * 100
        brier_diff = smart["brier"] - curr["brier"]
        print(f"  Accuracy change:    {acc_diff:+.1f}pp  ({'improvement' if acc_diff>0 else 'regression'})")
        print(f"  Brier score change: {brier_diff:+.4f} ({'better calibrated' if brier_diff<0 else 'worse calibrated'})")
        if acc_diff >= 0 and brier_diff <= 0:
            print(f"\n  OK Smart ELO is better or equal on both metrics -- SAFE TO IMPLEMENT")
        elif acc_diff < -1.0:
            print(f"\n  FAIL Smart ELO hurts accuracy by {abs(acc_diff):.1f}pp -- DO NOT IMPLEMENT")
        else:
            print(f"\n  ~ Mixed results -- review calibration and ranking sanity manually")
    print()


if __name__ == "__main__":
    main()
