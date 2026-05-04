"""
test_weighted_elo.py
=====================
Tests a Weighted ELO system that addresses the core problem:
ELO treats all wins/losses equally regardless of when, where, or against whom.

Improvements tested:
  - Round multiplier: SF/F wins count more than R64 wins
  - Tournament multiplier: Slams > Masters > 500 > 250
  - Recency window: matches >24 months old count at half weight (K halved)
  - Opponent quality: already implicit in ELO, but enhanced by K scaling

Tests 4 configs against current system. Read-only -- modifies nothing.

Run: python test_weighted_elo.py
"""

from __future__ import annotations
import glob, os, unicodedata, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPORTS_DIR = "./reports"
DATA_DIR    = "./data"

# -- MULTIPLIERS ------------------------------------------------

# Round multipliers -- later rounds count more
ROUND_MULT = {
    "F":    1.50,
    "SF":   1.35,
    "QF":   1.20,
    "R16":  1.10,
    "R32":  1.00,
    "R64":  0.90,
    "R128": 0.80,
    "RR":   1.00,  # round robin
}

# Tournament level multipliers
LEVEL_MULT = {
    "G":  1.50,   # Grand Slam
    "M":  1.30,   # Masters 1000
    "A":  1.10,   # ATP 500
    "B":  1.00,   # ATP 250
    "D":  0.90,   # Davis Cup
    "F":  1.20,   # ATP Finals
    "PM": 1.20,   # Paris Masters (sometimes coded differently)
}

def round_mult(rnd: str) -> float:
    if not rnd or pd.isna(rnd): return 1.0
    rnd = str(rnd).strip().upper()
    return ROUND_MULT.get(rnd, 1.0)

def level_mult(level: str) -> float:
    if not level or pd.isna(level): return 1.0
    level = str(level).strip().upper()
    return LEVEL_MULT.get(level, 1.0)


# -- ELO CONFIGS ------------------------------------------------

@dataclass
class WeightedEloConfig:
    name:            str
    base:            float = 1200.0
    k_start:         float = 40.0
    k_floor:         float = 8.0
    k_decay_n:       float = 60.0
    k_surf_start:    float = 28.0
    k_surf_floor:    float = 6.0
    use_round_mult:  bool  = False
    use_level_mult:  bool  = False
    recency_months:  int   = 0   # 0 = no recency window; 24 = halve K for matches >24mo old
    recency_weight:  float = 0.5 # weight for old matches when recency_months > 0

    def k(self, n: int) -> float:
        return self.k_floor + (self.k_start - self.k_floor) * np.exp(-n / self.k_decay_n)

    def k_surf(self, n: int) -> float:
        return self.k_surf_floor + (self.k_surf_start - self.k_surf_floor) * np.exp(-n / self.k_decay_n)

    def effective_k(self, n: int, rnd: str, level: str,
                    match_date: pd.Timestamp, today: pd.Timestamp) -> float:
        k = self.k(n)
        if self.use_round_mult:  k *= round_mult(rnd)
        if self.use_level_mult:  k *= level_mult(level)
        if self.recency_months > 0:
            months_old = (today - match_date).days / 30.44
            if months_old > self.recency_months:
                k *= self.recency_weight
        return k


CONFIGS = {
    "current": WeightedEloConfig(
        name="Current (base=1500, fixed K=24)",
        base=1500.0, k_start=24.0, k_floor=24.0, k_decay_n=999.0,
        k_surf_start=18.0, k_surf_floor=18.0,
    ),
    "base_deployed": WeightedEloConfig(
        name="Deployed (base=1200, diminishing K)",
        base=1200.0,
    ),
    "round_weighted": WeightedEloConfig(
        name="Round-weighted (SF/F count more)",
        base=1200.0, use_round_mult=True,
    ),
    "level_weighted": WeightedEloConfig(
        name="Level-weighted (Slams/Masters count more)",
        base=1200.0, use_level_mult=True,
    ),
    "round_level": WeightedEloConfig(
        name="Round + Level weighted",
        base=1200.0, use_round_mult=True, use_level_mult=True,
    ),
    "recency_24mo": WeightedEloConfig(
        name="Recency window (matches >24mo at 50% K)",
        base=1200.0, recency_months=24, recency_weight=0.5,
    ),
    "full_weighted": WeightedEloConfig(
        name="Full: Round + Level + Recency 24mo",
        base=1200.0, use_round_mult=True, use_level_mult=True,
        recency_months=24, recency_weight=0.5,
    ),
}


# -- NAME NORMALISATION -----------------------------------------

CANONICAL = {
    "Felix Auger-Aliassime":      "Felix Auger Aliassime",
    "Jan Lennard Struff":         "Jan-Lennard Struff",
    "Mackenzie Mcdonald":         "Mackenzie McDonald",
    "Christopher Oconnell":       "Christopher O'Connell",
    "Botic Van De Zandschulp":    "Botic van de Zandschulp",
    "Aleksandar Vukic":           "Aleksander Vukic",
    "Adolfo Daniel Vallejo":      "Diego Vallejo",
    "Dino Prizmic":               "Dusan Prizmic",
    "Lorenzo Darderi":            "Luciano Darderi",
    "Guy Gen Ouden":              "Guy Den Ouden",
    "John Isner":                 None,
    "Roger Federer":              None,
    "Rafael Nadal":               None,
    "Andy Murray":                None,
    "Marcelo Arevalo":            None,
}
SURFACE_MAP = {"Clay":"Clay","Hard":"Hard","Grass":"Grass","Carpet":"Hard"}

def norm_surf(s) -> str:
    if pd.isna(s): return "Hard"
    for k,v in SURFACE_MAP.items():
        if k.lower() in str(s).lower(): return v
    return "Hard"

def to_canon(raw) -> Optional[str]:
    if not raw or pd.isna(raw): return None
    n = unicodedata.normalize("NFKD", str(raw)).encode("ascii","ignore").decode("ascii").strip()
    if n in CANONICAL: return CANONICAL[n]
    return n

def dkey(name) -> str:
    if not name: return ""
    return name.lower().replace("-","").replace("'","").replace(" ","").replace(".","")


# -- LOAD DATA --------------------------------------------------

def load_matches() -> pd.DataFrame:
    rows = []

    # Raw 2021-2024 (has round/level info)
    cands = sorted(glob.glob(f"{DATA_DIR}/match_dataset*.csv"))
    if not cands: cands = sorted(glob.glob(f"{REPORTS_DIR}/match_dataset*.csv"))
    if cands:
        df = pd.read_csv(cands[-1])
        dc = next((c for c in ["date","tourney_date"] if c in df.columns), None)
        wc = next((c for c in ["winner_name","p1_name"] if c in df.columns), None)
        lc = next((c for c in ["loser_name","p2_name"] if c in df.columns), None)
        sc = "surface" if "surface" in df.columns else None
        rc = "round" if "round" in df.columns else None
        lvc = next((c for c in ["tourney_level","level"] if c in df.columns), None)
        if all([dc,wc,lc]):
            out = pd.DataFrame({
                "date":    pd.to_datetime(df[dc], errors="coerce"),
                "winner":  df[wc].apply(lambda x: to_canon(str(x))),
                "loser":   df[lc].apply(lambda x: to_canon(str(x))),
                "surface": df[sc].apply(norm_surf) if sc else "Hard",
                "round":   df[rc] if rc else "R32",
                "level":   df[lvc] if lvc else "B",
                "source":  "raw",
            })
            if wc == "p1_name" and "y" in df.columns:
                y = pd.to_numeric(df["y"], errors="coerce")
                out["winner"] = np.where(y==1,
                    df["p1_name"].apply(lambda x: to_canon(str(x))),
                    df["p2_name"].apply(lambda x: to_canon(str(x))))
                out["loser"] = np.where(y==1,
                    df["p2_name"].apply(lambda x: to_canon(str(x))),
                    df["p1_name"].apply(lambda x: to_canon(str(x))))
            rows.append(out.dropna(subset=["date","winner","loser"]))
            print(f"  Raw dataset: {len(out):,} matches")

    # Prediction files 2025-2026
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]
    for fpath in files:
        df = pd.read_csv(fpath)
        slug = os.path.basename(fpath).split("_predictions")[0]
        parts = slug.rsplit("_",1)
        rnd = parts[1] if len(parts)==2 else "R32"
        # Infer level from slug
        lv = "B"
        if any(x in slug for x in ["ao","usopen","rg","wimbledon"]): lv = "G"
        elif any(x in slug for x in ["madrid","miami","indian","montecarlo","rome",
                                       "canada","cincinnati","shanghai","paris","atpfinals"]): lv = "M"
        elif any(x in slug for x in ["barcelona","halle","queens","dubai","rotterdam"]): lv = "A"
        surface = norm_surf(df["surface"].iloc[0]) if "surface" in df.columns and len(df) else "Hard"
        df = df[pd.to_numeric(df.get("correct_prediction",""), errors="coerce").notna()].copy()
        df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
        for _, r in df.iterrows():
            pa  = to_canon(str(r.get("player_a",""))); pb = to_canon(str(r.get("player_b","")))
            if not pa or not pb: continue
            pred = to_canon(str(r.get("pred_winner","")))
            cp   = int(r["correct_prediction"])
            w    = pred if cp==1 else (pb if pred==pa else pa)
            l    = pb if w==pa else pa
            if not w or not l: continue
            rows.append(pd.DataFrame([{"date":r["date"],"winner":w,"loser":l,
                "surface":surface,"round":rnd,"level":lv,"source":"pred"}]))

    all_m = pd.concat(rows, ignore_index=True)
    all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
    all_m = all_m.dropna(subset=["date","winner","loser"])
    all_m = all_m[all_m["winner"].apply(lambda x: x not in (None,"None","nan",""))]
    all_m = all_m.drop_duplicates(subset=["date","winner","loser"])
    all_m = all_m.sort_values("date", kind="mergesort").reset_index(drop=True)
    print(f"  Total: {len(all_m):,} matches ({all_m['date'].min().date()} -> {all_m['date'].max().date()})")
    return all_m


# -- WALK-FORWARD ELO ------------------------------------------

def run_elo(matches: pd.DataFrame, cfg: WeightedEloConfig) -> Tuple[pd.DataFrame, dict, dict, dict]:
    elo:   Dict[str,float]  = {}
    n_m:   Dict[str,int]    = {}
    canon: Dict[str,str]    = {}
    last_match: Dict[str,pd.Timestamp] = {}
    today = matches["date"].max()
    records = []

    for _, r in matches.iterrows():
        w_raw = str(r.get("winner","")); l_raw = str(r.get("loser",""))
        if w_raw in ("","None","nan") or l_raw in ("","None","nan"): continue
        wk = dkey(w_raw); lk = dkey(l_raw)
        if not wk or not lk: continue
        if wk not in canon: canon[wk] = w_raw
        if lk not in canon: canon[lk] = l_raw
        dt  = pd.Timestamp(r["date"])
        rnd = str(r.get("round","R32"))
        lv  = str(r.get("level","B"))

        for key in [wk,lk]:
            if key not in elo: elo[key]=cfg.base; n_m[key]=0

        ew = elo[wk]; el = elo[lk]
        exp_w = 1.0 / (1.0 + 10**((el-ew)/400))

        # Effective K with all multipliers
        k_eff_w = cfg.effective_k(n_m[wk], rnd, lv, dt, today)
        k_eff_l = cfg.effective_k(n_m[lk], rnd, lv, dt, today)
        k = (k_eff_w + k_eff_l) / 2

        records.append({
            "date":            dt,
            "winner":          canon[wk],
            "loser":           canon[lk],
            "elo_w":           ew,
            "elo_l":           el,
            "elo_delta":       abs(ew-el),
            "exp_w":           exp_w,
            "elo_pred_correct":1 if ew>=el else 0,
            "n_w":             n_m[wk],
            "round":           rnd,
            "level":           lv,
            "k_used":          k,
        })

        elo[wk] = ew + k*(1-exp_w)
        elo[lk] = el - k*(1-exp_w)
        n_m[wk] += 1; n_m[lk] += 1
        last_match[wk] = dt; last_match[lk] = dt

    df = pd.DataFrame(records)
    return df, elo, n_m, canon


# -- EVALUATE --------------------------------------------------

def evaluate(df: pd.DataFrame, elo: dict, n_m: dict, canon: dict,
             cfg: WeightedEloConfig, test_from: pd.Timestamp) -> dict:
    test = df[df["date"] >= test_from].copy()
    if len(test) < 20:
        return {"n":0,"accuracy":None,"brier":None}

    acc   = test["elo_pred_correct"].mean()
    brier = ((test["exp_w"] - 1.0)**2).mean()

    # Calibration: 80-90% bucket
    test["elo_fav"] = test.apply(lambda r: max(r["exp_w"],1-r["exp_w"]), axis=1)
    b = test[(test["elo_fav"]>=0.80) & (test["elo_fav"]<0.90)]
    cal_8090 = b["elo_pred_correct"].mean() if len(b)>=5 else None

    # Rankings
    rows = [(canon[k],v,n_m.get(k,0)) for k,v in elo.items()
            if canon.get(k,"") not in ("","None","nan") and n_m.get(k,0)>=5]
    ranked = sorted(rows, key=lambda x:-x[1])

    # Tsitsipas and Sinner rank
    tsit_rank  = next((i+1 for i,(n,e,m) in enumerate(ranked) if "Tsitsipas" in n), None)
    sinner_rank= next((i+1 for i,(n,e,m) in enumerate(ranked) if "Sinner" in n), None)
    tsit_elo   = next((e for n,e,m in ranked if "Tsitsipas" in n), None)
    sinner_elo = next((e for n,e,m in ranked if "Sinner" in n), None)

    return {
        "n":          len(test),
        "accuracy":   float(acc),
        "brier":      float(brier),
        "cal_8090":   float(cal_8090) if cal_8090 else None,
        "tsit_rank":  tsit_rank,
        "tsit_elo":   tsit_elo,
        "sinner_rank":sinner_rank,
        "sinner_elo": sinner_elo,
        "ranked":     ranked[:25],
    }


# -- CALIBRATION -----------------------------------------------

def print_calibration(df: pd.DataFrame, test_from: pd.Timestamp, name: str) -> None:
    test = df[df["date"]>=test_from].copy()
    test["elo_fav"] = test.apply(lambda r: max(r["exp_w"],1-r["exp_w"]), axis=1)
    bins   = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.90,1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%","90%+"]
    print(f"\n  {name}")
    print(f"  {'Conf':<10}{'n':>5}{'Actual':>8}{'Expected':>9}{'Gap':>7}")
    for i,lbl in enumerate(labels):
        lo,hi=bins[i],bins[i+1]
        b=test[(test["elo_fav"]>=lo)&(test["elo_fav"]<hi)]
        if len(b)<5: continue
        act=b["elo_pred_correct"].mean(); exp=b["elo_fav"].mean()
        flag=" <- ANTI" if act<lo-0.05 else (" OK" if abs(act-exp)<0.05 else "")
        print(f"  {lbl:<10}{len(b):>5}{act:>7.1%}{exp:>9.1%}{(act-exp):>+6.1%}{flag}")


# -- MAIN ------------------------------------------------------

def main():
    print("\n"+"="*70)
    print("  Weighted ELO Validation Test")
    print("="*70)

    print("\nLoading matches...")
    matches = load_matches()

    n_test   = max(300, int(len(matches)*0.20))
    test_from= matches["date"].iloc[-n_test]
    print(f"\n  Test set: last {n_test} matches from {test_from.date()}")

    # Run all configs
    results = {}
    dfs     = {}
    for key, cfg in CONFIGS.items():
        df, elo, n_m, canon = run_elo(matches, cfg)
        r = evaluate(df, elo, n_m, canon, cfg, test_from)
        results[key] = r
        dfs[key]     = df

    # -- ACCURACY TABLE -------------------------------------------
    print(f"\n{'='*70}")
    print(f"  ACCURACY COMPARISON (walk-forward, n={results['current']['n']})")
    print(f"{'='*70}")
    print(f"  {'Config':<45}{'Acc':>7}{'Brier':>8}{'80-90%':>8}{'Tsit':>6}{'Sinner':>7}")
    print(f"  {'-'*70}")

    best_acc = max((r["accuracy"] for r in results.values() if r["n"]>0), default=0)
    for key,r in results.items():
        if r["n"]==0: continue
        cfg    = CONFIGS[key]
        flag   = " <-" if r["accuracy"]==best_acc else ""
        tsit   = f"#{r['tsit_rank']}"  if r["tsit_rank"]  else "?"
        sinner = f"#{r['sinner_rank']}" if r["sinner_rank"] else "?"
        cal    = f"{r['cal_8090']:.0%}" if r["cal_8090"] else "--"
        print(f"  {cfg.name:<45}{r['accuracy']:>6.1%}{r['brier']:>8.4f}{cal:>8}{tsit:>6}{sinner:>7}{flag}")

    # -- TSITSIPAS DEEP DIVE ---------------------------------------
    print(f"\n{'='*70}")
    print(f"  TSITSIPAS vs SINNER RANK COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<45}{'Tsit rank':>10}{'Tsit ELO':>10}{'Sinner rank':>12}")
    print(f"  {'-'*70}")
    for key,r in results.items():
        if r["n"]==0: continue
        tsit   = f"#{r['tsit_rank']}" if r["tsit_rank"] else "?"
        telo   = f"{r['tsit_elo']:.0f}" if r["tsit_elo"] else "?"
        sinner = f"#{r['sinner_rank']}" if r["sinner_rank"] else "?"
        print(f"  {CONFIGS[key].name:<45}{tsit:>10}{telo:>10}{sinner:>12}")

    # -- TOP 20 for best configs ------------------------------------
    best_key = max((k for k,r in results.items() if r["n"]>0),
                   key=lambda k: results[k]["accuracy"])
    print(f"\n{'='*70}")
    print(f"  TOP 20 -- Best config: {CONFIGS[best_key].name}")
    print(f"{'='*70}")
    for i,(name,elo_v,nm) in enumerate(results[best_key]["ranked"][:20]):
        print(f"  {i+1:>3}. {name:<30} {elo_v:>7.0f}  (n={nm})")

    # -- CALIBRATION for top 3 configs -----------------------------
    print(f"\n{'='*70}")
    print(f"  CALIBRATION CURVES (key = 80-90% bucket, was 47% in old system)")
    print(f"{'='*70}")
    for key in ["current","base_deployed","round_level","full_weighted"]:
        if key in dfs and results[key]["n"]>0:
            print_calibration(dfs[key], test_from, CONFIGS[key].name)

    # -- VERDICT ---------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    curr = results["current"]
    for key in ["round_level","full_weighted","round_weighted","level_weighted","recency_24mo"]:
        r = results.get(key,{})
        if not r.get("n"): continue
        acc_d   = (r["accuracy"]-curr["accuracy"])*100
        brier_d = r["brier"]-curr["brier"]
        tsit_d  = (curr["tsit_rank"] or 99) - (r["tsit_rank"] or 99)
        print(f"\n  {CONFIGS[key].name}")
        print(f"    Accuracy:  {acc_d:+.1f}pp  Brier: {brier_d:+.4f}  Tsitsipas: {tsit_d:+.0f} spots lower")
        if acc_d >= -0.5 and tsit_d > 3:
            print(f"    -> PROMISING: Tsitsipas drops without meaningful accuracy loss")
        elif acc_d < -1.5:
            print(f"    -> SKIP: hurts accuracy too much ({acc_d:.1f}pp)")
        else:
            print(f"    -> MARGINAL: small changes in both directions")


if __name__ == "__main__":
    main()
