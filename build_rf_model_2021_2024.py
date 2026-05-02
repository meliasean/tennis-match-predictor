"""
Random Forest match outcome model (ATP 2021–2024), leakage-safe, chronological 80/20 split.
Fixes NaN issue by imputing p1_/p2_ BEFORE computing diffs, plus a final numeric imputer.

Run:
    python build_rf_model_2021_2024.py
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Tuple as TTuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------
# Paths & IO helpers
# ----------------------------
PATH_MATCHES = "atp_matches_2021_2024.csv"
OUT_REPORTS = "./reports"
OUT_MODELS = "./models"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_write_table(df: pd.DataFrame, parquet_path: str, index: bool = False) -> str:
    base, _ = os.path.splitext(parquet_path)
    try:
        df.to_parquet(parquet_path, index=index)
        print(f"Saved: {parquet_path}")
        return parquet_path
    except Exception:
        csv_path = base + ".csv"
        df.to_csv(csv_path, index=index)
        print(f"Parquet engine not available; saved CSV instead: {csv_path}")
        return csv_path

def safe_write_text(text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")
    return path

# ----------------------------
# Utilities
# ----------------------------
def _safe_int_date(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    s = str(int(x))
    try:
        return pd.to_datetime(s, format="%Y%m%d")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def normalize_surface(s: str) -> str:
    if pd.isna(s): return "Other"
    s = str(s).strip().title()
    if "Clay" in s: return "Clay"
    if "Grass" in s: return "Grass"
    if "Hard" in s or "Acrylic" in s or "Carpet" in s: return "Hard"
    return "Other"

# ----------------------------
# Elo + player log (no leakage)
# ----------------------------
@dataclass
class EloParams:
    base: float = 1500.0
    k_overall: float = 24.0
    k_surface: float = 18.0

def build_player_log(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    req = ["tourney_date","tourney_name","surface","tourney_level","match_num","winner_name","loser_name","round","best_of"]
    for c in req:
        if c not in matches.columns:
            raise ValueError(f"Missing required column: {c}")

    m = matches.copy()
    m["tourney_date"] = m["tourney_date"].apply(_safe_int_date)
    m["surface"] = m["surface"].map(normalize_surface)
    m = m.sort_values(["tourney_date","tourney_name","match_num"], kind="mergesort").reset_index(drop=True)

    w_rank = "winner_rank" if "winner_rank" in m.columns else None
    l_rank = "loser_rank" if "loser_rank" in m.columns else None

    elo_overall: Dict[str, float] = {}
    elo_surface: Dict[Tuple[str,str], float] = {}
    rows: List[Dict] = []

    BASE, K, KS = params.base, params.k_overall, params.k_surface

    for _, r in m.iterrows():
        d = r["tourney_date"]; s = r["surface"]
        w = r["winner_name"]; l = r["loser_name"]

        ew_w = elo_overall.get(w, BASE); ew_l = elo_overall.get(l, BASE)
        es_w = elo_surface.get((w, s), BASE); es_l = elo_surface.get((l, s), BASE)

        rows.append({
            "player": w, "opp": l, "date": d, "surface": s,
            "tourney_name": r["tourney_name"], "tourney_level": r["tourney_level"], "round": r["round"], "best_of": r["best_of"],
            "is_win": 1, "pre_elo": ew_w, "pre_selo": es_w, "opp_pre_elo": ew_l, "opp_pre_selo": es_l,
            "rank": r[w_rank] if w_rank else np.nan,
        })
        rows.append({
            "player": l, "opp": w, "date": d, "surface": s,
            "tourney_name": r["tourney_name"], "tourney_level": r["tourney_level"], "round": r["round"], "best_of": r["best_of"],
            "is_win": 0, "pre_elo": ew_l, "pre_selo": es_l, "opp_pre_elo": ew_w, "opp_pre_selo": es_w,
            "rank": r[l_rank] if l_rank else np.nan,
        })

        # Elo update
        exp_w = 1.0 / (1.0 + 10 ** ((ew_l - ew_w) / 400))
        elo_overall[w] = ew_w + K * (1 - exp_w)
        elo_overall[l] = ew_l - K * (1 - exp_w)
        exp_w_s = 1.0 / (1.0 + 10 ** ((es_l - es_w) / 400))
        elo_surface[(w, s)] = es_w + KS * (1 - exp_w_s)
        elo_surface[(l, s)] = es_l - KS * (1 - exp_w_s)

    log = pd.DataFrame(rows).sort_values(["player","date"], kind="mergesort").reset_index(drop=True)

    # Rest/workload
    log["prev_date"] = log.groupby("player")["date"].shift(1)
    log["rest_days"] = (log["date"] - log["prev_date"]).dt.days.clip(lower=0)
    log["rest_days"] = log["rest_days"].fillna(30).clip(0, 60)

    def _rolling_counts(dates: List[pd.Timestamp], window: int) -> List[int]:
        dq = deque(); out = []
        for dt in dates:
            while dq and (dt - dq[0]).days > window: dq.popleft()
            out.append(len(dq)); dq.append(dt)
        return out

    for w in (7, 14, 28):
        log[f"matches_{w}d"] = log.groupby("player")["date"].transform(lambda s: _rolling_counts(list(s), w))

    # Rolling form (prior only)
    log["rolling_10_winrate"] = (
        log.groupby("player")["is_win"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()).fillna(0.5).clip(0,1)
    )
    log["rolling_5_winrate"] = (
        log.groupby("player")["is_win"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean()).fillna(0.5).clip(0,1)
    )

    # Streak (prior)
    def _streak_prior(series: pd.Series) -> pd.Series:
        out = []; cur = 0
        for v in series.shift(1).fillna(-1).astype(int):
            if v == 1: cur = cur + 1 if cur >= 0 else 1
            elif v == 0: cur = cur - 1 if cur <= 0 else -1
            else: cur = 0
            out.append(cur)
        return pd.Series(out, index=series.index)
    log["streak"] = log.groupby("player")["is_win"].transform(_streak_prior)

    # Cumulative priors
    log["avg_rest_days"] = log.groupby("player")["rest_days"].transform(lambda s: s.shift(1).expanding().mean()).fillna(20.0).clip(0,60)
    log["win_rate"] = log.groupby("player")["is_win"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.5).clip(0,1)
    log["peak_elo"] = log.groupby("player")["pre_elo"].transform(lambda s: s.shift(1).cummax()).fillna(1500.0)
    log["current_elo"] = log["pre_elo"]

    # Surface WRs
    log = add_surface_wrs(log)

    # Rank prior
    log["rank_prior"] = log.groupby("player")["rank"].shift(1) if "rank" in log.columns else np.nan

    # H2H prior (Beta(1,1))
    def _h2h_prior(g: pd.DataFrame) -> pd.Series:
        wins = g["is_win"].shift(1).fillna(0)
        csum = wins.cumsum()
        cnt = (~g["is_win"].shift(1).isna()).cumsum()
        return ((1 + csum) / (2 + cnt)).fillna(0.5)
    log["h2h_wr_prior"] = (
        log.sort_values("date").groupby(["player","opp"], sort=False).apply(_h2h_prior).reset_index(level=[0,1], drop=True)
    )

    return log

def add_surface_wrs(log: pd.DataFrame) -> pd.DataFrame:
    log_sorted = log.sort_values(["player","surface","date"])
    wr_surf = (
        log_sorted.groupby(["player","surface"])["is_win"]
                  .transform(lambda s: s.shift(1).expanding().mean())
                  .fillna(0.5).clip(0,1)
    )
    tmp = log_sorted[["player","date","surface"]].copy()
    tmp["wr_surface"] = wr_surf.values

    surf_wide = tmp.pivot_table(index=["player","date"], columns="surface", values="wr_surface", aggfunc="last")
    for s in ["Clay","Grass","Hard"]:
        if s not in surf_wide.columns: surf_wide[s] = np.nan
    surf_wide = surf_wide[["Clay","Grass","Hard"]].sort_values(["player","date"]).reset_index()
    surf_wide[["Clay","Grass","Hard"]] = surf_wide.groupby("player")[["Clay","Grass","Hard"]].ffill()
    surf_wide[["Clay","Grass","Hard"]] = surf_wide[["Clay","Grass","Hard"]].fillna(0.5)

    out = log.merge(surf_wide.rename(columns={"Clay":"wr_Clay","Grass":"wr_Grass","Hard":"wr_Hard"}),
                    on=["player","date"], how="left")
    for c in ["wr_Clay","wr_Grass","wr_Hard"]:
        out[c] = out[c].fillna(0.5)
    return out

# ----------------------------
# Match dataset (impute p1_/p2_ BEFORE diffs)
# ----------------------------
def build_match_dataset(matches: pd.DataFrame, log: pd.DataFrame) -> TTuple[pd.DataFrame, List[str], List[str]]:
    m = matches.copy()
    m["tourney_date"] = m["tourney_date"].apply(_safe_int_date)
    m["surface"] = m["surface"].map(normalize_surface)
    m = m.sort_values(["tourney_date","tourney_name","match_num"], kind="mergesort").reset_index(drop=True)

    p1 = np.minimum(m["winner_name"], m["loser_name"])
    p2 = np.maximum(m["winner_name"], m["loser_name"])
    y = (m["winner_name"] == p1).astype(int)

    base = pd.DataFrame({
        "match_id": np.arange(len(m)),
        "date": m["tourney_date"],
        "surface": m["surface"].fillna("Other"),
        "tourney_level": m["tourney_level"].fillna("UNK"),
        "round": m["round"].fillna("UNK"),
        "best_of": m["best_of"].fillna(3),
        "p1_name": p1, "p2_name": p2,
        "winner_name": m["winner_name"],
        "y": y,
    })

    key_cols = ["player","opp","date"]
    base_feats = [
        "pre_elo","pre_selo","rest_days","matches_28d","rolling_10_winrate","rolling_5_winrate",
        "streak","avg_rest_days","win_rate","peak_elo","current_elo","wr_Clay","wr_Grass","wr_Hard",
        "rank_prior","h2h_wr_prior"
    ]
    log_keyed = log[key_cols + base_feats].copy().sort_values(key_cols).drop_duplicates(key_cols, keep="last")

    # p1 snapshot
    p1_snap = base.rename(columns={"p1_name":"player","p2_name":"opp"}) \
                 .merge(log_keyed, on=["player","opp","date"], how="left").add_prefix("p1_")
    # p2 snapshot
    p2_snap = base.rename(columns={"p2_name":"player","p1_name":"opp"}) \
                 .merge(log_keyed, on=["player","opp","date"], how="left").add_prefix("p2_")

    ds = base.merge(p1_snap, left_on=["p1_name","p2_name","date"], right_on=["p1_player","p1_opp","p1_date"], how="left") \
             .merge(p2_snap, left_on=["p1_name","p2_name","date"], right_on=["p2_opp","p2_player","p2_date"], how="left")

    # --- Impute p1_/p2_ BEFORE diffs ---
    def _fill_cols(df: pd.DataFrame, prefix: str) -> None:
        cols = [f"{prefix}{c}" for c in base_feats]
        for col in cols:
            if col.endswith(("wr_Clay","wr_Grass","wr_Hard")) or "winrate" in col or col.endswith("win_rate") or "h2h" in col:
                df[col] = df[col].fillna(0.5)
            elif "rest" in col or "matches_28d" in col or "streak" in col:
                df[col] = df[col].fillna(0.0)
            elif "elo" in col or "peak_elo" in col or "current_elo" in col or "selo" in col:
                df[col] = df[col].fillna(1500.0)
            elif "rank" in col:
                df[col] = df[col].fillna(2000.0)
            else:
                df[col] = df[col].fillna(0.0)

    _fill_cols(ds, "p1_")
    _fill_cols(ds, "p2_")

    # Diffs (p1 - p2) after imputation
    for f in base_feats:
        ds[f"diff_{f}"] = ds[f"p1_{f}"] - ds[f"p2_{f}"]

    ds["elo_diff"] = ds["diff_pre_elo"]
    ds["selo_diff"] = ds["diff_pre_selo"]
    ds["rank_diff"] = ds["diff_rank_prior"]

    requested = [
        "diff_avg_rest_days","diff_matches_28d","diff_rolling_10_winrate","diff_rolling_5_winrate",
        "diff_streak","diff_win_rate","diff_peak_elo","diff_current_elo","diff_wr_Clay","diff_wr_Grass","diff_wr_Hard",
    ]
    extras = ["elo_diff","selo_diff","rank_diff","diff_h2h_wr_prior"]
    cat_feats = ["surface","tourney_level","best_of","round"]
    feature_cols = requested + extras + cat_feats

    # Final sanity: ensure no NaN in features
    for c in feature_cols:
        if ds[c].isna().any():
            # For categoricals we already filled; for safety:
            if c in cat_feats:
                ds[c] = ds[c].fillna({"surface":"Other","tourney_level":"UNK","round":"UNK","best_of":3}.get(c, "UNK"))
            else:
                ds[c] = ds[c].fillna(0.0)

    keep_cols = ["match_id","date","p1_name","p2_name","y"] + feature_cols
    ds = ds[keep_cols].sort_values("date").reset_index(drop=True)
    return ds, feature_cols, cat_feats

# ----------------------------
# Train / Evaluate
# ----------------------------
def train_eval_rf(ds: pd.DataFrame, feature_cols: List[str], cat_feats: List[str]) -> TTuple[Pipeline, dict]:
    n = len(ds)
    cutoff_idx = int(round(n * 0.8))
    train = ds.iloc[:cutoff_idx].copy()
    test = ds.iloc[cutoff_idx:].copy()

    y_train = train["y"].values
    y_test = test["y"].values

    cat = [c for c in cat_feats if c in feature_cols]
    num = [c for c in feature_cols if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_split=6,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline([("prep", pre), ("rf", rf)])
    pipe.fit(train[feature_cols], y_train)

    p_train = pipe.predict_proba(train[feature_cols])[:,1]
    p_test = pipe.predict_proba(test[feature_cols])[:,1]

    thr = 0.5
    yhat_train = (p_train >= thr).astype(int)
    yhat_test = (p_test >= thr).astype(int)

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cutoff_idx": int(cutoff_idx),
        "cutoff_date": str(ds.iloc[cutoff_idx]["date"].date()) if len(ds) > 0 else "NA",
        "train_acc": float(accuracy_score(y_train, yhat_train)),
        "test_acc": float(accuracy_score(y_test, yhat_test)),
        "train_auc": float(roc_auc_score(y_train, p_train)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "train_brier": float(brier_score_loss(y_train, p_train)),
        "test_brier": float(brier_score_loss(y_test, p_test)),
        "cm_test": confusion_matrix(y_test, yhat_test).tolist(),
    }

    # Importances (map back through transformer)
    rf_model = pipe.named_steps["rf"]
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_out = list(ohe.get_feature_names_out(cat)) if cat else []
    feature_names = num + cat_out
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    return pipe, {"metrics": metrics, "feat_importances": imp_df, "splits": (train, test)}

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    _ensure_dir(OUT_REPORTS); _ensure_dir(OUT_MODELS)

    matches = pd.read_csv(PATH_MATCHES)
    log = build_player_log(matches, EloParams())

    ds, feature_cols, cat_feats = build_match_dataset(matches, log)
    ds_path = safe_write_table(ds, os.path.join(OUT_REPORTS, "match_dataset.parquet"), index=False)

    pipe, out = train_eval_rf(ds, feature_cols, cat_feats)

    # Save model
    model_path = os.path.join(OUT_MODELS, "rf_model.joblib")
    dump({"pipeline": pipe, "feature_cols": feature_cols, "cat_feats": cat_feats}, model_path)
    print(f"Saved model: {model_path}")

    # Save metrics & importances
    mt = out["metrics"]
    metrics_txt = "\n".join([
        "RandomForest (chronological 80/20 split)",
        f"Rows train/test: {mt['n_train']} / {mt['n_test']}",
        f"Cutoff date (first test row): {mt['cutoff_date']}",
        f"Train acc:  {mt['train_acc']:.4f}",
        f"Test acc:   {mt['test_acc']:.4f}",
        f"Train AUC:  {mt['train_auc']:.4f}",
        f"Test AUC:   {mt['test_auc']:.4f}",
        f"Train Brier:{mt['train_brier']:.4f}",
        f"Test Brier: {mt['test_brier']:.4f}",
        f"Confusion matrix (test) [[tn, fp],[fn, tp]]: {mt['cm_test']}",
        f"Features used: {feature_cols}",
    ])
    safe_write_text(metrics_txt, os.path.join(OUT_REPORTS, "rf_metrics.txt"))

    imp_path = safe_write_table(out["feat_importances"], os.path.join(OUT_REPORTS, "rf_feature_importances.parquet"), index=False)

    print("Done.")
    print(f"Dataset: {ds_path}")
    print(f"Metrics: {os.path.join(OUT_REPORTS, 'rf_metrics.txt')}")
    print(f"Importances: {imp_path}")
    print(f"Model: {model_path}")

if __name__ == "__main__":
    main()
