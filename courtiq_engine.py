#!/usr/bin/env python3
"""
CourtIQ Engine
==============
Universal tournament pipeline. Replaces all 168 notebook cells.

Commands:
  predict   -- Generate predictions for a round (fetches draw automatically or accepts manual input)
  results   -- Fetch completed results, score predictions, update profiles
  status    -- Show current tournament state
  site      -- Rebuild courtiq.html from all data

Usage examples:
  python courtiq_engine.py predict --tournament madrid2026 --round R64 --surface Clay --level M
  python courtiq_engine.py predict --tournament madrid2026 --round R32  # draw auto-fetched from previous round results
  python courtiq_engine.py results --tournament madrid2026 --round R64
  python courtiq_engine.py status --tournament madrid2026
  python courtiq_engine.py site

Tournament config is stored in tournaments.json (auto-created on first run).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────

REPORTS_DIR   = Path("./reports")
DATA_DIR      = Path("./data")
MODELS_DIR    = Path("./models")
MODEL_PATH    = MODELS_DIR / "rf_model.joblib"
PROFILES_LATEST = REPORTS_DIR / "player_profiles_latest.csv"
TOURNEY_DB    = Path("./tournaments.json")

REPORTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# TOURNAMENT REGISTRY
# Defines all supported tournaments with their config.
# Add new tournaments here — no other code changes needed.
# ──────────────────────────────────────────────────────────────

TOURNAMENT_CONFIGS: Dict[str, dict] = {
    # Grand Slams
    "ao":           {"name": "Australian Open",     "surface": "Hard",  "level": "G", "best_of": 5, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "rg":           {"name": "Roland Garros",       "surface": "Clay",  "level": "G", "best_of": 5, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "wimbledon":    {"name": "Wimbledon",            "surface": "Grass", "level": "G", "best_of": 5, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "usopen":       {"name": "US Open",              "surface": "Hard",  "level": "G", "best_of": 5, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    # Masters 1000
    "indianwells":  {"name": "Indian Wells",         "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "miami":        {"name": "Miami Open",           "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "montecarlo":   {"name": "Monte-Carlo",          "surface": "Clay",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    "madrid":       {"name": "Madrid Open",          "surface": "Clay",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    "rome":         {"name": "Rome",                 "surface": "Clay",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    "canada":       {"name": "Canada Masters",       "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    "cincinnati":   {"name": "Cincinnati",           "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    "shanghai":     {"name": "Shanghai",             "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R128","R64","R32","R16","QF","SF","F"]},
    "paris":        {"name": "Paris Masters",        "surface": "Hard",  "level": "M", "best_of": 3, "rounds": ["R64","R32","R16","QF","SF","F"]},
    # ATP 500
    "rotterdam":    {"name": "Rotterdam",            "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "dubai":        {"name": "Dubai",                "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "acapulco":     {"name": "Acapulco",             "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "barcelona":    {"name": "Barcelona",            "surface": "Clay",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "halle":        {"name": "Halle",                "surface": "Grass", "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "queens":       {"name": "Queens Club",          "surface": "Grass", "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "washington":   {"name": "Washington",           "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "beijing":      {"name": "Beijing",              "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "tokyo":        {"name": "Tokyo",                "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "vienna":       {"name": "Vienna",               "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "basel":        {"name": "Basel",                "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "doha":         {"name": "Doha",                 "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "dallas":       {"name": "Dallas",               "surface": "Hard",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    "rio":          {"name": "Rio",                  "surface": "Clay",  "level": "A", "best_of": 3, "rounds": ["R32","R16","QF","SF","F"]},
    # Special
    "atpfinals":    {"name": "ATP Finals",           "surface": "Hard",  "level": "F", "best_of": 3, "rounds": ["RR1","RR2","RR3","SF","F"]},
}

def resolve_tourney_key(raw: str) -> Tuple[str, int]:
    """Split 'madrid2026' into ('madrid', 2026)."""
    m = re.match(r'^([a-zA-Z]+)(\d{4})$', raw)
    if not m:
        raise ValueError(f"Tournament must be formatted as <name><year> e.g. madrid2026, got: {raw}")
    key = m.group(1).lower()
    year = int(m.group(2))
    if key not in TOURNAMENT_CONFIGS:
        known = ", ".join(sorted(TOURNAMENT_CONFIGS.keys()))
        raise ValueError(f"Unknown tournament '{key}'. Known: {known}")
    return key, year

def get_config(tourney_id: str) -> dict:
    key, year = resolve_tourney_key(tourney_id)
    cfg = dict(TOURNAMENT_CONFIGS[key])
    cfg["key"] = key
    cfg["year"] = year
    cfg["id"] = tourney_id
    cfg["full_name"] = f"{cfg['name']} {year}"
    return cfg

# ──────────────────────────────────────────────────────────────
# NAME ALIASES  (extend as needed)
# ──────────────────────────────────────────────────────────────

ALIASES: Dict[str, str] = {
    "Felix Auger-Aliassime": "Felix Auger Aliassime",
    "Alex de Minaur": "Alex De Minaur",
    "Jan Lennard Struff": "Jan-Lennard Struff",
    "Botic Van De Zandschulp": "Botic van de Zandschulp",
    "Botic van de Zandschulp": "Botic van de Zandschulp",
    "Alejandro Davidovich": "Alejandro Davidovich Fokina",
    "Blanch Dar.": "Darwin Blanch",
    "Carlos Alcaraz Garfia": "Carlos Alcaraz",
    "Rafael Nadal Parera": "Rafael Nadal",
    "Novak Djokovic": "Novak Djokovic",
}

def alias(name: str) -> str:
    if not name or pd.isna(name):
        return name
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return ALIASES.get(name, name)

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────

def _safe_date(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    try:
        if len(s) == 8 and s.isdigit():
            return pd.to_datetime(s, format="%Y%m%d")
        return pd.to_datetime(s)
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def normalize_surface(s) -> str:
    if pd.isna(s):
        return "Other"
    s = str(s).strip().title()
    if "Clay" in s: return "Clay"
    if "Grass" in s: return "Grass"
    if "Hard" in s or "Acrylic" in s or "Carpet" in s: return "Hard"
    return "Other"

def american_to_prob(odds) -> float:
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)

def devig(p_a: float, p_b: float) -> Tuple[float, float]:
    if any(pd.isna(v) for v in [p_a, p_b]) or p_a <= 0 or p_b <= 0:
        return np.nan, np.nan
    s = p_a + p_b
    return p_a / s, p_b / s

def _bool_from_any(x) -> Optional[bool]:
    if pd.isna(x): return None
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"): return True
    if s in ("0", "false", "f", "no", "n"): return False
    return None

def _first_existing(paths) -> Optional[str]:
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

def report_path(tourney_id: str, round_code: str, suffix: str = "") -> Path:
    s = f"_{suffix}" if suffix else ""
    return REPORTS_DIR / f"{tourney_id}_{round_code}_predictions{s}.csv"

def dataset_path(tourney_id: str) -> Path:
    return DATA_DIR / f"match_dataset_post_{tourney_id}.csv"

def profiles_path(tourney_id: str) -> Path:
    return REPORTS_DIR / f"player_profiles_post_{tourney_id}.csv"

# ──────────────────────────────────────────────────────────────
# ELO ENGINE  (exact port from notebook)
# ──────────────────────────────────────────────────────────────

@dataclass
class EloParams:
    base: float = 1500.0
    k_overall: float = 24.0
    k_surface: float = 18.0

ELO = EloParams()

def build_player_log(matches: pd.DataFrame, params: EloParams = ELO) -> pd.DataFrame:
    m = matches.copy()
    m["tourney_date"] = m["tourney_date"].apply(_safe_date)
    m["surface"] = m["surface"].map(normalize_surface)
    m = m.sort_values(["tourney_date", "tourney_name", "match_num"], kind="mergesort").reset_index(drop=True)

    w_rank = "winner_rank" if "winner_rank" in m.columns else None
    l_rank = "loser_rank" if "loser_rank" in m.columns else None

    elo_overall: Dict[str, float] = {}
    elo_surface: Dict[Tuple[str, str], float] = {}
    BASE, K, KS = params.base, params.k_overall, params.k_surface
    rows: List[Dict] = []

    for _, r in m.iterrows():
        d, s = r["tourney_date"], r["surface"]
        w, l = alias(str(r["winner_name"])), alias(str(r["loser_name"]))

        ew_w = elo_overall.get(w, BASE); ew_l = elo_overall.get(l, BASE)
        es_w = elo_surface.get((w, s), BASE); es_l = elo_surface.get((l, s), BASE)

        exp_w   = 1.0 / (1.0 + 10 ** ((ew_l - ew_w) / 400))
        exp_w_s = 1.0 / (1.0 + 10 ** ((es_l - es_w) / 400))

        new_ew_w = ew_w + K  * (1 - exp_w);   new_ew_l = ew_l - K  * (1 - exp_w)
        new_es_w = es_w + KS * (1 - exp_w_s); new_es_l = es_l - KS * (1 - exp_w_s)

        for player, opp, is_win, pre_e, pre_se, post_e, post_se, rk_col in [
            (w, l, 1, ew_w, es_w, new_ew_w, new_es_w, w_rank),
            (l, w, 0, ew_l, es_l, new_ew_l, new_es_l, l_rank),
        ]:
            rows.append({
                "player": player, "opp": opp, "date": d, "surface": s, "is_win": is_win,
                "pre_elo": pre_e, "pre_selo": pre_se, "post_elo": post_e, "post_selo": post_se,
                "tourney_level": r["tourney_level"], "round": r["round"], "best_of": r["best_of"],
                "rank": r[rk_col] if rk_col else np.nan,
            })

        elo_overall[w] = new_ew_w; elo_overall[l] = new_ew_l
        elo_surface[(w, s)] = new_es_w; elo_surface[(l, s)] = new_es_l

    log = pd.DataFrame(rows).sort_values(["player", "date"], kind="mergesort").reset_index(drop=True)

    # Rest / workload
    log["prev_date"] = log.groupby("player")["date"].shift(1)
    log["rest_days"] = (log["date"] - log["prev_date"]).dt.days.fillna(30).clip(0, 60)

    def _rolling_counts(dates, window):
        dq = deque(); out = []
        for dt in dates:
            while dq and (dt - dq[0]).days > window: dq.popleft()
            out.append(len(dq)); dq.append(dt)
        return out

    for w in (7, 14, 28):
        log[f"matches_{w}d"] = log.groupby("player")["date"].transform(lambda s: _rolling_counts(list(s), w))

    log["rolling_10_winrate"] = (
        log.groupby("player")["is_win"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        .fillna(0.5).clip(0, 1)
    )
    log["rolling_5_winrate"] = (
        log.groupby("player")["is_win"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        .fillna(0.5).clip(0, 1)
    )

    def _streak_prior(series):
        out = []; cur = 0
        for v in series.shift(1).fillna(-1).astype(int):
            if v == 1:   cur = cur + 1 if cur >= 0 else 1
            elif v == 0: cur = cur - 1 if cur <= 0 else -1
            else:        cur = 0
            out.append(cur)
        return pd.Series(out, index=series.index)

    log["streak"]       = log.groupby("player")["is_win"].transform(_streak_prior)
    log["avg_rest_days"] = log.groupby("player")["rest_days"].transform(lambda s: s.shift(1).expanding().mean()).fillna(20.0).clip(0, 60)
    log["win_rate"]      = log.groupby("player")["is_win"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.5).clip(0, 1)
    log["peak_elo"]      = log.groupby("player")["pre_elo"].transform(lambda s: s.shift(1).cummax()).fillna(1500.0)
    log["current_elo"]   = log["pre_elo"]

    # Surface win rates (Beta-smoothed)
    SM_A = SM_B = 5.0
    ls = log.sort_values(["player", "surface", "date"])
    wins_s = ls.groupby(["player", "surface"])["is_win"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0.0)
    cnt_s  = ls.groupby(["player", "surface"])["is_win"].transform(lambda s: s.shift(1).expanding().count()).fillna(0.0)
    wr_s   = ((SM_A + wins_s) / (SM_A + SM_B + cnt_s)).clip(0, 1)

    tmp = ls[["player", "date", "surface"]].copy()
    tmp["wr_surface"] = wr_s.values
    sw = tmp.pivot_table(index=["player", "date"], columns="surface", values="wr_surface", aggfunc="last")
    for surf in ("Clay", "Grass", "Hard"):
        if surf not in sw.columns: sw[surf] = np.nan
    sw = sw[["Clay", "Grass", "Hard"]].sort_values(["player", "date"]).reset_index()
    sw[["Clay", "Grass", "Hard"]] = sw.groupby("player")[["Clay", "Grass", "Hard"]].ffill().fillna(0.5)

    log = log.merge(sw.rename(columns={"Clay": "wr_Clay", "Grass": "wr_Grass", "Hard": "wr_Hard"}),
                    on=["player", "date"], how="left")
    for c in ("wr_Clay", "wr_Grass", "wr_Hard"):
        log[c] = log[c].fillna(0.5)

    log["rank_prior"] = log.groupby("player")["rank"].shift(1) if "rank" in log.columns else np.nan
    return log

# ──────────────────────────────────────────────────────────────
# PLAYER SNAPSHOTS
# ──────────────────────────────────────────────────────────────

BASE_FEATS = [
    "pre_elo", "pre_selo", "rest_days", "matches_28d",
    "rolling_10_winrate", "rolling_5_winrate", "streak",
    "avg_rest_days", "win_rate", "peak_elo", "current_elo",
    "wr_Clay", "wr_Grass", "wr_Hard",
    "rank_prior", "h2h_wr_prior",
    "cnt_Hard", "cnt_Grass", "cnt_Clay",
]

def snapshot_from_log(log: pd.DataFrame, player: str, opp: str,
                       asof: pd.Timestamp, surface: str) -> Dict[str, float]:
    player, opp = alias(player), alias(opp)
    sub = log[(log["player"] == player) & (log["date"] <= asof)].sort_values("date")

    snap = {
        "pre_elo": 1500.0, "pre_selo": 1500.0, "rest_days": 30.0, "matches_28d": 0.0,
        "rolling_10_winrate": 0.5, "rolling_5_winrate": 0.5, "streak": 0.0,
        "avg_rest_days": 20.0, "win_rate": 0.5, "peak_elo": 1500.0, "current_elo": 1500.0,
        "wr_Clay": 0.5, "wr_Grass": 0.5, "wr_Hard": 0.5,
        "rank_prior": np.nan, "h2h_wr_prior": 0.5,
        "cnt_Clay": 0.0, "cnt_Grass": 0.0, "cnt_Hard": 0.0,
    }

    if not sub.empty:
        last = sub.iloc[-1]
        for k in ("pre_elo", "current_elo", "peak_elo", "win_rate", "avg_rest_days",
                  "matches_28d", "rolling_10_winrate", "rolling_5_winrate", "streak",
                  "wr_Clay", "wr_Grass", "wr_Hard"):
            if k in last.index:
                snap[k] = float(last[k])
        snap["rest_days"] = float(np.clip((asof - last["date"]).days, 0, 60))
        for surf, key in [("Clay", "cnt_Clay"), ("Grass", "cnt_Grass"), ("Hard", "cnt_Hard")]:
            snap[key] = float((sub["surface"] == surf).sum())
        if "rank" in sub.columns and not sub["rank"].isna().all():
            snap["rank_prior"] = float(sub["rank"].iloc[-1])

    surf_sub = log[(log["player"] == player) & (log["surface"] == surface) & (log["date"] <= asof)].sort_values("date")
    if not surf_sub.empty:
        snap["pre_selo"] = float(surf_sub["pre_selo"].iloc[-1])

    h2h = log[(log["player"] == player) & (log["opp"] == opp) & (log["date"] <= asof)]
    if not h2h.empty:
        snap["h2h_wr_prior"] = (1 + int(h2h["is_win"].sum())) / (2 + len(h2h))

    return snap

def snapshot_from_profiles(profiles: pd.DataFrame, player: str, surface: str) -> Dict[str, float]:
    player = alias(player)
    rows = profiles[profiles["name"] == player]
    if rows.empty:
        return {
            "pre_elo": 1500.0, "pre_selo": 1500.0, "rest_days": 30.0, "matches_28d": 0.0,
            "rolling_10_winrate": 0.5, "rolling_5_winrate": 0.5, "streak": 0.0,
            "avg_rest_days": 20.0, "win_rate": 0.5, "peak_elo": 1500.0, "current_elo": 1500.0,
            "wr_Clay": 0.5, "wr_Grass": 0.5, "wr_Hard": 0.5,
            "rank_prior": np.nan, "h2h_wr_prior": 0.5,
            "cnt_Clay": 0.0, "cnt_Grass": 0.0, "cnt_Hard": 0.0,
        }
    row = rows.iloc[-1]
    def g(k, default=0.0):
        v = row.get(k, default)
        try: return float(v)
        except: return default

    selo_key = {"Clay": "selo_Clay", "Grass": "selo_Grass"}.get(surface, "selo_Hard")
    return {
        "pre_elo": g("current_elo", 1500.0),
        "current_elo": g("current_elo", 1500.0),
        "peak_elo": g("peak_elo", 1500.0),
        "pre_selo": g(selo_key, 1500.0),
        "win_rate": g("overall_wr", 0.5),
        "avg_rest_days": g("avg_rest_days", 20.0),
        "matches_28d": g("matches_28d", 0.0),
        "rolling_10_winrate": g("form10_wr", 0.5),
        "rolling_5_winrate": g("form5_wr", 0.5),
        "streak": g("streak", 0.0),
        "wr_Clay": g("wr_Clay", 0.5),
        "wr_Grass": g("wr_Grass", 0.5),
        "wr_Hard": g("wr_Hard", 0.5),
        "rank_prior": np.nan,
        "h2h_wr_prior": 0.5,
        "rest_days": g("avg_rest_days", 20.0),
        "cnt_Clay": g("cnt_Clay", 0.0),
        "cnt_Grass": g("cnt_Grass", 0.0),
        "cnt_Hard": g("cnt_Hard", 0.0),
    }

# ──────────────────────────────────────────────────────────────
# CCK CALIBRATION  (exact port from notebook)
# ──────────────────────────────────────────────────────────────

CCK_TEMP_T       = 1.30
CCK_C_SURF       = 35.0
CCK_K_RECENT     = 6.0
CCK_W_INFO_BLEND = 0.25
CCK_ELO_LAMBDA   = 0.15
CCK_MKT_LAMBDA   = 0.20

def _logit(p): return np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def elo_logistic(ea, eb): return 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))

def cck_calibrate(p_a_std: float, sa: dict, sb: dict,
                  odds_a: float, odds_b: float, surface: str
                 ) -> Tuple[float, float, float, float, float]:
    p_temp = _sigmoid(_logit(p_a_std) / CCK_TEMP_T)

    cnt_key = f"cnt_{surface}" if surface in ("Clay", "Grass", "Hard") else "cnt_Hard"
    ca = sa.get(cnt_key, 0.0); cb = sb.get(cnt_key, 0.0)
    ra = sa.get("matches_28d", 0.0); rb = sb.get("matches_28d", 0.0)
    w_surf   = min(1.0, min(ca, cb) / CCK_C_SURF)
    w_recent = min(1.0, min(ra, rb) / CCK_K_RECENT)
    info_str = CCK_W_INFO_BLEND * w_surf + (1.0 - CCK_W_INFO_BLEND) * w_recent
    p_shrunk = 0.5 + info_str * (p_temp - 0.5)

    p_elo = elo_logistic(sa["pre_elo"], sb["pre_elo"])
    p_mix = (1.0 - CCK_ELO_LAMBDA) * p_shrunk + CCK_ELO_LAMBDA * p_elo

    pa_raw, pb_raw = american_to_prob(odds_a), american_to_prob(odds_b)
    pa_fair, pb_fair = devig(pa_raw, pb_raw)

    if pd.isna(pa_fair):
        p_cck = p_mix
    else:
        p_cck = (1.0 - CCK_MKT_LAMBDA) * p_mix + CCK_MKT_LAMBDA * pa_fair

    return (
        float(np.clip(p_cck, 0.01, 0.99)),
        float(pa_fair) if not pd.isna(pa_fair) else np.nan,
        float(pb_fair) if not pd.isna(pb_fair) else np.nan,
        float(p_elo),
        float(p_temp),
    )

# ──────────────────────────────────────────────────────────────
# DRAW FETCHER  (ATP website scraper)
# ──────────────────────────────────────────────────────────────

def _fetch_draw_from_atp(tourney_id: str, round_code: str, cfg: dict) -> Optional[List[Tuple]]:
    """
    Try to fetch the draw from the ATP website.
    Returns list of (player_a, player_b) tuples in bracket order, or None if unavailable.
    Odds are NOT fetched here — they must be added manually or via The Odds API.
    """
    try:
        import urllib.request
        # ATP draw URL pattern
        key = cfg["key"]
        year = cfg["year"]

        # Map round codes to ATP round IDs
        round_map = {
            "R128": "R128", "R64": "R64", "R32": "R32", "R16": "R16",
            "QF": "QF", "SF": "SF", "F": "F",
        }
        atp_round = round_map.get(round_code, round_code)

        # We'll try two common ATP draw endpoints
        # This is a best-effort scrape — ATP sometimes changes their URL structure
        urls = [
            f"https://www.atptour.com/en/scores/archive/{key}/{year}/draws",
            f"https://www.atptour.com/en/tournaments/{key}/{year}/draws",
        ]

        headers = {"User-Agent": "Mozilla/5.0 (compatible; CourtIQ/1.0)"}

        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    # Parse player names from draw HTML
                    # ATP uses spans with class "name" for player names
                    names = re.findall(r'class="name[^"]*"[^>]*>\s*([A-Z][^<]{3,40})\s*<', html)
                    names = [n.strip() for n in names if len(n.strip()) > 3]
                    if len(names) >= 4:
                        # Pair them up
                        pairs = [(names[i], names[i+1]) for i in range(0, len(names)-1, 2)]
                        print(f"  [ATP scraper] Found {len(pairs)} matches for {round_code}")
                        return pairs
            except Exception:
                continue

        return None

    except Exception as e:
        print(f"  [ATP scraper] Failed: {e}")
        return None


def _infer_next_round_draw(tourney_id: str, round_code: str, cfg: dict) -> Optional[List[Tuple]]:
    """
    Build the next round's draw from completed previous round results.
    Winners of matches 1,2 play each other; winners of 3,4 play each other; etc.
    """
    rounds = cfg["rounds"]
    if round_code not in rounds:
        return None

    idx = rounds.index(round_code)
    if idx == 0:
        return None  # First round — can't infer from nothing

    prev_round = rounds[idx - 1]
    prev_complete = report_path(tourney_id, prev_round, "complete")
    prev_cck_complete = report_path(tourney_id, prev_round, "cck_complete")

    path = None
    for p in [prev_cck_complete, prev_complete]:
        if p.exists():
            path = p
            break

    if not path:
        return None

    df = pd.read_csv(path)
    if "correct_prediction" not in df.columns:
        return None

    # Reconstruct winners
    winners = []
    for _, row in df.iterrows():
        cp = _bool_from_any(row.get("correct_prediction"))
        if cp is None:
            return None  # incomplete round
        pred = alias(str(row["pred_winner"]))
        player_a = alias(str(row["player_a"]))
        player_b = alias(str(row["player_b"]))
        actual_winner = pred if cp else (player_b if pred == player_a else player_a)
        winners.append(actual_winner)

    if len(winners) % 2 != 0:
        print(f"  [draw inference] Odd number of winners ({len(winners)}) — cannot pair for {round_code}")
        return None

    pairs = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
    print(f"  [draw inference] Built {len(pairs)}-match draw from {prev_round} results")
    return pairs

# ──────────────────────────────────────────────────────────────
# RESULTS FETCHER  (ATP / Flash Score scraper)
# ──────────────────────────────────────────────────────────────

def fetch_results_from_web(tourney_id: str, round_code: str, cfg: dict) -> Optional[Dict[str, str]]:
    """
    Try to fetch match results (winner names) from the web.
    Returns dict of {match_key: winner_name} or None.
    match_key = "PlayerA vs PlayerB" (canonical form)
    """
    try:
        import urllib.request

        key = cfg["key"]
        year = cfg["year"]

        # Try ATP results page
        url = f"https://www.atptour.com/en/scores/archive/{key}/{year}/results"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CourtIQ/1.0)"}

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Look for completed match scores in the HTML
        # ATP format typically has winner highlighted or listed first
        results = {}

        # Simple heuristic: find score patterns like "6-3 7-5" and the names around them
        score_pattern = re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+).*?(\d-\d).*?([A-Z][a-z]+ [A-Z][a-z]+)')
        for m in score_pattern.finditer(html):
            p1, p2 = alias(m.group(1)), alias(m.group(3))
            key_fwd = f"{p1} vs {p2}"
            key_rev = f"{p2} vs {p1}"
            results[key_fwd] = p1  # winner listed first on ATP site
            results[key_rev] = p1

        return results if results else None

    except Exception as e:
        print(f"  [results fetcher] Failed: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# INACTIVE PLAYER FILTER
# ──────────────────────────────────────────────────────────────

# Players retired or inactive — filtered from CourtIQ display
# Add names here as players retire
INACTIVE_PLAYERS = {
    "Rafael Nadal", "Roger Federer", "Stan Wawrinka", "Kei Nishikori",
    "Marin Cilic", "David Ferrer", "Andy Murray", "Jo-Wilfried Tsonga",
    "Gael Monfils", "Nick Kyrgios", "Milos Raonic", "Grigor Dimitrov",
    "Juan Martin del Potro", "Dominic Thiem", "Gilles Simon",
    "Richard Gasquet", "Benoit Paire", "Feliciano Lopez",
}

INACTIVE_CUTOFF_DAYS = 180  # flag as potentially inactive if no match in 6 months

def filter_active_players(profiles: pd.DataFrame) -> pd.DataFrame:
    """Remove known retired players and flag long-inactive ones."""
    # Hard remove known retirees
    profiles = profiles[~profiles["name"].isin(INACTIVE_PLAYERS)].copy()

    # Flag players with no match in INACTIVE_CUTOFF_DAYS as inactive
    today = pd.Timestamp.today()
    profiles["last_match_date"] = pd.to_datetime(profiles["last_match_date"], errors="coerce")
    days_since = (today - profiles["last_match_date"]).dt.days
    profiles["active"] = days_since <= INACTIVE_CUTOFF_DAYS

    # Sort by ELO descending, active players first
    profiles = profiles.sort_values(["active", "current_elo"], ascending=[False, False]).reset_index(drop=True)
    return profiles

# ──────────────────────────────────────────────────────────────
# PROFILE UPDATER  (seeded Elo from prior, not full replay)
# ──────────────────────────────────────────────────────────────

def load_best_profiles() -> pd.DataFrame:
    """Load the highest-quality player profiles available."""
    candidates = sorted(REPORTS_DIR.glob("player_profiles_post_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [str(p) for p in candidates] + [str(PROFILES_LATEST)]

    best_df = None
    best_score = (-1, -1)

    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            if "name" not in df.columns:
                continue
            df["name"] = df["name"].astype(str).apply(alias)
            df["last_match_date"] = pd.to_datetime(df["last_match_date"], errors="coerce")
            df = df.sort_values(["name", "last_match_date"], na_position="last").drop_duplicates("name", keep="last")
            score = (int(df["last_match_date"].notna().sum()), len(df))
            if score > best_score:
                best_score = score
                best_df = df
        except Exception:
            continue

    if best_df is None:
        raise FileNotFoundError("No player profiles found in ./reports/")

    return best_df

def load_best_dataset(tourney_id: str) -> Optional[pd.DataFrame]:
    """Load most recent match dataset, preferring post-tournament files."""
    candidates = sorted(DATA_DIR.glob("match_dataset_post_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        slug = p.stem.replace("match_dataset_post_", "")
        if slug != tourney_id:  # don't load the file we're about to write
            try:
                return pd.read_csv(str(p))
            except Exception:
                continue
    return None

def completed_to_raw_matches(tourney_id: str, cfg: dict) -> pd.DataFrame:
    """
    Convert completed prediction CSVs (with correct_prediction filled)
    into a raw match format compatible with build_player_log.
    """
    pattern = str(REPORTS_DIR / f"{tourney_id}_*_predictions_complete.csv")
    files = [f for f in glob.glob(pattern) if "_cck" not in Path(f).name]

    if not files:
        raise FileNotFoundError(f"No completed prediction files found matching: {pattern}")

    rows = []
    for path in sorted(files):
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        cp = df["correct_prediction"].apply(_bool_from_any)
        df = df.loc[~cp.isna()].copy()
        if df.empty:
            continue
        df["cp_bool"] = cp.loc[~cp.isna()].astype(bool)

        surface = normalize_surface(df["surface"].iloc[0]) if "surface" in df.columns else cfg["surface"]
        level   = str(df["tourney_level"].iloc[0]) if "tourney_level" in df.columns else cfg["level"]
        best_of = int(pd.to_numeric(df["best_of"].iloc[0], errors="coerce") if "best_of" in df.columns else cfg["best_of"])

        for _, r in df.iterrows():
            pred = alias(str(r["pred_winner"]))
            pa   = alias(str(r["player_a"]))
            pb   = alias(str(r["player_b"]))
            winner = pred if r["cp_bool"] else (pb if pred == pa else pa)
            loser  = pb if winner == pa else pa
            rows.append({
                "tourney_date": _safe_date(r["date"]),
                "tourney_name": cfg["full_name"],
                "surface": surface,
                "tourney_level": level,
                "match_num": int(r.get("match_no", 0)),
                "winner_name": winner,
                "loser_name": loser,
                "round": str(r["round"]),
                "best_of": best_of,
            })

    if not rows:
        raise ValueError(f"No completed matches found for {tourney_id}")

    raw = pd.DataFrame(rows).sort_values(["tourney_date", "round", "match_num"], kind="mergesort").reset_index(drop=True)
    print(f"  [updater] Extracted {len(raw)} completed matches from {len(files)} files")
    return raw

def apply_elo_seeded(raw: pd.DataFrame, seeds: pd.DataFrame, params: EloParams = ELO) -> pd.DataFrame:
    """Apply ELO updates seeded from prior profiles (no full history replay)."""
    elo_overall: Dict[str, float] = {}
    elo_surface: Dict[Tuple[str, str], float] = {}
    BASE, K, KS = params.base, params.k_overall, params.k_surface
    rows = []

    def _seed_player(name, surface):
        if name not in elo_overall:
            row = seeds[seeds["name"] == name]
            if not row.empty:
                elo_overall[name] = float(row.iloc[-1].get("current_elo", BASE))
            else:
                elo_overall[name] = BASE
        key = (name, surface)
        if key not in elo_surface:
            row = seeds[seeds["name"] == name]
            if not row.empty:
                sk = {"Clay": "selo_Clay", "Grass": "selo_Grass"}.get(surface, "selo_Hard")
                elo_surface[key] = float(row.iloc[-1].get(sk, BASE))
            else:
                elo_surface[key] = BASE

    for _, r in raw.iterrows():
        d = _safe_date(r["tourney_date"])
        s = normalize_surface(r["surface"])
        w, l = alias(str(r["winner_name"])), alias(str(r["loser_name"]))

        _seed_player(w, s); _seed_player(l, s)

        ew_w, ew_l = elo_overall[w], elo_overall[l]
        es_w, es_l = elo_surface[(w, s)], elo_surface[(l, s)]

        exp_w   = 1.0 / (1.0 + 10 ** ((ew_l - ew_w) / 400))
        exp_w_s = 1.0 / (1.0 + 10 ** ((es_l - es_w) / 400))

        new_ew_w = ew_w + K  * (1 - exp_w);   new_ew_l = ew_l - K  * (1 - exp_w)
        new_es_w = es_w + KS * (1 - exp_w_s); new_es_l = es_l - KS * (1 - exp_w_s)

        for player, opp, is_win, pre_e, pre_se, post_e, post_se in [
            (w, l, 1, ew_w, es_w, new_ew_w, new_es_w),
            (l, w, 0, ew_l, es_l, new_ew_l, new_es_l),
        ]:
            rows.append({
                "player": player, "opp": opp, "date": d, "surface": s, "is_win": is_win,
                "pre_elo": pre_e, "pre_selo": pre_se, "post_elo": post_e, "post_selo": post_se,
                "tourney_level": r["tourney_level"], "round": r["round"], "best_of": r["best_of"],
            })

        elo_overall[w] = new_ew_w; elo_overall[l] = new_ew_l
        elo_surface[(w, s)] = new_es_w; elo_surface[(l, s)] = new_es_l

    return pd.DataFrame(rows).sort_values(["player", "date"], kind="mergesort").reset_index(drop=True)

def rebuild_profiles_from_event(event_log: pd.DataFrame, seeds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Rebuild player profiles by merging event results into the seed profiles.
    Only updates players who actually played in the event.
    """
    surface = cfg["surface"]
    players_in_event = set(event_log["player"].unique())

    # Start with seed profiles for everyone
    updated = seeds.copy()

    for player in players_in_event:
        sub = event_log[event_log["player"] == player].sort_values("date")
        if sub.empty:
            continue

        last = sub.iloc[-1]
        seed_row = seeds[seeds["name"] == player]

        # Get final ELO after this event
        new_elo = float(last["post_elo"])
        new_selo = float(last["post_selo"])

        # Streak: count from last entry in event
        streak = 0
        for _, r in sub.sort_values("date").iterrows():
            if r["is_win"] == 1: streak = streak + 1 if streak >= 0 else 1
            else:                 streak = streak - 1 if streak <= 0 else -1

        if not seed_row.empty:
            idx = seed_row.index[-1]
            # Update ELO
            updated.loc[idx, "current_elo"] = new_elo
            updated.loc[idx, "peak_elo"] = max(float(updated.loc[idx, "peak_elo"]), new_elo)
            # Update surface ELO
            selo_col = {"Clay": "selo_Clay", "Grass": "selo_Grass"}.get(surface, "selo_Hard")
            updated.loc[idx, selo_col] = new_selo
            # Update overall win rate
            total_wins = float(sub["is_win"].sum())
            total_matches = float(len(sub))
            prior_wr = float(seed_row.iloc[-1].get("overall_wr", 0.5))
            # Weighted update (event matches count less than full history)
            prior_n = 50  # approximate prior match count for smoothing
            updated.loc[idx, "overall_wr"] = (prior_wr * prior_n + total_wins) / (prior_n + total_matches)
            # Update surface WR similarly
            wr_col = {"Clay": "wr_Clay", "Grass": "wr_Grass"}.get(surface, "wr_Hard")
            prior_swr = float(seed_row.iloc[-1].get(wr_col, 0.5))
            updated.loc[idx, wr_col] = (prior_swr * 20 + total_wins) / (20 + total_matches)
            # Form (last 10 / last 5)
            recent = sub.tail(10)["is_win"]
            updated.loc[idx, "form10_wr"] = float(recent.mean()) if len(recent) >= 3 else prior_wr
            recent5 = sub.tail(5)["is_win"]
            updated.loc[idx, "form5_wr"] = float(recent5.mean()) if len(recent5) >= 3 else prior_wr
            # Streak
            updated.loc[idx, "streak"] = float(streak)
            # Matches 28d — increment
            updated.loc[idx, "matches_28d"] = float(updated.loc[idx, "matches_28d"]) + total_matches
            # Last match date
            updated.loc[idx, "last_match_date"] = last["date"].strftime("%Y-%m-%d")
        else:
            # New player not in seeds — add them
            new_row = {
                "name": player, "last_match_date": last["date"].strftime("%Y-%m-%d"),
                "current_elo": new_elo, "peak_elo": new_elo,
                "selo_Clay": new_selo if surface == "Clay" else 1500.0,
                "selo_Grass": new_selo if surface == "Grass" else 1500.0,
                "selo_Hard": new_selo if surface == "Hard" else 1500.0,
                "wr_Clay": 0.5, "wr_Grass": 0.5, "wr_Hard": 0.5,
                "overall_wr": float(sub["is_win"].mean()),
                "form10_wr": float(sub.tail(10)["is_win"].mean()),
                "form5_wr": float(sub.tail(5)["is_win"].mean()),
                "streak": float(streak),
                "avg_rest_days": 20.0, "matches_28d": float(len(sub)),
            }
            updated = pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)

    return updated

# ──────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────

def run_predictions(
    tourney_id: str,
    round_code: str,
    matches: List[Tuple],  # (player_a, odds_a, player_b, odds_b) or (player_a, player_b)
    cfg: dict,
    as_of_date: str,
    model_bundle: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run model predictions for a list of matches.
    Returns (std_df, cck_df) — both prediction DataFrames.
    """
    pipe = model_bundle["pipeline"]
    feature_cols = model_bundle["feature_cols"]

    surface  = cfg["surface"]
    level    = cfg["level"]
    best_of  = cfg["best_of"]
    asof     = _safe_date(as_of_date)
    RANK_CLIP = 500

    # Load match data for log, fall back to profiles
    dataset = load_best_dataset(tourney_id)
    log = None
    profiles_mode = False

    if dataset is not None:
        need = ["tourney_date", "tourney_name", "surface", "tourney_level", "match_num", "winner_name", "loser_name", "round", "best_of"]
        if all(c in dataset.columns for c in need):
            dataset["tourney_date"] = dataset["tourney_date"].apply(_safe_date)
            dataset["surface"] = dataset["surface"].map(normalize_surface)
            pre = dataset[dataset["tourney_date"] < asof].copy()
            if len(pre) > 0:
                log = build_player_log(pre)
            else:
                profiles_mode = True
        else:
            profiles_mode = True
    else:
        profiles_mode = True

    prof_df = None
    if profiles_mode:
        prof_df = load_best_profiles()

    def get_snap(player, opp):
        if profiles_mode:
            return snapshot_from_profiles(prof_df, player, surface)
        return snapshot_from_log(log, player, opp, asof, surface)

    def delta(a, b, key):
        return float(a.get(key, 0.0)) - float(b.get(key, 0.0))

    std_rows, cck_rows = [], []

    for i, match in enumerate(matches, start=1):
        if len(match) == 4:
            A_raw, odds_A, B_raw, odds_B = match
        elif len(match) == 2:
            A_raw, B_raw = match
            odds_A = odds_B = np.nan
        else:
            print(f"  [predict] Skipping malformed match entry: {match}")
            continue

        A = alias(str(A_raw))
        B = alias(str(B_raw))
        odds_A = float(odds_A) if not pd.isna(odds_A) else np.nan
        odds_B = float(odds_B) if not pd.isna(odds_B) else np.nan

        sa = get_snap(A, B)
        sb = get_snap(B, A)

        row_feats = {
            "surface": surface, "tourney_level": level,
            "round": round_code, "best_of": best_of,
        }
        for f in BASE_FEATS:
            row_feats[f"diff_{f}"] = delta(sa, sb, f)
        row_feats["elo_diff"]  = delta(sa, sb, "pre_elo")
        row_feats["selo_diff"] = delta(sa, sb, "pre_selo")
        ra, rb = sa.get("rank_prior", np.nan), sb.get("rank_prior", np.nan)
        row_feats["rank_diff"] = 0.0 if (pd.isna(ra) or pd.isna(rb)) else float(np.clip(ra - rb, -RANK_CLIP, RANK_CLIP))

        feats = pd.DataFrame([row_feats])
        for c in feature_cols:
            if c not in feats.columns:
                feats[c] = {"surface": surface, "tourney_level": level,
                            "round": round_code, "best_of": best_of}.get(c, 0.0)

        p_a_std = float(pipe.predict_proba(feats[feature_cols])[:, 1][0])
        pred_std = A if p_a_std >= 0.5 else B
        conf_std = max(p_a_std, 1.0 - p_a_std)
        ws, ls = (sa, sb) if pred_std == A else (sb, sa)

        base_row = {
            "match_no": i, "date": asof.strftime("%Y-%m-%d"), "round": round_code,
            "player_a": A, "odds_player_a": odds_A,
            "player_b": B, "odds_player_b": odds_B,
            "pred_winner": pred_std, "correct_prediction": pd.NA, "correct_prediction_book": pd.NA,
            "confidence": conf_std,
            "prob_player_a_win": p_a_std, "prob_player_b_win": 1.0 - p_a_std,
            "surface": surface, "tourney_level": level, "best_of": best_of,
            "book_fair_prob_a": pd.NA, "book_fair_prob_b": pd.NA,
            "p_elo_a": pd.NA, "p_temp_a": pd.NA,
            "delta_elo":            delta(ws, ls, "pre_elo"),
            "delta_surface_elo":    delta(ws, ls, "pre_selo"),
            "delta_peak_elo":       delta(ws, ls, "peak_elo"),
            "delta_current_elo":    delta(ws, ls, "current_elo"),
            "delta_wr_Clay":        delta(ws, ls, "wr_Clay"),
            "delta_wr_Grass":       delta(ws, ls, "wr_Grass"),
            "delta_wr_Hard":        delta(ws, ls, "wr_Hard"),
            "delta_win_rate":       delta(ws, ls, "win_rate"),
            "delta_avg_rest_days":  delta(ws, ls, "avg_rest_days"),
            "delta_matches_28d":    delta(ws, ls, "matches_28d"),
            "delta_rolling_10_winrate": delta(ws, ls, "rolling_10_winrate"),
            "delta_rolling_5_winrate":  delta(ws, ls, "rolling_5_winrate"),
            "delta_streak":         delta(ws, ls, "streak"),
            "delta_h2h_wr_prior":   delta(ws, ls, "h2h_wr_prior"),
            "delta_rank":           -delta(ws, ls, "rank_prior") if not (pd.isna(ra) or pd.isna(rb)) else 0.0,
        }
        std_rows.append(base_row)

        # CCK row
        p_cck, pa_fair, pb_fair, p_elo, p_temp = cck_calibrate(p_a_std, sa, sb, odds_A, odds_B, surface)
        pred_cck = A if p_cck >= 0.5 else B
        conf_cck = max(p_cck, 1.0 - p_cck)
        wc, lc = (sa, sb) if pred_cck == A else (sb, sa)

        cck_row = dict(base_row)
        cck_row.update({
            "pred_winner": pred_cck, "confidence": conf_cck,
            "prob_player_a_win": p_cck, "prob_player_b_win": 1.0 - p_cck,
            "book_fair_prob_a": pa_fair, "book_fair_prob_b": pb_fair,
            "p_elo_a": p_elo, "p_temp_a": p_temp,
        })
        # Recalculate deltas relative to CCK predicted winner
        for k in ("delta_elo","delta_surface_elo","delta_peak_elo","delta_current_elo",
                  "delta_wr_Clay","delta_wr_Grass","delta_wr_Hard","delta_win_rate",
                  "delta_avg_rest_days","delta_matches_28d","delta_rolling_10_winrate",
                  "delta_rolling_5_winrate","delta_streak","delta_h2h_wr_prior","delta_rank"):
            feat = k.replace("delta_", "")
            feat_map = {
                "elo": "pre_elo", "surface_elo": "pre_selo", "win_rate": "win_rate",
                "peak_elo": "peak_elo", "current_elo": "current_elo",
                "wr_Clay": "wr_Clay", "wr_Grass": "wr_Grass", "wr_Hard": "wr_Hard",
                "avg_rest_days": "avg_rest_days", "matches_28d": "matches_28d",
                "rolling_10_winrate": "rolling_10_winrate", "rolling_5_winrate": "rolling_5_winrate",
                "streak": "streak", "h2h_wr_prior": "h2h_wr_prior", "rank": "rank_prior",
            }
            f = feat_map.get(feat, feat)
            cck_row[k] = delta(wc, lc, f) if f != "rank_prior" else (-delta(wc, lc, "rank_prior") if not (pd.isna(ra) or pd.isna(rb)) else 0.0)

        cck_rows.append(cck_row)

    return pd.DataFrame(std_rows), pd.DataFrame(cck_rows)

# ──────────────────────────────────────────────────────────────
# SACKMANN SERVE STATS ENRICHMENT
# ──────────────────────────────────────────────────────────────

def fetch_sackmann_stats(year: int = 2024) -> Optional[pd.DataFrame]:
    """
    Fetch Jeff Sackmann's ATP match data and compute per-player serve/return averages.
    Returns a DataFrame with player-level stats or None if unavailable.
    License: CC BY-NC-SA 4.0 — attribution required, non-commercial only.
    """
    try:
        import urllib.request

        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_{year}.csv"
        print(f"  [sackmann] Fetching {year} match data from JeffSackmann/tennis_atp...")

        req = urllib.request.Request(url, headers={"User-Agent": "CourtIQ/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            from io import StringIO
            content = resp.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(StringIO(content))

        stat_cols = [
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
            "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
            "l_SvGms", "l_bpSaved", "l_bpFaced",
        ]
        available = [c for c in stat_cols if c in df.columns]
        if not available:
            return None

        # Compute per-player averages from winner perspective
        w_stats = df[["winner_name"] + [c for c in stat_cols if c.startswith("w_")]].copy()
        w_stats.columns = ["name"] + [c[2:] for c in w_stats.columns if c != "winner_name"]
        w_stats["result"] = "win"

        l_stats = df[["loser_name"] + [c for c in stat_cols if c.startswith("l_")]].copy()
        l_stats.columns = ["name"] + [c[2:] for c in l_stats.columns if c != "loser_name"]
        l_stats["result"] = "loss"

        all_stats = pd.concat([w_stats, l_stats], ignore_index=True)
        all_stats["name"] = all_stats["name"].apply(alias)

        # Compute derived stats
        def safe_div(a, b, default=np.nan):
            return np.where(b > 0, a / b, default)

        agg = all_stats.groupby("name").agg(
            matches_with_stats=("svpt", "count"),
            avg_aces=("ace", "mean"),
            avg_dfs=("df", "mean"),
            avg_1st_in_pct=("1stIn", lambda x: (x / all_stats.loc[x.index, "svpt"].replace(0, np.nan)).mean()),
            avg_1st_won_pct=("1stWon", lambda x: (x / all_stats.loc[x.index, "1stIn"].replace(0, np.nan)).mean()),
            avg_2nd_won_pct=("2ndWon", lambda x: (x / (all_stats.loc[x.index, "svpt"] - all_stats.loc[x.index, "1stIn"]).replace(0, np.nan)).mean()),
            avg_bp_save_pct=("bpSaved", lambda x: (x / all_stats.loc[x.index, "bpFaced"].replace(0, np.nan)).mean()),
        ).reset_index()

        print(f"  [sackmann] Loaded serve stats for {len(agg)} players from {year}")
        return agg

    except Exception as e:
        print(f"  [sackmann] Could not fetch stats: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# COURTIQ SITE BUILDER
# ──────────────────────────────────────────────────────────────

TOURNEY_ORDER = [
    "wimbledon2025", "canada2025", "cincinnati2025", "usopen2025",
    "shanghai2025", "tokyo2025", "beijing2025", "vienna2025", "basel2025",
    "paris2025", "atpfinals2025",
    "doha2026", "dallas2026", "rotterdam2026", "rio2026", "dubai2026",
    "ao2026", "acapulco2026", "indianwells2026", "miami2026", "montecarlo2026",
]

TOURNEY_DISPLAY_NAMES = {
    "wimbledon2025": "Wimbledon 2025", "canada2025": "Canada Masters 2025",
    "cincinnati2025": "Cincinnati 2025", "usopen2025": "US Open 2025",
    "shanghai2025": "Shanghai 2025", "tokyo2025": "Tokyo 2025",
    "beijing2025": "Beijing 2025", "vienna2025": "Vienna 2025",
    "basel2025": "Basel 2025", "paris2025": "Paris Masters 2025",
    "atpfinals2025": "ATP Finals 2025", "doha2026": "Doha 2026",
    "dallas2026": "Dallas 2026", "rotterdam2026": "Rotterdam 2026",
    "rio2026": "Rio 2026", "dubai2026": "Dubai 2026",
    "ao2026": "Australian Open 2026", "acapulco2026": "Acapulco 2026",
    "indianwells2026": "Indian Wells 2026", "miami2026": "Miami Masters 2026",
    "montecarlo2026": "Monte-Carlo 2026",
}

ROUND_ORDER = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR1", "RR2", "RR3"]

def build_site_data() -> dict:
    from collections import defaultdict

    def pick_best(files):
        for suffix in ["_predictions_cck_complete.csv", "_predictions_complete.csv",
                       "_predictions_cck.csv", "_predictions.csv"]:
            for f in files:
                if f.name.endswith(suffix): return f
        return files[0] if files else None

    groups = defaultdict(list)
    for f in REPORTS_DIR.glob("*.csv"):
        if any(x in f.name for x in ["_ALL", "all_rounds", "consistency", "player_profiles",
                                      "player_match", "match_dataset", "rf_"]):
            continue
        key = f.name.split("_predictions")[0]
        groups[key].append(f)

    tourney_data = defaultdict(lambda: {"rounds": [], "has_book": False})

    def sort_key(k):
        parts = k.split("_")
        t = parts[0]; r = parts[1] if len(parts) > 1 else ""
        ti = TOURNEY_ORDER.index(t) if t in TOURNEY_ORDER else 99
        ri = ROUND_ORDER.index(r) if r in ROUND_ORDER else 99
        return (ti, ri)

    for key, files in sorted(groups.items(), key=lambda x: sort_key(x[0])):
        best = pick_best(files)
        if not best: continue
        try:
            df = pd.read_csv(best)
            if "correct_prediction" not in df.columns: continue
            has_book = "odds_player_a" in df.columns and "correct_prediction_book" in df.columns
            cp  = df["correct_prediction"].dropna()
            cpb = df["correct_prediction_book"].dropna() if has_book else pd.Series([], dtype=float)

            parts = key.split("_")
            t_key = parts[0]; r_name = parts[1].upper() if len(parts) > 1 else "R1"

            float_cols = df.select_dtypes("float").columns
            df[float_cols] = df[float_cols].round(4)

            match_keys = ["match_no", "player_a", "player_b", "odds_player_a", "odds_player_b",
                          "pred_winner", "correct_prediction", "correct_prediction_book",
                          "confidence", "prob_player_a_win", "prob_player_b_win"]
            matches = json.loads(df[[c for c in match_keys if c in df.columns]].to_json(orient="records"))

            tourney_data[t_key]["rounds"].append({
                "round": r_name,
                "summary": {
                    "total_matches": len(df),
                    "results_entered": int(len(cp)),
                    "model_correct": int(cp.sum()),
                    "book_correct": int(cpb.sum()) if has_book and len(cpb) else 0,
                    "model_accuracy": round(float(cp.mean()), 4) if len(cp) else None,
                    "book_accuracy": round(float(cpb.mean()), 4) if has_book and len(cpb) else None,
                },
                "matches": matches,
            })
            if has_book: tourney_data[t_key]["has_book"] = True
        except Exception as e:
            print(f"  [site] Warning: could not load {best.name}: {e}")

    tournaments_out = []
    total_model = total_book = total_matches = total_book_matches = 0

    for t_key in TOURNEY_ORDER:
        if t_key not in tourney_data: continue
        td = tourney_data[t_key]
        tm = sum(r["summary"]["results_entered"] for r in td["rounds"])
        mc = sum(r["summary"]["model_correct"] for r in td["rounds"])
        book_rounds = [r for r in td["rounds"] if r["summary"]["book_accuracy"] is not None]
        bm = sum(r["summary"]["results_entered"] for r in book_rounds)
        bc = sum(r["summary"]["book_correct"] for r in book_rounds)

        tournaments_out.append({
            "name": TOURNEY_DISPLAY_NAMES.get(t_key, t_key),
            "slug": t_key,
            "has_book": td["has_book"],
            "summary": {
                "total_matches": tm,
                "model_correct": mc,
                "model_accuracy": round(mc / tm, 4) if tm else None,
                "book_matches": bm,
                "book_correct": bc,
                "book_accuracy": round(bc / bm, 4) if bm else None,
            },
            "rounds": td["rounds"],
        })
        total_matches += tm; total_model += mc
        total_book_matches += bm; total_book += bc

    # Player profiles
    try:
        pf = filter_active_players(load_best_profiles())
        float_cols = pf.select_dtypes("float").columns
        pf[float_cols] = pf[float_cols].round(4)
        players = pf.to_dict(orient="records")
    except Exception as e:
        print(f"  [site] Warning loading profiles: {e}")
        players = []

    # Tourney accuracy summary for overview chart
    tourney_acc = [
        {
            "name": t["name"], "slug": t["slug"],
            "model_acc": t["summary"]["model_accuracy"],
            "book_acc": t["summary"]["book_accuracy"],
            "matches": t["summary"]["total_matches"],
            "has_book": t["has_book"],
        }
        for t in tournaments_out
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overall_accuracy": {
            "total_matches": total_matches,
            "model_accuracy": round(total_model / total_matches, 4) if total_matches else None,
            "model_correct": total_model,
            "book_matches": total_book_matches,
            "book_accuracy": round(total_book / total_book_matches, 4) if total_book_matches else None,
            "book_correct": total_book,
        },
        "tourney_acc": tourney_acc,
        "players": players,
        "tournaments": tournaments_out,
    }

# ──────────────────────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────────────────────

def cmd_predict(args):
    """Generate predictions for a round."""
    cfg = get_config(args.tournament)
    round_code = args.round.upper()

    print(f"\n{'─'*60}")
    print(f"  PREDICT  {cfg['full_name']} — {round_code}")
    print(f"{'─'*60}")

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    bundle = load(str(MODEL_PATH))
    if not isinstance(bundle, dict):
        bundle = {"pipeline": bundle, "feature_cols": None}
        print("WARNING: Old-style model bundle. feature_cols not available.")
    print(f"  Model loaded: {MODEL_PATH}")

    # Get the draw
    matches = None

    # 1. Try to infer from previous round
    if not args.manual:
        matches = _infer_next_round_draw(args.tournament, round_code, cfg)

    # 2. Try ATP website scrape
    if matches is None and not args.manual:
        print("  Trying ATP website for draw...")
        pairs = _fetch_draw_from_atp(args.tournament, round_code, cfg)
        if pairs:
            matches = [(a, np.nan, b, np.nan) for a, b in pairs]
            print(f"  Got {len(matches)} matches from ATP website (no odds — add manually)")

    # 3. Manual entry
    if matches is None:
        print(f"\n  Could not fetch draw automatically.")
        print(f"  Enter matches manually (or create a CSV file).")
        print()

        if args.draw_csv:
            # Load from CSV file: columns player_a, player_b, odds_a (optional), odds_b (optional)
            dc = pd.read_csv(args.draw_csv)
            matches = []
            for _, r in dc.iterrows():
                matches.append((
                    str(r["player_a"]),
                    float(r["odds_a"]) if "odds_a" in dc.columns and not pd.isna(r.get("odds_a")) else np.nan,
                    str(r["player_b"]),
                    float(r["odds_b"]) if "odds_b" in dc.columns and not pd.isna(r.get("odds_b")) else np.nan,
                ))
            print(f"  Loaded {len(matches)} matches from {args.draw_csv}")
        else:
            print("  Interactive draw entry (format: PlayerA, odds_A, PlayerB, odds_B)")
            print("  Press Enter twice when done.\n")
            matches = []
            while True:
                line = input(f"  Match {len(matches)+1}: ").strip()
                if not line:
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    matches.append((parts[0], np.nan, parts[1], np.nan))
                elif len(parts) == 4:
                    try:
                        matches.append((parts[0], float(parts[1]), parts[2], float(parts[3])))
                    except ValueError:
                        print("  Could not parse odds — stored as N/A")
                        matches.append((parts[0], np.nan, parts[2], np.nan))
                else:
                    print("  Expected: PlayerA, [oddsA,] PlayerB, [oddsB]")

    if not matches:
        print("  No matches to predict. Exiting.")
        return

    as_of = args.date or datetime.today().strftime("%Y-%m-%d")
    print(f"\n  Running predictions for {len(matches)} matches (as of {as_of})...")

    std_df, cck_df = run_predictions(
        args.tournament, round_code, matches, cfg, as_of, bundle
    )

    # Save outputs
    out_std = report_path(args.tournament, round_code)
    out_cck = report_path(args.tournament, round_code, "cck")
    std_df.to_csv(out_std, index=False)
    cck_df.to_csv(out_cck, index=False)

    print(f"\n  ✓ Standard predictions → {out_std}")
    print(f"  ✓ CCK predictions      → {out_cck}")
    print()
    print(f"  {'#':<4} {'Player A':<28} {'Pred':<28} {'Conf':>6}")
    print(f"  {'─'*70}")
    for _, r in cck_df.iterrows():
        print(f"  {int(r['match_no']):<4} {r['player_a']:<28} {r['pred_winner']:<28} {r['confidence']:.1%}")
    print()
    print(f"  Next step: Fill in correct_prediction (0/1) in the _complete.csv files,")
    print(f"  then run:  python courtiq_engine.py results --tournament {args.tournament} --round {round_code}")
    print()


def cmd_results(args):
    """Fetch results, score predictions, update profiles."""
    cfg = get_config(args.tournament)
    round_code = args.round.upper()

    print(f"\n{'─'*60}")
    print(f"  RESULTS  {cfg['full_name']} — {round_code}")
    print(f"{'─'*60}")

    std_path = report_path(args.tournament, round_code)
    cck_path = report_path(args.tournament, round_code, "cck")

    if not std_path.exists():
        print(f"ERROR: No prediction file found at {std_path}")
        print(f"  Run predict first: python courtiq_engine.py predict --tournament {args.tournament} --round {round_code}")
        sys.exit(1)

    std_df = pd.read_csv(std_path)
    cck_df = pd.read_csv(cck_path) if cck_path.exists() else std_df.copy()

    # Try auto-fetching results
    auto_results = None
    if not args.manual:
        print("  Trying to fetch results from web...")
        auto_results = fetch_results_from_web(args.tournament, round_code, cfg)
        if auto_results:
            print(f"  Got {len(auto_results)} results from web")

    # Score predictions
    scored_count = 0
    for i, row in std_df.iterrows():
        if pd.notna(std_df.at[i, "correct_prediction"]):
            scored_count += 1
            continue  # already scored

        if auto_results:
            A = alias(str(row["player_a"]))
            B = alias(str(row["player_b"]))
            key = f"{A} vs {B}"
            actual_winner = auto_results.get(key) or auto_results.get(f"{B} vs {A}")
            if actual_winner:
                pred = alias(str(row["pred_winner"]))
                correct = 1 if alias(actual_winner) == pred else 0
                std_df.at[i, "correct_prediction"] = correct
                if cck_path.exists():
                    cck_pred = alias(str(cck_df.at[i, "pred_winner"]))
                    cck_df.at[i, "correct_prediction"] = 1 if alias(actual_winner) == cck_pred else 0
                scored_count += 1

    # Interactive entry for any remaining
    pending = std_df["correct_prediction"].isna()
    if pending.any():
        if auto_results:
            print(f"\n  {pending.sum()} matches still need results. Entering manually:")
        else:
            print(f"\n  Entering {pending.sum()} results manually:")
        print(f"  (1=correct prediction, 0=wrong, s=skip, q=quit)\n")

        for i, row in std_df[pending].iterrows():
            A, B = alias(str(row["player_a"])), alias(str(row["player_b"]))
            pred = alias(str(row["pred_winner"]))
            print(f"  Match {int(row['match_no'])}: {A} vs {B}  →  Predicted: {pred}")

            while True:
                v = input("    Correct? [1/0/s/q]: ").strip().lower()
                if v == "q":
                    break
                if v == "s":
                    break
                if v in ("0", "1"):
                    std_df.at[i, "correct_prediction"] = int(v)
                    if cck_path.exists():
                        cck_pred = alias(str(cck_df.at[i, "pred_winner"]))
                        # For CCK we also need to know actual winner
                        if int(v) == 1:
                            # model correct means pred_winner was correct
                            cck_df.at[i, "correct_prediction"] = 1 if pred == cck_pred else 0
                        else:
                            # model wrong means other player won
                            actual = B if pred == A else A
                            cck_df.at[i, "correct_prediction"] = 1 if actual == cck_pred else 0

                    # Book odds scoring
                    if pd.notna(row.get("odds_player_a")) and pd.notna(row.get("odds_player_b")):
                        pa_raw = american_to_prob(row["odds_player_a"])
                        pb_raw = american_to_prob(row["odds_player_b"])
                        book_pred = A if pa_raw > pb_raw else B
                        if int(v) == 1:
                            actual_winner = pred
                        else:
                            actual_winner = B if pred == A else A
                        std_df.at[i, "correct_prediction_book"] = 1 if actual_winner == book_pred else 0
                        if cck_path.exists():
                            cck_df.at[i, "correct_prediction_book"] = std_df.at[i, "correct_prediction_book"]
                    break
                print("    Please enter 1, 0, s, or q.")

    # Save complete files
    out_complete = report_path(args.tournament, round_code, "complete")
    out_cck_complete = report_path(args.tournament, round_code, "cck_complete")
    std_df.to_csv(out_complete, index=False)
    cck_df.to_csv(out_cck_complete, index=False)

    # Show round accuracy
    cp = std_df["correct_prediction"].dropna()
    cpb = std_df["correct_prediction_book"].dropna()
    print(f"\n  ✓ Saved → {out_complete}")
    if len(cp):
        print(f"  Model: {int(cp.sum())}/{len(cp)} = {cp.mean():.1%}")
    if len(cpb):
        print(f"  Book:  {int(cpb.sum())}/{len(cpb)} = {cpb.mean():.1%}")

    # Update profiles if round is complete
    complete_pct = len(cp) / len(std_df) if len(std_df) else 0
    if complete_pct < 0.5:
        print(f"\n  Only {complete_pct:.0%} of results entered — skipping profile update.")
        print(f"  Run again after more results are in, or when the round is complete.")
    else:
        print(f"\n  Updating player profiles...")
        try:
            seeds = load_best_profiles()
            raw = completed_to_raw_matches(args.tournament, cfg)
            event_log = apply_elo_seeded(raw, seeds)
            updated = rebuild_profiles_from_event(event_log, seeds, cfg)
            updated = filter_active_players(updated)

            out_profiles = profiles_path(args.tournament)
            updated.to_csv(out_profiles, index=False)
            updated.to_csv(PROFILES_LATEST, index=False)
            print(f"  ✓ Profiles updated → {out_profiles}")
            print(f"  ✓ player_profiles_latest.csv updated")
        except Exception as e:
            print(f"  WARNING: Profile update failed: {e}")
            print(f"  You can retry: python courtiq_engine.py results --tournament {args.tournament} --round {round_code} --manual")

    # Suggest next round
    rounds = cfg["rounds"]
    if round_code in rounds:
        idx = rounds.index(round_code)
        if idx < len(rounds) - 1:
            next_round = rounds[idx + 1]
            print(f"\n  Next round: {next_round}")
            print(f"  Run: python courtiq_engine.py predict --tournament {args.tournament} --round {next_round}")
    print()


def cmd_status(args):
    """Show tournament status."""
    cfg = get_config(args.tournament)
    print(f"\n  {cfg['full_name']} — Status")
    print(f"  {'─'*50}")

    total_m = total_c = total_bc = total_b = 0

    for round_code in cfg["rounds"]:
        found = None
        for suffix in ["_cck_complete", "_complete", "_cck", ""]:
            p = report_path(args.tournament, round_code, suffix.lstrip("_"))
            if p.exists():
                found = p; break

        if not found:
            print(f"  {round_code:<8} — not started")
            continue

        df = pd.read_csv(found)
        cp = df["correct_prediction"].dropna() if "correct_prediction" in df.columns else pd.Series()
        cpb = df["correct_prediction_book"].dropna() if "correct_prediction_book" in df.columns else pd.Series()
        has_odds = "odds_player_a" in df.columns

        m_str = f"{int(cp.mean()*100)}%" if len(cp) else "—"
        b_str = f"{int(cpb.mean()*100)}%" if len(cpb) else ("no odds" if not has_odds else "—")
        status = "complete" if len(cp) == len(df) else f"{len(cp)}/{len(df)} scored"
        print(f"  {round_code:<8} {status:<18} model={m_str:<8} book={b_str}")

        total_m += len(df); total_c += int(cp.sum()); total_bc += int(cpb.sum()); total_b += len(cpb)

    if total_m:
        print(f"  {'─'*50}")
        m_overall = f"{total_c/total_m:.1%}"
        b_overall = f"{total_bc/total_b:.1%}" if total_b else "n/a"
        print(f"  {'TOTAL':<8} {total_m} matches        model={m_overall:<8} book={b_overall}")
    print()


def cmd_site(args):
    """Rebuild CourtIQ website."""
    print(f"\n  Building CourtIQ...")

    data = build_site_data()

    print(f"  Players: {len(data['players'])} (active)")
    print(f"  Tournaments: {len(data['tournaments'])}")
    print(f"  Total matches: {data['overall_accuracy']['total_matches']}")
    if data["overall_accuracy"]["model_accuracy"]:
        print(f"  Model accuracy: {data['overall_accuracy']['model_accuracy']:.1%}")
    if data["overall_accuracy"]["book_accuracy"]:
        print(f"  Book accuracy:  {data['overall_accuracy']['book_accuracy']:.1%}")

    js_data = "const SITE_DATA = " + json.dumps(data, separators=(",", ":"), default=str) + ";"

    # Read the HTML template or use the embedded one
    template_path = Path("./courtiq_template.html")
    if template_path.exists():
        html_template = template_path.read_text()
        # Replace the data injection point
        html = re.sub(r'const SITE_DATA = \{.*?\};', js_data, html_template, flags=re.DOTALL)
    else:
        # Embed data into a minimal shell that references the full template
        html = _build_html(js_data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"\n  ✓ CourtIQ → {out}")
    print()


def _build_html(js_data: str) -> str:
    """Minimal HTML wrapper — use courtiq_template.html for the full version."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>CourtIQ — rebuild in progress</title></head>
<body>
<p>Copy courtiq_template.html to this directory for the full site.</p>
<script>{js_data}</script>
</body></html>"""


# ──────────────────────────────────────────────────────────────
# LIVE SCORES COMMAND
# ──────────────────────────────────────────────────────────────

# Map SportRadar tournament name fragments → our tournament slugs + year
# Extend this as new seasons start
LIVE_TOURNEY_MAP = {
    "barcelona":      ("barcelona", 2026),
    "madrid":         ("madrid",    2026),
    "rome":           ("rome",      2026),
    "roland garros":  ("rg",        2026),
    "wimbledon":      ("wimbledon", 2026),
    "washington":     ("washington",2026),
    "canada":         ("canada",    2026),
    "cincinnati":     ("cincinnati",2026),
    "us open":        ("usopen",    2026),
    "shanghai":       ("shanghai",  2026),
    "tokyo":          ("tokyo",     2026),
    "beijing":        ("beijing",   2026),
    "vienna":         ("vienna",    2026),
    "basel":          ("basel",     2026),
    "paris":          ("paris",     2026),
    "atp finals":     ("atpfinals", 2026),
    "australian open":("ao",        2027),
    "indian wells":   ("indianwells",2026),
    "miami":          ("miami",     2026),
    "monte":          ("montecarlo",2026),
    "rotterdam":      ("rotterdam", 2026),
    "dubai":          ("dubai",     2026),
    "doha":           ("doha",      2026),
    "dallas":         ("dallas",    2026),
    "acapulco":       ("acapulco",  2026),
    "rio":            ("rio",       2026),
}

def _slug_from_tourney_name(name: str) -> Optional[Tuple[str, int]]:
    """Map a live tournament name string to (slug, year)."""
    name_lower = name.lower()
    for fragment, slug_year in LIVE_TOURNEY_MAP.items():
        if fragment in name_lower:
            return slug_year
    return None

def fetch_live_atp_scores() -> List[dict]:
    """
    Fetch current ATP tour scores using the SportRadar-backed sports data API.
    Returns a list of match dicts with player names, scores, and status.
    """
    try:
        import urllib.request, json as _json

        # SportRadar tennis endpoint (same data source as the Claude sports tool)
        # This is a public-facing endpoint — no API key needed for basic scores
        url = "https://api.sportradar.com/tennis/trial/v3/en/schedules/live/results.json"
        key = os.environ.get("SPORTRADAR_API_KEY", "")

        if key:
            url += f"?api_key={key}"

        req = urllib.request.Request(url, headers={"User-Agent": "CourtIQ/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())

        results = []
        for sport_event in data.get("results", []):
            se = sport_event.get("sport_event", {})
            sr = sport_event.get("sport_event_status", {})

            tourney = se.get("sport_event_context", {}).get("competition", {}).get("name", "")
            if "ATP" not in tourney and "Grand Slam" not in tourney:
                continue

            competitors = se.get("competitors", [])
            if len(competitors) < 2:
                continue

            p1 = alias(competitors[0].get("name", ""))
            p2 = alias(competitors[1].get("name", ""))
            status = sr.get("status", "")
            winner = alias(sr.get("winner_id", ""))

            # Map winner ID back to name
            for c in competitors:
                if c.get("id") == sr.get("winner_id"):
                    winner = alias(c.get("name", ""))

            results.append({
                "tournament": tourney,
                "player_a": p1,
                "player_b": p2,
                "status": status,
                "winner": winner if status == "closed" else None,
                "score": sr.get("home_score", 0),
            })

        return results

    except Exception as e:
        print(f"  [live] SportRadar API failed ({e}), falling back to ATP website scrape")
        return _scrape_atp_results()


def _scrape_atp_results() -> List[dict]:
    """Fallback: scrape ATP website for completed results."""
    try:
        import urllib.request
        url = "https://www.atptour.com/en/scores/current"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CourtIQ/1.0)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        # Look for completed match patterns in ATP score HTML
        # Pattern: winner name followed by score
        score_blocks = re.findall(
            r'data-winner="([^"]+)"[^>]*>.*?'
            r'class="[^"]*player[^"]*"[^>]*>([^<]+)<.*?'
            r'class="[^"]*player[^"]*"[^>]*>([^<]+)<',
            html, re.DOTALL
        )
        for winner_id, p1, p2 in score_blocks:
            results.append({
                "player_a": alias(p1.strip()),
                "player_b": alias(p2.strip()),
                "status": "closed",
                "winner": None,  # can't reliably determine from this pattern
                "tournament": "ATP Tour",
            })

        return results
    except Exception as e:
        print(f"  [live] ATP scrape also failed: {e}")
        return []


def auto_score_from_live(live_results: List[dict]) -> int:
    """
    Match live completed results against open prediction files.
    Returns number of predictions scored.
    """
    total_scored = 0

    # Build lookup: "player_a vs player_b" → winner
    result_lookup: Dict[str, str] = {}
    for r in live_results:
        if r.get("status") != "closed" or not r.get("winner"):
            continue
        pa, pb, w = r["player_a"], r["player_b"], r["winner"]
        result_lookup[f"{pa}|{pb}"] = w
        result_lookup[f"{pb}|{pa}"] = w

    if not result_lookup:
        return 0

    # Scan all open (non-complete) prediction files
    for pred_file in sorted(REPORTS_DIR.glob("*_predictions.csv")):
        tourney_round = pred_file.stem.replace("_predictions", "")
        complete_file = REPORTS_DIR / f"{tourney_round}_predictions_complete.csv"
        cck_file      = REPORTS_DIR / f"{tourney_round}_predictions_cck.csv"
        cck_complete  = REPORTS_DIR / f"{tourney_round}_predictions_cck_complete.csv"

        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)
        if "correct_prediction" not in df.columns:
            df["correct_prediction"] = pd.NA
        if "correct_prediction_book" not in df.columns:
            df["correct_prediction_book"] = pd.NA

        cck_df = pd.read_csv(cck_file) if cck_file.exists() else df.copy()
        if "correct_prediction" not in cck_df.columns:
            cck_df["correct_prediction"] = pd.NA
        if "correct_prediction_book" not in cck_df.columns:
            cck_df["correct_prediction_book"] = pd.NA

        changed = 0
        for i, row in df.iterrows():
            if pd.notna(df.at[i, "correct_prediction"]):
                continue  # already scored

            pa = alias(str(row["player_a"]))
            pb = alias(str(row["player_b"]))
            key = f"{pa}|{pb}"

            actual_winner = result_lookup.get(key)
            if not actual_winner:
                continue

            # Score model prediction
            pred = alias(str(row["pred_winner"]))
            correct = 1 if actual_winner == pred else 0
            df.at[i, "correct_prediction"] = correct

            # Score CCK prediction
            if i < len(cck_df):
                cck_pred = alias(str(cck_df.at[i, "pred_winner"]))
                cck_df.at[i, "correct_prediction"] = 1 if actual_winner == cck_pred else 0

            # Score book prediction (if odds available)
            if pd.notna(row.get("odds_player_a")) and pd.notna(row.get("odds_player_b")):
                pa_raw = american_to_prob(float(row["odds_player_a"]))
                pb_raw = american_to_prob(float(row["odds_player_b"]))
                book_pred = pa if pa_raw > pb_raw else pb
                book_correct = 1 if actual_winner == book_pred else 0
                df.at[i, "correct_prediction_book"] = book_correct
                if i < len(cck_df):
                    cck_df.at[i, "correct_prediction_book"] = book_correct

            changed += 1

        if changed:
            df.to_csv(complete_file, index=False)
            cck_df.to_csv(cck_complete, index=False)
            print(f"  [live] Scored {changed} new result(s) in {tourney_round}")
            total_scored += changed

            # Check if round is fully complete — if so, update profiles
            cp = df["correct_prediction"].dropna()
            if len(cp) == len(df):
                print(f"  [live] {tourney_round} complete — triggering profile update")
                parts = tourney_round.split("_")
                t_key = parts[0]
                try:
                    cfg = get_config(t_key)
                    seeds = load_best_profiles()
                    raw = completed_to_raw_matches(t_key, cfg)
                    event_log = apply_elo_seeded(raw, seeds)
                    updated = rebuild_profiles_from_event(event_log, seeds, cfg)
                    updated = filter_active_players(updated)
                    out_profiles = profiles_path(t_key)
                    updated.to_csv(out_profiles, index=False)
                    updated.to_csv(PROFILES_LATEST, index=False)
                    print(f"  [live] Profiles updated for {t_key}")
                except Exception as e:
                    print(f"  [live] Profile update failed for {t_key}: {e}")

    return total_scored


def write_live_scores_json(live_results: List[dict]) -> None:
    """Write current live scores to a JSON file for the CourtIQ UI to poll."""
    # Filter to ATP 500+ only
    atp_results = [
        r for r in live_results
        if any(kw in r.get("tournament", "").lower()
               for kw in ["atp", "grand slam", "wimbledon", "australian", "roland", "us open"])
        and "challenger" not in r.get("tournament", "").lower()
        and "doubles" not in r.get("tournament", "").lower()
    ]

    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "matches": atp_results,
    }

    out = Path("docs/live_scores.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  [live] Wrote {len(atp_results)} ATP matches to {out}")


def cmd_live(args):
    """Fetch live scores, auto-score predictions, update profiles if round complete."""
    print(f"\n  [live] Fetching ATP scores...")
    live_results = fetch_live_atp_scores()
    print(f"  [live] Got {len(live_results)} live/completed matches")

    write_live_scores_json(live_results)

    scored = auto_score_from_live(live_results)
    if scored:
        print(f"  [live] Auto-scored {scored} prediction(s)")
    else:
        print(f"  [live] No new results to score")


def cmd_results_all_rounds(args):
    """Score all rounds for a tournament (used by update_profiles workflow)."""
    cfg = get_config(args.tournament)
    for round_code in cfg["rounds"]:
        pred_file = report_path(args.tournament, round_code)
        if not pred_file.exists():
            continue
        # Reuse cmd_results but suppress interactive prompts
        args.round = round_code
        args.manual = False
        print(f"\n  Processing {args.tournament} {round_code}...")
        try:
            cmd_results(args)
        except Exception as e:
            print(f"  Skipping {round_code}: {e}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="courtiq_engine",
        description="CourtIQ — universal tennis prediction pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p1 = sub.add_parser("predict", help="Generate predictions for a round")
    p1.add_argument("--tournament", required=True, help="e.g. madrid2026")
    p1.add_argument("--round", required=True, help="R64, R32, QF, SF, F, etc.")
    p1.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: today)")
    p1.add_argument("--manual", action="store_true", help="Skip auto-fetch, enter draw manually")
    p1.add_argument("--draw-csv", default=None, help="Path to CSV with draw (player_a, player_b, odds_a, odds_b)")

    # results
    p2 = sub.add_parser("results", help="Score predictions and update profiles")
    p2.add_argument("--tournament", required=True)
    p2.add_argument("--round", required=True)
    p2.add_argument("--manual", action="store_true", help="Skip web fetch, enter results manually")

    # status
    p3 = sub.add_parser("status", help="Show tournament progress")
    p3.add_argument("--tournament", required=True)

    # site
    p4 = sub.add_parser("site", help="Rebuild CourtIQ website")
    p4.add_argument("--output", default="./courtiq.html")

    # list-tournaments
    sub.add_parser("list", help="List all supported tournaments")

    # live
    p5 = sub.add_parser("live", help="Fetch live scores and auto-score open predictions")
    p5.add_argument("--tournament", default=None, help="Optional: limit to one tournament")
    p5.add_argument("--round", default=None, help="Optional: limit to one round")

    # results also accepts --all-rounds for CI use
    p2.add_argument("--all-rounds", action="store_true", help="Score all rounds for the tournament")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "results":
        if getattr(args, "all_rounds", False):
            cmd_results_all_rounds(args)
        else:
            cmd_results(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "site":
        cmd_site(args)
    elif args.command == "list":
        print("\n  Supported tournaments:")
        for k, v in sorted(TOURNAMENT_CONFIGS.items()):
            print(f"  {k:<16} {v['name']:<25} {v['surface']:<6} {v['level']}")
        print()


if __name__ == "__main__":
    main()
