# 🎾 Tennis Match Predictor (ATP) — Elo + ML + Market-Aware Calibration

A Python project for predicting ATP men’s match outcomes using a feature-engineered dataset (2021–2024+), Elo-based player profiling, and a **Random Forest** baseline model with an optional **CCK calibration layer** designed to behave better against market odds.

This repo is built to do two things well:
1) **Train and evaluate** a strong pre-match prediction model  
2) **Run tournaments round-by-round** (R128 → F) while **updating player profiles** after results

---

## What this predicts

For any match `(player_a vs player_b)` the prediction outputs include:

- Predicted winner
- Win probability for each player
- Confidence
- Feature deltas (Elo, surface Elo, form, surface win rates, etc.)
- (Optional) market-aware fields in the CCK output (devigged book probability, p_elo, p_temp)

---

## Model & Accuracy

### Baseline model
- **Random Forest Classifier**
- Trained on ATP match data (2021–2024+) with chronological splits
- Symmetrical data generation (each match represented from both player perspectives)

### Current offline performance (holdout test)
- **Test accuracy: 70.23%** (latest run)

> Note: match prediction “accuracy” is a blunt metric. This repo also supports probability-style evaluation (calibration, edge vs. book, etc.) where you can get a more honest picture than raw accuracy alone.

---

## Key Ideas in the Feature Set

### Core signals
- **Elo delta** (overall)  
- **Surface Elo delta** (Hard/Clay/Grass)  
- **Recent form** (rolling win-rate windows, streak)  
- **Surface performance priors** (win rates by surface)  
- **Workload / rest** (matches in last 28 days, rest-day priors)  
- **Tournament context** (surface, level, round, best-of)

### Optional market-aware layer (CCK)
CCK takes the Standard model probability and adjusts it via:
- temperature scaling (`TEMP_T`)
- evidence shrink toward 0.5 when data is thin (`C_SURF`, `K_RECENT`, `W_INFO_BLEND`)
- blending with Elo logistic (`ELO_LAMBDA`)
- blending with the devigged market probability (`MKT_LAMBDA`)

---

## Repo Structure (typical)

