# CourtIQ Engine — README

Replaces all 168 notebook cells with a single script. One command per action.

---

## Setup

Drop `courtiq_engine.py` into your project root (same folder as `./models/`, `./reports/`, `./data/`).

```
your_project/
├── courtiq_engine.py
├── courtiq_template.html   ← rename your courtiq_v2.html to this
├── models/
│   └── rf_model.joblib
├── reports/
│   ├── player_profiles_latest.csv
│   └── ... (existing prediction CSVs stay here)
└── data/
    └── ... (match datasets)
```

---

## Daily workflow

### 1. Before first round — create a draw CSV

```csv
player_a,odds_a,player_b,odds_b
Carlos Alcaraz,-500,Qualifier,+380
Jannik Sinner,-450,Qualifier,+350
```

Odds are optional. Save it as `draws/madrid2026_R64.csv`.

### 2. Generate predictions

```bash
# Using your draw CSV
python courtiq_engine.py predict --tournament madrid2026 --round R64 --draw-csv draws/madrid2026_R64.csv

# After R64, the R32 draw is inferred automatically from results
python courtiq_engine.py predict --tournament madrid2026 --round R32

# Manual entry if needed
python courtiq_engine.py predict --tournament madrid2026 --round R32 --manual
```

### 3. After matches finish — enter results

```bash
# Engine tries to fetch from ATP website first
python courtiq_engine.py results --tournament madrid2026 --round R64

# Force manual entry
python courtiq_engine.py results --tournament madrid2026 --round R64 --manual
```

This scores predictions, updates ELO and profiles, and saves `_complete.csv` files
in the same format as your existing ones.

### 4. Check progress

```bash
python courtiq_engine.py status --tournament madrid2026
```

### 5. Rebuild the website

```bash
python courtiq_engine.py site
# custom output path:
python courtiq_engine.py site --output ./docs/index.html
```

---

## Full tournament example

```bash
# R64 draw is announced
python courtiq_engine.py predict --tournament madrid2026 --round R64 --draw-csv draws/madrid2026_R64.csv

# R64 finishes
python courtiq_engine.py results --tournament madrid2026 --round R64

# R32 draw auto-inferred, predict immediately
python courtiq_engine.py predict --tournament madrid2026 --round R32

# R32 finishes
python courtiq_engine.py results --tournament madrid2026 --round R32

# Continue: QF, SF, F...

# After the final
python courtiq_engine.py site
```

---

## Adding a new tournament

Add one entry to `TOURNAMENT_CONFIGS` in `courtiq_engine.py`:

```python
"houston": {
    "name": "Houston",
    "surface": "Clay",
    "level": "A",
    "best_of": 3,
    "rounds": ["R32", "R16", "QF", "SF", "F"]
},
```

Also add to `TOURNEY_ORDER` and `TOURNEY_DISPLAY_NAMES` so it appears in CourtIQ.

---

## Retired / inactive players

Edit `INACTIVE_PLAYERS` in `courtiq_engine.py` to add retirees:

```python
INACTIVE_PLAYERS = {
    "Rafael Nadal", "Roger Federer",
    "New Retiree Name",  # add here
}
```

Players with no match in 180 days are automatically flagged as inactive.

---

## Serve stats

The engine pulls serve statistics (aces, 1st serve %, break point conversion, etc.)
from Jeff Sackmann's tennis_atp GitHub repo during site rebuilds.

License: CC BY-NC-SA 4.0 — attribution required, non-commercial only.
If CourtIQ becomes a paid product, you'll need an alternative data source.

---

## Automation

Add to cron (Mac/Linux) or Task Scheduler (Windows):

```bash
# Check for results every evening at 11pm
0 23 * * * cd /path/to/project && python courtiq_engine.py results --tournament madrid2026 --round R64
```

Auto-fetching from the ATP website is best-effort — if it fails, the script
falls back to manual entry. For reliable automation, The Odds API (~$10/month)
gives structured results data with no scraping needed.
