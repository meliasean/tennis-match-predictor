# 🎾 Tennis Match Predictor

A machine learning-based tennis match outcome predictor using ATP match data (2021–2024), player Elo ratings, recent form, and surface-specific performance.

---

## 📊 Features & Model

### ✅ Inputs:
- Player 1 and Player 2 names
- Elo rating delta
- Age, height, seed, and rank deltas
- Recent form (last 5 matches)
- Surface win percentage (per surface)
- Tournament level and round

### 🧠 Model:
- **Random Forest Classifier** (68.3% test accuracy)
- Trained on feature-engineered, chronologically-sorted ATP match data
- Symmetrical data for each match to balance training labels

---

## 🧱 Architecture

### 🔁 Data Pipeline:
1. Load and clean ATP match data (`2021-2024`)
2. Create symmetrical match dataset
3. Assign Elo scores and update dynamically
4. Track recent form and surface stats per player
5. Encode categorical data and generate match feature vectors

### 🔍 Target:
- `outcome` = 1 if Player 1 wins, 0 otherwise

---

## 📦 Requirements

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install -r requirements.txt

🚀 Usage
python
Copy
# Generate prediction
result = predict_match(make_input(
    p1='Carlos Alcaraz',
    p2='Novak Djokovic',
    p1_rank=2, p2_rank=1,
    p1_age=21, p2_age=37,
    surface='Hard',
    round='Final',
    tourney_level='G',
    best_of=5
))

print(result)
📈 Example Output
json
Copy
{
  "predicted_winner": "Player 2",
  "win_probability": 0.47,
  "expected_sets_lost": 1.32,
  "expected_game_win_pct": 0.48
}


⚠️ Disclaimer
This model is for research and entertainment purposes only. It is not intended for gambling or commercial exploitation.

---

### Wimbledon 2025 Tennis Match Predictor 🏆

This project uses a custom ensemble model to predict outcomes for every match in the 2025 Wimbledon Men's Singles tournament. Predictions are based on player profiles updated dynamically after each round.

## Features

- ✅ Predicts each match from R128 through Final
- 🎾 Considers player Elo, surface-specific form, recent match momentum, and head-to-head record
- 📊 Outputs include win probability, expected sets played, and feature deltas with directionality
- 📈 Accuracy Score metrics available for evaluation

## Files

- `*.csv`: Prediction output per round (`R128` → `F`) + final combined file
- `wimbledon_2025_predictions_combined.csv`: All matches, labeled by round
- `ensemble_predict_match()`: Main function for prediction logic
- `update_player_profiles_from_results()`: Updates Elo, form, and surface scores

## Accuracy

📌 **Overall Prediction Accuracy**: ~65.3%  
📌 **Correct Matches**: 83 / 127  
Model is trained on data up to late 2024, and still performs strongly into mid-2025.

## How to Run

1. Ensure `player_profiles`, `model_B`, and `feature_cols` are defined
2. Use round-wise scripts to:
   - Load previous round results
   - Update player profiles
   - Generate and save predictions for the next round


🧑‍💻 Author
Sean Melia
https://github.com/meliasean