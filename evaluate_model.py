import neat
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
import visualize
import os

# --- Load Processed Data ---
data = pd.read_csv("processed_football_data.csv")

# --- Team Encoding ---
teams = sorted(set(data["Home"].unique()).union(data["Away"].unique()))
team_to_id = {team: i for i, team in enumerate(teams)}
data["HomeID"] = data["Home"].map(team_to_id)
data["AwayID"] = data["Away"].map(team_to_id)

# --- Features and Targets ---
features = [
    "HomeID", "AwayID", "HomeGF5", "HomeGA5", "AwayGF5", "AwayGA5",
    "H2H_GF", "H2H_GA", "H Shots", "A Shots", "H SOT", "A SOT",
    "H Fouls", "A Fouls", "H Corners", "A Corners",
    "H Yellow", "A Yellow", "H Red", "A Red"
]

X = data[features].values
y = data[["HomeGoals", "AwayGoals"]].values

# --- Normalize Features ---
X_max = np.max(X, axis=0)
X_max[X_max == 0] = 1
X = X / X_max

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Load NEAT Config ---
config_path = "config-feedforward.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# --- Load Best Genome ---
with open("best_genome.pkl", "rb") as f:
    best_genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# --- Predict on Test Set ---
y_pred = np.array([net.activate(xi) for xi in X_test])
y_pred_clipped = np.clip(y_pred, 0, None)  # no negative goals
y_pred_rounded = np.round(y_pred_clipped).astype(int)

# --- MAE: Score Prediction ---
mae_home = mean_absolute_error(y_test[:, 0], y_pred_rounded[:, 0])
mae_away = mean_absolute_error(y_test[:, 1], y_pred_rounded[:, 1])

# --- F1-Score: Win/Draw/Loss Classification ---
def match_result(goals):
    if goals[0] > goals[1]:
        return 1  # Home win
    elif goals[0] < goals[1]:
        return -1  # Away win
    else:
        return 0  # Draw

y_true_result = np.array([match_result(pair) for pair in y_test])
y_pred_result = np.array([match_result(pair) for pair in y_pred_rounded])

f1 = f1_score(y_true_result, y_pred_result, average='macro')

# --- Report ---
print("n📊 Evaluation Results:")
print(f"🏠 MAE (Home Goals): {mae_home:.3f}")
print(f"🛫 MAE (Away Goals): {mae_away:.3f}")
print(f"⚽ Outcome F1 Score (W/D/L): {f1:.3f}")
# --- VISUALIZE BEST GENOME ARCHITECTURE ---


