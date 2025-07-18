import neat
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# --- LOAD PROCESSED DATA ---
data = pd.read_csv("processed_football_data.csv")

# --- TEAM ENCODING ---
teams = sorted(set(data["Home"].unique()).union(data["Away"].unique()))
team_to_id = {team: i for i, team in enumerate(teams)}
data["HomeID"] = data["Home"].map(team_to_id)
data["AwayID"] = data["Away"].map(team_to_id)

# --- FEATURES & TARGETS ---
features = [
    "HomeID", "AwayID", "HomeGF5", "HomeGA5", "AwayGF5", "AwayGA5",
    "H2H_GF", "H2H_GA", "H Shots", "A Shots", "H SOT", "A SOT",
    "H Fouls", "A Fouls", "H Corners", "A Corners",
    "H Yellow", "A Yellow", "H Red", "A Red"
]

X = data[features].values
y = data[["HomeGoals", "AwayGoals"]].values

# --- NORMALIZATION ---
X_max = np.max(X, axis=0)
X_max[X_max == 0] = 1
X = X / X_max

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- NEAT TRAINING FUNCTION ---
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        error = 0.0
        for xi, yi in zip(X_train, y_train):
            output = net.activate(xi)
            if len(output) >= 2:
                error += ((output[0] - yi[0])**2 + (output[1] - yi[1])**2)
            else:
                error += 1000
        genome.fitness = 1.0 / (error + 1)

# --- NEAT SETUP ---
config_path = "config-feedforward.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

# --- TRAIN NEAT ---
winner = p.run(eval_genomes, 50)

# --- SAVE BEST GENOME ---
with open("best_genome.pkl", "wb") as f:
    pickle.dump(winner, f)

print("\n✅ Training complete. Best genome saved as 'best_genome.pkl'.")

# --- PREDICT FUNCTION ---
def predict(home_team, away_team, stats):
    with open("best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    home_id = team_to_id.get(home_team, 0)
    away_id = team_to_id.get(away_team, 0)

    input_vector = [home_id, away_id] + stats
    input_vector = np.array(input_vector)
    input_vector = input_vector / X_max

    output = net.activate(input_vector)
    home_goals, away_goals = round(max(0, output[0])), round(max(0, output[1]))

    print(f"\n🔮 Predicted Score:")
    print(f"🏠 {home_team}: {home_goals}")
    print(f"🛫 {away_team}: {away_goals}")

    if home_goals > away_goals:
        print("🏆 Winner: Home Team")
    elif away_goals > home_goals:
        print("🏆 Winner: Away Team")
    else:
        print("⚖️ Draw")

# 🔹 Example usage:
# stats = [HomeGF5, HomeGA5, AwayGF5, AwayGA5, H2H_GF, H2H_GA, ...]
# predict("Chelsea", "Arsenal", stats)
