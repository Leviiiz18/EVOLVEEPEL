from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import neat
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

# Top 10 EPL teams
TEAMS = [
    "Manchester City",
    "Arsenal", 
    "Manchester United",
    "Newcastle United",
    "Liverpool",
    "Brighton & Hove Albion",
    "Aston Villa",
    "Tottenham Hotspur",
    "Brentford",
    "Fulham"
]

# Global variables for model
neat_model = None
config = None
X_max = None
team_to_id = {team: i for i, team in enumerate(TEAMS)}


def load_model():
    """Load the trained NEAT model and configuration"""
    global neat_model, config, X_max

    try:
        # Load the NEAT configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward.txt')
        if not os.path.exists(config_path):
            raise FileNotFoundError("config-feedforward.txt not found. Please ensure it's in the same directory as app.py")

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError("best_genome.pkl not found. Please ensure it's in the same directory as app.py")

        with open(model_path, 'rb') as f:
            neat_model = pickle.load(f)

        # Load X_max for normalization
        x_max_path = os.path.join(os.path.dirname(__file__), 'x_max.npy')
        if os.path.exists(x_max_path):
            X_max = np.load(x_max_path)
        else:
            # Fallback in case not available
            X_max = np.ones(20)

        print("✅ NEAT model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def encode_teams_features(home_team, away_team, stats):
    """
    Encode teams and input stats (feature vector) for prediction.
    Includes team IDs and feature normalization.
    """
    if home_team not in team_to_id or away_team not in team_to_id:
        raise ValueError("Invalid team selection")

    home_id = team_to_id[home_team]
    away_id = team_to_id[away_team]

    input_vector = [home_id, away_id] + stats
    input_vector = np.array(input_vector)
    input_vector = input_vector / X_max  # Normalize

    return input_vector


def predict_match(home_team, away_team, stats):
    """
    Use NEAT network to predict match outcome
    """
    global neat_model, config

    if neat_model is None or config is None:
        raise RuntimeError("Model not loaded")

    # Prepare input vector
    input_vector = encode_teams_features(home_team, away_team, stats)

    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(neat_model, config)

    # Get prediction
    output = net.activate(input_vector)

    # Assume output is [home_goals, away_goals]
    home_goals = max(0, round(output[0]))
    away_goals = max(0, round(output[1]))

    if home_goals > away_goals:
        winner = home_team
    elif away_goals > home_goals:
        winner = away_team
    else:
        winner = "Draw"

    return {
        'predicted_home_goals': home_goals,
        'predicted_away_goals': away_goals,
        'winner': winner
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify({'teams': TEAMS})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        home_team = data.get('home_team')
        away_team = data.get('away_team')
        stats = data.get('stats')

        if not home_team or not away_team or not stats:
            return jsonify({'error': 'home_team, away_team and stats are required'}), 400

        if home_team == away_team:
            return jsonify({'error': 'Home and away teams cannot be the same'}), 400

        if home_team not in TEAMS or away_team not in TEAMS:
            return jsonify({'error': 'Invalid team selection'}), 400

        if not isinstance(stats, list) or len(stats) != 18:
            return jsonify({'error': 'Invalid stats vector. Must be a list of 18 features.'}), 400

        result = predict_match(home_team, away_team, stats)

        return jsonify({
            'success': True,
            'prediction': result,
            'home_team': home_team,
            'away_team': away_team
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    model_loaded = neat_model is not None and config is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'teams_count': len(TEAMS)
    })


if __name__ == '__main__':
    if load_model():
        print("🚀 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Server not started.")
        print("📝 Make sure you have:")
        print("   - config-feedforward.txt")
        print("   - best_genome.pkl")
        print("   - x_max.npy")
        print("   - neat-python installed: pip install neat-python")
