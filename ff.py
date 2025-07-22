import pandas as pd
import numpy as np
import neat
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self, csv_path='England CSV.csv'):
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.config = None
        self.best_genome = None
        self.best_net = None
        self.class_weights = None
        self.feature_names = [
            'home_form', 'away_form', 'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg', 'h2h_home_advantage',
            'home_shots_avg', 'away_shots_avg', 'home_sot_avg', 'away_sot_avg',
            'home_wins_ratio', 'away_wins_ratio', 'goal_difference_home', 'goal_difference_away'
        ]
        self.load_and_preprocess_data(csv_path)
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the football data"""
        try:
            self.df = pd.read_csv(csv_path, parse_dates=['Date'])
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
            # Clean data
            self.df = self.df.dropna(subset=['FTH Goals', 'FTA Goals', 'FT Result'])
            
            # Encode teams
            all_teams = list(set(self.df['HomeTeam'].tolist() + self.df['AwayTeam'].tolist()))
            self.team_encoder.fit(all_teams)
            
            print(f"Loaded {len(self.df)} matches with {len(all_teams)} unique teams")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def calculate_team_metrics(self, team, date, matches_back=8):
        """Calculate comprehensive team metrics"""
        # Get team's recent matches
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) & 
            (self.df['Date'] < date)
        ].sort_values('Date', ascending=False).head(matches_back)
        
        if len(team_matches) == 0:
            return {
                'form': 0.5, 'goals_scored': 1.5, 'goals_conceded': 1.5,
                'shots': 12, 'sot': 4, 'wins_ratio': 0.33, 'goal_difference': 0
            }
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        shots = 0
        sot = 0
        wins = 0
        
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                # Team playing at home
                goals_scored += row['FTH Goals']
                goals_conceded += row['FTA Goals']
                shots += row.get('H Shots', 12)
                sot += row.get('H SOT', 4)
                
                if row['FT Result'] == 'H':
                    points += 3
                    wins += 1
                elif row['FT Result'] == 'D':
                    points += 1
            else:
                # Team playing away
                goals_scored += row['FTA Goals']
                goals_conceded += row['FTH Goals']
                shots += row.get('A Shots', 12)
                sot += row.get('A SOT', 4)
                
                if row['FT Result'] == 'A':
                    points += 3
                    wins += 1
                elif row['FT Result'] == 'D':
                    points += 1
        
        num_matches = len(team_matches)
        return {
            'form': points / (num_matches * 3),  # Normalize to 0-1
            'goals_scored': goals_scored / num_matches,
            'goals_conceded': goals_conceded / num_matches,
            'shots': shots / num_matches,
            'sot': sot / num_matches,
            'wins_ratio': wins / num_matches,
            'goal_difference': (goals_scored - goals_conceded) / num_matches
        }
    
    def calculate_h2h_metrics(self, home_team, away_team, date, matches_back=10):
        """Calculate head-to-head metrics"""
        h2h_matches = self.df[
            (((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) | 
             ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))) & 
            (self.df['Date'] < date)
        ].sort_values('Date', ascending=False).head(matches_back)
        
        if len(h2h_matches) == 0:
            return 0.5  # Neutral
        
        home_advantage = 0
        total_matches = len(h2h_matches)
        
        for _, row in h2h_matches.iterrows():
            if row['HomeTeam'] == home_team:
                if row['FT Result'] == 'H':
                    home_advantage += 1
                elif row['FT Result'] == 'D':
                    home_advantage += 0.5
            else:
                if row['FT Result'] == 'A':
                    home_advantage += 1
                elif row['FT Result'] == 'D':
                    home_advantage += 0.5
        
        return home_advantage / total_matches
    
    def extract_features(self):
        """Extract features for all matches"""
        features = []
        targets = []
        
        print("Extracting features from matches...")
        
        for i, row in self.df.iterrows():
            if i % 500 == 0:
                print(f"Processing match {i}/{len(self.df)}")
            
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            date = row['Date']
            
            # Calculate team metrics
            home_metrics = self.calculate_team_metrics(home_team, date)
            away_metrics = self.calculate_team_metrics(away_team, date)
            
            # Calculate head-to-head
            h2h = self.calculate_h2h_metrics(home_team, away_team, date)
            
            # Create feature vector
            feature_vector = [
                home_metrics['form'],
                away_metrics['form'],
                home_metrics['goals_scored'],
                home_metrics['goals_conceded'],
                away_metrics['goals_scored'],
                away_metrics['goals_conceded'],
                h2h,
                home_metrics['shots'],
                away_metrics['shots'],
                home_metrics['sot'],
                away_metrics['sot'],
                home_metrics['wins_ratio'],
                away_metrics['wins_ratio'],
                home_metrics['goal_difference'],
                away_metrics['goal_difference']
            ]
            
            # Target: match result (0=Home win, 1=Draw, 2=Away win)
            if row['FT Result'] == 'H':
                target = 0
            elif row['FT Result'] == 'D':
                target = 1
            else:
                target = 2
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def prepare_data(self, test_size=0.2, validation_size=0.1):
        """Prepare and split the data"""
        print("Preparing data...")
        features, targets = self.extract_features()
        
        # Remove any invalid features
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features = features[valid_indices]
        targets = targets[valid_indices]
        
        print(f"Valid samples: {len(features)}")
        print(f"Class distribution: {Counter(targets)}")
        
        # Calculate class weights for imbalanced data
        self.class_weights = compute_class_weight(
            'balanced', classes=np.unique(targets), y=targets
        )
        print(f"Class weights: {dict(zip([0, 1, 2], self.class_weights))}")
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Stratified split to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(sss.split(features_normalized, targets))
        
        X_train, X_test = features_normalized[train_idx], features_normalized[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        # Further split training data for validation
        if validation_size > 0:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=validation_size/(1-test_size), random_state=42)
            train_idx, val_idx = next(sss_val.split(X_train, y_train))
            
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_train, X_test, y_train, y_test
    
    def create_neat_config(self):
        """Create optimized NEAT configuration"""
        config_text = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.05
activation_options      = sigmoid relu tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.05
aggregation_options     = sum product

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.4
bias_mutate_rate        = 0.8
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.6

# connection add/remove rates
conn_add_prob           = 0.4
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.02

feed_forward            = True
initial_connection      = partial_direct 0.5

# node add/remove rates
node_add_prob           = 0.3
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 15
num_outputs             = 3

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.1
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.1
response_mutate_rate    = 0.1
response_replace_rate   = 0.05

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 2.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.8
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.15

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 3

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.25
min_species_size   = 2
"""
        
        config_path = 'optimized_config.txt'
        with open(config_path, 'w') as f:
            f.write(config_text)
        
        return config_path
    
    def evaluate_genome(self, genome, config, X_train, y_train, X_val=None, y_val=None):
        """Evaluate a single genome with improved fitness calculation"""
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Training predictions
            train_predictions = []
            for xi in X_train:
                output = net.activate(xi)
                prediction = np.argmax(output)
                train_predictions.append(prediction)
            
            # Calculate weighted F1 score on training data
            train_f1 = f1_score(y_train, train_predictions, average='weighted', zero_division=0)
            train_accuracy = accuracy_score(y_train, train_predictions)
            
            # Penalize for class imbalance in predictions
            pred_distribution = Counter(train_predictions)
            balance_penalty = 0
            for class_id in [0, 1, 2]:
                expected_ratio = np.sum(y_train == class_id) / len(y_train)
                actual_ratio = pred_distribution.get(class_id, 0) / len(train_predictions)
                balance_penalty += abs(expected_ratio - actual_ratio)
            
            # Calculate complexity penalty
            complexity_penalty = len(genome.connections) * 0.001 + len(genome.nodes) * 0.002
            
            # Validation score if available
            validation_bonus = 0
            if X_val is not None and y_val is not None:
                val_predictions = []
                for xi in X_val:
                    output = net.activate(xi)
                    prediction = np.argmax(output)
                    val_predictions.append(prediction)
                
                val_f1 = f1_score(y_val, val_predictions, average='weighted', zero_division=0)
                validation_bonus = val_f1 * 20  # Bonus for good validation performance
            
            # Combined fitness score
            fitness = (train_f1 * 100 + 
                      train_accuracy * 50 + 
                      validation_bonus - 
                      balance_penalty * 10 - 
                      complexity_penalty)
            
            return max(fitness, 0.01)  # Ensure positive fitness
            
        except Exception as e:
            return 0.01  # Very low fitness for invalid genomes
    
    def train_neat(self, generations=75):
        """Train the NEAT algorithm with improved evaluation"""
        print("Starting NEAT training...")
        
        # Prepare data with validation split
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(validation_size=0.15)
        
        # Store test data for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Create config
        config_path = self.create_neat_config()
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path
        )
        
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                fitness = self.evaluate_genome(genome, config, X_train, y_train, X_val, y_val)
                genome.fitness = fitness
        
        # Create population
        population = neat.Population(self.config)
        
        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Add checkpointing
        checkpoint_reporter = neat.Checkpointer(generation_interval=10)
        population.add_reporter(checkpoint_reporter)
        
        # Run evolution
        self.best_genome = population.run(eval_genomes, generations)
        
        # Create the best network
        self.best_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        
        # Save the best genome
        with open('best_football_genome.pkl', 'wb') as f:
            pickle.dump((self.best_genome, self.config), f)
        
        print("Training completed!")
        return self.best_genome
    
    def evaluate_model(self, X=None, y=None):
        """Evaluate the trained model"""
        if X is None or y is None:
            X, y = self.X_test, self.y_test
        
        if self.best_net is None:
            print("Model not trained yet!")
            return None
        
        predictions = []
        prediction_probs = []
        
        for xi in X:
            output = self.best_net.activate(xi)
            prediction = np.argmax(output)
            predictions.append(prediction)
            prediction_probs.append(output)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        f1_macro = f1_score(y, predictions, average='macro')
        
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        
        # Detailed classification report
        target_names = ['Home Win', 'Draw', 'Away Win']
        print(f"\nDetailed Classification Report:")
        print(classification_report(y, predictions, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'predictions': predictions,
            'probabilities': prediction_probs
        }
    
    def predict_match(self, home_team, away_team, date=None):
        """Predict the outcome of a match"""
        if self.best_net is None:
            print("Model not trained yet!")
            return None
        
        if date is None:
            date = self.df['Date'].max()
        
        # Calculate features
        home_metrics = self.calculate_team_metrics(home_team, date)
        away_metrics = self.calculate_team_metrics(away_team, date)
        h2h = self.calculate_h2h_metrics(home_team, away_team, date)
        
        feature_vector = [
            home_metrics['form'],
            away_metrics['form'],
            home_metrics['goals_scored'],
            home_metrics['goals_conceded'],
            away_metrics['goals_scored'],
            away_metrics['goals_conceded'],
            h2h,
            home_metrics['shots'],
            away_metrics['shots'],
            home_metrics['sot'],
            away_metrics['sot'],
            home_metrics['wins_ratio'],
            away_metrics['wins_ratio'],
            home_metrics['goal_difference'],
            away_metrics['goal_difference']
        ]
        
        # Normalize features
        feature_vector_normalized = self.scaler.transform([feature_vector])[0]
        
        # Make prediction
        output = self.best_net.activate(feature_vector_normalized)
        prediction = np.argmax(output)
        confidence = np.max(output)
        
        outcomes = ['Home Win', 'Draw', 'Away Win']
        
        print(f"\nPrediction for {home_team} vs {away_team}:")
        print(f"Predicted Outcome: {outcomes[prediction]}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Probabilities: Home {output[0]:.3f}, Draw {output[1]:.3f}, Away {output[2]:.3f}")
        
        return {
            'prediction': prediction,
            'outcome': outcomes[prediction],
            'confidence': confidence,
            'probabilities': output
        }

# Usage example
if __name__ == "__main__":
    # Initialize the predictor
    predictor = FootballPredictor('England CSV.csv')
    
    # Train the model
    best_genome = predictor.train_neat(generations=100)
    
    # Evaluate the model
    results = predictor.evaluate_model()
    
    # Check if F1 score target is met
    if results and results['f1_weighted'] >= 0.75:
        print(f"\n✅ SUCCESS! F1 Score of {results['f1_weighted']:.4f} exceeds the 75% target!")
    else:
        print(f"\n❌ F1 Score of {results['f1_weighted']:.4f} is below the 75% target.")
    
    # Make example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # You can replace these with actual team names from your dataset
    example_matches = [
        ("Man United", "Liverpool"),
        ("Chelsea", "Arsenal"),
        ("Man City", "Tottenham")
    ]
    
    for home, away in example_matches:
        try:
            predictor.predict_match(home, away)
        except Exception as e:
            print(f"Could not predict {home} vs {away}: {e}")