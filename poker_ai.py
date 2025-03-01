import tensorflow as tf
import numpy as np
import random
import os
from collections import Counter
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb

# ----------------------------- #
#          Deck & Mapping       #
# ----------------------------- #
def create_deck():
    """Creates a standard 52-card deck.

    Returns:
        list: A list of tuples, each representing a card (rank, suit).
    """
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    return [(rank, suit) for rank in ranks for suit in suits]

rank_values = {rank: idx for idx, rank in enumerate(["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"], start=2)}
suit_values = {suit: idx for idx, suit in enumerate(["Hearts", "Diamonds", "Clubs", "Spades"])}

# ----------------------------- #
#       Feature Engineering     #
# ----------------------------- #
def create_features(hole_cards, community_cards):
    """Creates numerical features for a poker hand.

    Args:
        hole_cards (list): A list of tuples representing the player's hole cards.
        community_cards (list): A list of tuples representing the community cards.

    Returns:
        np.ndarray: A NumPy array representing the features.
    """
    full_hand = hole_cards + community_cards
    ranks = sorted([rank_values[card[0]] for card in full_hand])
    suits = [suit_values[card[1]] for card in full_hand]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Explicitly represent community card ranks (up to 5 cards)
    community_card_features = [0] * 5
    community_card_features[:len(community_cards)] = [rank_values[card[0]] for card in community_cards]

    # Feature vector:
    #   - Ranks of all cards (up to 7)
    #   - Ranks of community cards (up to 5, padded)
    #   - Counts of each rank
    #   - Counts of each suit
    #   - Flush and straight indicators
    features = ranks + community_card_features + [rank_counts.get(r, 0) for r in range(2, 15)] + [suit_counts.get(s, 0) for s in range(4)]
    features += [int(is_flush(suits)), int(is_straight(ranks))]

    # Pad the features to a constant length (7 ranks + 5 community ranks + 13 rank counts + 4 suit counts + 2 indicators)
    return np.pad(features, (0, 7 - len(ranks)), 'constant')[:7 + 5 + 13 + 4 + 2]

def is_flush(suits):
    """Checks if a hand contains a flush (all cards of the same suit)."""
    return len(set(suits)) == 1

def is_straight(ranks):
    """Checks if a hand contains a straight (5 consecutive ranks)."""
    if len(set(ranks)) < 5:
        return False  # Not enough unique ranks for a straight
    
    # Handle Ace-low straight (A-2-3-4-5)
    if set(ranks) == {14, 2, 3, 4, 5}:
        return True

    return max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5

# ----------------------------- #
#        Dataset Generation     #
# ----------------------------- #
def generate_hand(num_hole_cards, num_community_cards, exclude=None):
    """Generates a random poker hand.

    Args:
        num_hole_cards (int): The number of hole cards to generate.
        num_community_cards (int): The number of community cards to generate.
        exclude (list, optional): A list of cards to exclude from the deck.

    Returns:
        tuple: A tuple containing the hole cards and community cards (both as lists of tuples).
    """
    deck = set(create_deck())
    if exclude:
        deck -= set(exclude)
    hole_cards = random.sample(list(deck), num_hole_cards)
    deck -= set(hole_cards)
    community_cards = random.sample(list(deck), num_community_cards)
    return hole_cards, community_cards


def rank_hand(hole_cards, community_cards):
    """Ranks a poker hand (improved version).

    Returns a tuple: (hand_rank, high_card_rank, kicker_ranks)
    hand_rank: 9=Straight Flush, 8=Four of a Kind, ..., 1=High Card
    """
    full_hand = hole_cards + community_cards
    ranks = sorted([rank_values[card[0]] for card in full_hand], reverse=True)
    suits = [suit_values[card[1]] for card in full_hand]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    is_flush_ = is_flush(suits)
    is_straight_ = is_straight(ranks)
    
    # Adjust ranks for Ace-low straights (A-2-3-4-5)
    if set(ranks) == {14, 5, 4, 3, 2}:
        ranks = [5, 4, 3, 2, 1]
        is_straight_ = True

    if is_straight_ and is_flush_:
        return (9, max(ranks), [])  # Straight Flush

    four_of_a_kind = [rank for rank, count in rank_counts.items() if count == 4]
    if four_of_a_kind:
        kicker = [r for r in ranks if r != four_of_a_kind[0]][0]
        return (8, four_of_a_kind[0], [kicker])  # Four of a Kind

    three_of_a_kind = [rank for rank, count in rank_counts.items() if count == 3]
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    if three_of_a_kind and pairs:
        return (7, three_of_a_kind[0], [pairs[0]]) # Full House
    if is_flush_:
        return (6, max(ranks), sorted([r for r in ranks], reverse=True)[:5])  # Flush

    if is_straight_:
        return (5, max(ranks), [])  # Straight

    if three_of_a_kind:
        kickers = sorted([r for r in ranks if r != three_of_a_kind[0]], reverse=True)[:2]
        return (4, three_of_a_kind[0], kickers)  # Three of a Kind
    
    if len(pairs) == 2:
        kickers = sorted([r for r in ranks if r not in pairs], reverse=True)[:1]
        return (3, max(pairs), [min(pairs)] + kickers)  # Two Pair

    if pairs:
        kickers = sorted([r for r in ranks if r != pairs[0]], reverse=True)[:3]
        return (2, pairs[0], kickers) # Pair
    return (1, max(ranks), sorted(ranks, reverse=True)[:5])  # High Card

def create_dataset(samples=10000, num_hole_cards=2, num_community_cards=5):
    """Creates a dataset of poker hands and their ranks.

    Args:
        samples (int): The number of samples to generate.
        num_hole_cards (int): The number of hole cards per hand.
        num_community_cards (int): The number of community cards per hand.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the rank vector (y).
    """
    X, y = [], []
    for _ in range(samples):
        hole_cards, community_cards = generate_hand(num_hole_cards, num_community_cards)
        features = create_features(hole_cards, community_cards)
        rank, _, _ = rank_hand(hole_cards, community_cards)  # Use the new ranking function
        X.append(features)
        y.append(rank)
    return np.array(X), np.array(y)

# ----------------------------- #
#        Model Building         #
# ----------------------------- #
def build_model(input_dim, model_type='nn'):
    """Builds a neural network or XGBoost model.

    Args:
        input_dim (int): The dimensionality of the input features.
        model_type (str): The type of model ('nn' for neural network, 'xgb' for XGBoost).

    Returns:
        object: The compiled model (either a Keras model or an XGBoost parameter dictionary).
    """
    if model_type == 'nn':
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    elif model_type == 'xgb':
        return {
            'objective': 'reg:squarederror',
            'max_depth': 6,  # Maximum depth of trees
            'eta': 0.1,      # Learning rate
            'subsample': 0.8, # Subsample ratio of the training instances
            'colsample_bytree': 0.8, # Subsample ratio of columns when constructing each tree
            'seed': 42       # Random seed for reproducibility
        }

def train_models(X_train, y_train, X_val, y_val, epochs=100):
    """Trains both a neural network and an XGBoost model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        epochs (int): The number of training epochs.

    Returns:
        tuple: A tuple containing the trained neural network and XGBoost models.
    """
    nn_model = build_model(X_train.shape[1], 'nn')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    nn_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=epochs, batch_size=64, callbacks=callbacks, verbose=1)
    nn_model.save('nn_model.h5')

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = build_model(None, 'xgb')  # Get XGBoost parameters
    xgb_model = xgb.train(params, dtrain, num_boost_round=1000,  # Increased boosting rounds
                          evals=[(dtrain, 'train'), (dval, 'eval')],
                          early_stopping_rounds=50, verbose_eval=100)  # Increased patience
    xgb_model.save_model('xgb_model.json')

    return nn_model, xgb_model

# ----------------------------- #
#         Evaluation            #
# ----------------------------- #
def load_models():
    """Loads trained models from disk.

    Returns:
        tuple: A tuple containing the loaded neural network and XGBoost models (or None if not found).
    """
    nn_model = tf.keras.models.load_model('nn_model.h5') if os.path.exists('nn_model.h5') else None
    xgb_model = xgb.Booster()
    if os.path.exists('xgb_model.json'):
        xgb_model.load_model('xgb_model.json')
    else:
        xgb_model = None
    return nn_model, xgb_model

def evaluate_hand(hole_cards, community_cards, models):
    """Evaluates a poker hand using the trained models.

    Args:
        hole_cards (list): A list of tuples representing the player's hole cards.
        community_cards (list): A list of tuples representing the community cards.
        models (tuple): A tuple containing the neural network and XGBoost models.

    Returns:
        dict: A dictionary containing the predictions from each model.
    """
    try:
        features = create_features(hole_cards, community_cards).reshape(1, -1)
        nn_model, xgb_model = models
        preds = {}
        if nn_model:
            preds['nn'] = nn_model.predict(features, verbose=0)[0][0]
        if xgb_model:
            preds['xgb'] = xgb_model.predict(xgb.DMatrix(features))[0]
        return preds
    except (ValueError, KeyError) as e:
        print(f"Error evaluating hand: {e}")
        return {}  # Return empty dictionary on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def calculate_win_prob(hole_cards, community_cards, num_players=2, simulations=1000):
    """Calculates the win probability of a poker hand using Monte Carlo simulation.

    Args:
        hole_cards (list):  Player's hole cards.
        community_cards (list): Community cards.
        num_players (int): Number of players in the game.
        simulations (int):  Number of simulation runs.

    Returns:
        tuple: (win probability, tie probability, loss probability)
    """
    player_rank, _, _ = rank_hand(hole_cards, community_cards)
    deck = set(create_deck()) - set(hole_cards) - set(community_cards)
    wins, ties, losses = 0, 0, 0

    if len(deck) < (num_players - 1) * len(hole_cards): #Not Enough Cards Check
        print("Not enough cards in the deck for the simulation.")
        return 0.0, 0.0, 0.0

    for _ in range(simulations):
        try:
            opp_hands = []
            remaining_deck = list(deck) #Deck copy each simulation.
            for _ in range(num_players - 1):
                opp_hole_cards = random.sample(remaining_deck, len(hole_cards))
                remaining_deck = [card for card in remaining_deck if card not in opp_hole_cards]
                opp_rank, _, _ = rank_hand(opp_hole_cards, community_cards)
                opp_hands.append(opp_rank)

            if all(player_rank > opp_rank for opp_rank in opp_hands):
                wins += 1
            elif any(player_rank == opp_rank for opp_rank in opp_hands):
                ties += 1
            else:
                losses += 1
        except ValueError as e:
            # Handle cases where there aren't enough cards left
            print(f"Error during simulation: {e}.  Likely not enough cards left.")
            return 0.0, 0.0, 0.0 #Return 0 probabilities
        except Exception as e:
            print(f"Unexpected error during simulation: {e}")
            return 0.0, 0.0, 0.0


    probs = (wins / simulations, ties / simulations, losses / simulations)
    print(f"Wins: {probs[0]*100:.2f}%, Ties: {probs[1]*100:.2f}%, Losses: {probs[2]*100:.2f}%")
    return probs

# ----------------------------- #
#   Expected Value Calculation   #
# ----------------------------- #
def calculate_ev(win_prob, pot_size, bet_size):
    """Calculates the expected value (EV) of a bet.

    Args:
        win_prob (float): The probability of winning the hand.
        pot_size (float): The current size of the pot.
        bet_size (float): The size of the bet.

    Returns:
        float: The expected value of the bet.
    """
    # Calculate expected value
    losses = (1 - win_prob) * bet_size
    ev = (win_prob * pot_size) - losses
    return ev

# ----------------------------- #
#  Suggest Bet Function         #
# ----------------------------- #
def suggest_bet(win_prob, pot_size=100, bet_size=20):
    """Suggests a betting action based on win probability and pot odds.

    Args:
        win_prob (tuple): Win, tie, and loss probabilities (from calculate_win_prob).
        pot_size (float): The current size of the pot.
        bet_size (float): The size of the current bet.

    Returns:
        str: A suggested betting action ('Raise', 'Call', or 'Fold').
    """
    pot_odds = bet_size / (pot_size + bet_size) if pot_size + bet_size > 0 else 0
    win_probability = win_prob[0]  # Use the win probability
    ev = calculate_ev(win_probability, pot_size, bet_size)

    if ev >= 0:  # Positive expected value
        if win_probability >= 0.70:
            # Strong hand, raise (up to half the pot size)
            raise_amount = min(pot_size / 2, pot_size)
            return f"Raise by {raise_amount:.2f}"
        elif win_probability >= 0.40 and win_probability >= pot_odds:
            # Decent hand, call if pot odds are favorable
            return "Call"
        else:
            return "Fold"  # Positive EV, but not strong enough to justify a call
    else:
        return "Fold"  # Negative expected value

# ----------------------------- #
#              Main             #
# ----------------------------- #
def main():
    """Main function for the poker simulation."""
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Step 1: Create dataset
    print("Generating dataset...")
    X, y = create_dataset(samples=10000)

    # Step 2: Split dataset
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Step 3: Train models
    print("Training models...")
    nn_model, xgb_model = train_models(X_train, y_train, X_val, y_val, epochs=100)

    # Step 4: Load models
    print("Loading models...")
    models = load_models()

    # Step 5: Game simulation loop
    print("\nWelcome to the Poker Simulation!")
    total_chips = float(input("Enter total chips: "))

    while True:
        try:
            num_players = int(input("\nEnter number of players (including yourself): "))
            if num_players < 2:
                raise ValueError("There must be at least 2 players.")

            print("Enter your hole cards (e.g., 'Ace of Hearts, King of Spades'): ")
            hole_cards_input = input().strip().split(',')
            hole_cards = []
            for card_str in hole_cards_input:
                parts = card_str.strip().split(' of ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid card format: '{card_str}'.  Use 'Rank of Suit'.")
                rank, suit = parts
                if rank not in rank_values or suit not in suit_values:
                    raise ValueError(f"Invalid card: '{card_str}'.  Check rank and suit.")
                hole_cards.append((rank, suit))

            if len(hole_cards) != 2: #Check for valid number of cards.
                raise ValueError("You must enter exactly two hole cards.")


            community_cards_input = input("Enter community cards (or press Enter to skip): ")
            community_cards = []
            if community_cards_input:
                community_cards_input_list = community_cards_input.strip().split(',')
                for card_str in community_cards_input_list:
                    parts = card_str.strip().split(' of ')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid card format: '{card_str}'. Use 'Rank of Suit'.")
                    rank, suit = parts
                    if rank not in rank_values or suit not in suit_values:
                        raise ValueError(f"Invalid card: '{card_str}'. Check rank and suit.")
                    community_cards.append((rank, suit))

            if len(community_cards) > 5: #Check number of community cards.
                raise ValueError("There can be at most 5 community cards.")


            pot_size = float(input("Enter current pot size: "))
            bet_size = float(input("Enter size of the current bet: "))

            opponents_actions = []
            for i in range(num_players - 1):
                action = input(f"Enter action for opponent {i + 1} (Check, Call, Raise, Fold): ").strip().lower()
                if action not in ['check', 'call', 'raise', 'fold']:
                    raise ValueError("Invalid action. Please enter Check, Call, Raise, or Fold.")
                opponents_actions.append(action)


            # Evaluate hand
            print("\nEvaluating hand...")
            preds = evaluate_hand(hole_cards, community_cards, models)
            if preds: # Check if the evaluation had errors.
                print(f"Predictions: {preds}")
            else:
                print("Could not evaluate hand. Check inputs.")
                continue #Skip to the next iteration

            # Calculate win probability
            win_prob = calculate_win_prob(hole_cards, community_cards, num_players=num_players)

            # Suggest betting action
            if all(p == 0.0 for p in win_prob):
                print("Could not calculate win probability. Skipping betting suggestion.")
            else:
                suggested_action = suggest_bet(win_prob, pot_size=pot_size, bet_size=bet_size)
                print(f"Suggested action: {suggested_action}")

            if input("\nType 'exit' to stop or press Enter to continue...").lower() == 'exit':
                break

        except ValueError as e:
            print(f"Error: {e}. Please enter valid values.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("Exiting the game. Goodbye!")

if __name__ == "__main__":
    main()
