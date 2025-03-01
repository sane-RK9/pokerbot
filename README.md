# Poker AI Simulator

A machine learning-based poker hand evaluation and decision-making system that helps players analyze their hands and make optimal betting decisions.

## Overview

This program uses machine learning models (Neural Networks and XGBoost) to evaluate poker hands and calculate win probabilities. It provides actionable betting suggestions based on hand strength, pot odds, and expected value calculations.

## Features

- **Hand Evaluation**: Uses trained ML models to evaluate the strength of poker hands
- **Win Probability Calculation**: Simulates thousands of possible outcomes to determine the probability of winning with your current hand
- **Expected Value Calculation**: Determines the mathematical expectation of different betting actions
- **Betting Suggestions**: Recommends whether to fold, call, or raise based on hand strength and pot odds
- **Multi-player Support**: Accounts for multiple opponents when calculating win probabilities

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- NumPy
- XGBoost
- Random (standard library)
- Collections (standard library)
- OS (standard library)

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install tensorflow numpy xgboost
```

## Usage

Run the main script to start the poker simulator:

```bash
python poker_ai.py
```

Follow the interactive prompts to:
1. Enter your total chips
2. Specify the number of players
3. Input your hole cards (e.g., 'Ace of Hearts, King of Spades')
4. Input any community cards (optional, leave blank for pre-flop)
5. Enter the current pot size and bet size
6. Input the actions of your opponents

The system will then:
- Evaluate your hand using the trained models
- Calculate your win probability through simulations
- Suggest an optimal betting action (fold, call, or raise)

## How It Works

### Data Generation
The system creates a dataset of poker hands with their corresponding values for training.

### Feature Engineering
Each hand is transformed into a set of features including:
- Card ranks and suits
- Rank and suit frequency counts
- Straight and flush indicators
- Community card information

### Model Training
Two models are trained:
1. A neural network with multiple dense layers
2. An XGBoost regression model

Both models predict the value of a given hand.

### Win Probability Calculation
The system simulates random opponent hands and community cards to determine how often your hand would win.

### Betting Strategy
Based on win probability, pot odds, and expected value calculations, the system suggests optimal betting actions.

## Customization

You can modify various parameters in the code:
- Adjust the `simulations` parameter in `calculate_win_prob()` to balance speed vs. accuracy
- Modify the thresholds in `suggest_bet()` to change betting aggression
- Adjust model hyperparameters for different training outcomes

## Files

- `poker_ai.py` - The main script containing all functionality

## Future Improvements

- Support for more poker variants beyond basic hand evaluation
- Opponent modeling based on past actions
- Improved UI/UX for easier interaction
- Web or mobile interface for on-the-fly calculations
- More sophisticated betting strategies based on game theory

