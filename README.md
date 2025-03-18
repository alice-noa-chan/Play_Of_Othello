# Play\_Of\_Othello

This repository contains a reinforcement learning (RL) environment for Reversi (Othello) and an AI agent trained using Proximal Policy Optimization (PPO) from Stable-Baselines3.

## Features

- Custom `ReversiEnv` environment compatible with OpenAI Gym.
- Self-play reinforcement learning with an opponent policy that updates periodically.
- Reward shaping to encourage good gameplay behavior.
- Custom callbacks for progress tracking and model saving.

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Training Process

1. **Environment Setup:** The `ReversiEnv` class defines the game environment, handling legal moves, rewards, and self-play mechanics.
2. **Agent Initialization:** If a trained model exists (`reversi_ai_model_final.zip`), it is loaded. Otherwise, a new PPO agent is initialized with a multi-layer perceptron (MLP) policy.
3. **Training:** The PPO agent plays against itself, with periodic updates to the opponent policy to improve competitiveness.
4. **Reward Shaping:**
   - Incremental rewards for gaining more pieces.
   - Penalties for illegal moves.
   - Final reward scaling based on game outcome.
5. **Model Saving:** The model is saved every 5,000 steps for checkpointing.

## Training Execution

Run the training script:

```bash
python train.py
```

This will train the agent for `400,000` timesteps while periodically saving model checkpoints and updating the opponent policy.

## Environment Details

- **State Representation:** `8x8` numpy array where `1` represents the agent's pieces, `-1` represents the opponent's pieces, and `0` represents empty spaces.
- **Action Space:** `65` discrete actions (placing a piece on `0-63` board indices + `64` for pass).
- **Opponent Strategy:**
  - If a trained model exists, the opponent plays using the same policy.
  - Otherwise, the opponent selects moves randomly.

## Custom Callbacks

1. `TqdmCallback`: Displays a progress bar for training.
2. `SaveModelCallback`: Saves model checkpoints every 5,000 steps.
3. `OpponentUpdateCallback`: Updates the opponent policy every 5,000 steps for dynamic self-play.

## Model Saving and Loading

- The trained model is saved as `reversi_ai_model_final.zip`.
- To continue training, the script automatically loads an existing model if available.

## Running Inference

To test the trained model, download `play_othello_{version}.zip` from the release page, extract it, and install dependencies using `requirements.txt`. The model file is included in the archive.

## License

This project is released under the MIT License.

