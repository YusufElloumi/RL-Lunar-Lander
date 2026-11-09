# Project goal

Build a Deep Q-Network (DQN) agent from scratch with PyTorch to solve Gymnasium's LunarLander-v3 environment.

# Environment facts

- State: 8 continuous numbers.
- Actions: 4 discrete actions — 0 (nothing), 1 (left engine), 2 (main engine), and 3 (right engine).
- Reward: A scalar signal supplied by the environment at each step; an episode's reward is the sum of its step rewards.
- Solved threshold: An average reward of 200 or more over 100 episodes.

# Algorithm choice and why DQN

DQN applies semi-gradient TD learning when the action-value function is represented by a neural network. The neural network replaces the q table, a replay buffer breaks up correlated samples by providing random minibatches of past transitions, and a separate target network stabilizes the TD target.

# File map

- `src/__init__.py` — Marks `src` as a Python package.
- `src/network.py` — Defines the neural network that estimates one action value per action.
- `src/replay_buffer.py` — Stores transitions and samples minibatches for learning.
- `src/agent.py` — Implements action selection and DQN learning updates.
- `src/train.py` — Runs the training loop and saves the best model and reward plot.
- `src/evaluate.py` — Evaluates a saved model without exploration.
- `src/record.py` — Records video clips of the trained agent.
- `scripts/random_agent.py` — Runs a baseline agent that selects random actions.
- `models/.gitkeep` — Keeps the model output directory in version control.
- `plots/.gitkeep` — Keeps the plot output directory in version control.
- `videos/.gitkeep` — Keeps the video output directory in version control.

# Hyperparameters

| Hyperparameter | Value |
|---|---|

# Progress log

- 2026-11-08: Added and smoke-tested the two-hidden-layer PyTorch Q-network.
- 2026-11-08: Added a three-episode random-action baseline with human rendering.
- 2026-11-08: Created the initial project layout and project guide.

# Known issues and lessons learned

- None recorded yet.
