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
| Discount factor (gamma) | 0.99 |
| Learning rate | 0.0005 |
| Batch size | 64 |
| Replay buffer capacity | 100000 transitions |
| Epsilon start | 1.0 |
| Epsilon minimum | 0.01 |
| Epsilon decay per episode | 0.995 |
| Target network update interval | Every 1000 learning steps |

# Progress log

- 2026-11-08: Re-rendered the successful trained-agent GIF at the environment's native 600x400 resolution.
- 2026-11-08: Re-recorded the random baseline and 50-episode intermediate-policy GIFs with full-size frames captured throughout each episode.
- 2026-11-08: Added random-baseline, 30-episode intermediate-training, and successful trained-agent GIF demos to `videos/` and linked them in the README.
- 2026-11-08: Completed the README with setup/run instructions, project structure, hyperparameters, and placeholders for demo clips and evaluation results.
- 2026-11-08: Added greedy evaluation and video recording that continues until a successful landing.
- 2026-11-08: Added the LunarLander DQN training loop, best-model checkpointing, and reward-curve plotting.
- 2026-11-08: Added and smoke-tested epsilon-greedy action selection and the DQN learning update.
- 2026-11-08: Added and smoke-tested the replay buffer with random minibatch sampling.
- 2026-11-08: Added and smoke-tested the two-hidden-layer PyTorch Q-network.
- 2026-11-08: Added a three-episode random-action baseline with human rendering.
- 2026-11-08: Created the initial project layout and project guide.

# Known issues and lessons learned

- Gymnasium's `RecordVideo` requires MoviePy, which is not currently listed in `requirements.txt`; install the Gymnasium `other` extra to record MP4 clips.
- The intermediate-training demo is from a separate 50-episode run; it is illustrative and not a checkpoint from the main training run.
- Final evaluation results still need to be refreshed after the desired training run.
