# RL-Lunar-Lander

A from-scratch DQN project for Gymnasium's LunarLander-v3. The project is being
built in phases; it currently includes a random-action baseline and an
untrained PyTorch Q-network smoke test. Training and evaluation are not
implemented yet, so the agent does not learn to land.

## Current functionality

- `scripts/random_agent.py` runs three episodes with random actions and opens
  the LunarLander render window. It prints each episode's total reward.
- `src/network.py` defines a neural network that maps an 8-number state to
  four Q-value estimates, one for each action. Run it with a randomly
  generated test input to print those estimates. The network is untrained, so
  the values are not yet meaningful action recommendations.
- The project folders for the future agent, replay buffer, training, model
  checkpoints, plots, and videos are in place, but those features are not
  implemented yet.

## Run the current tests

```bash
.venv/bin/python scripts/random_agent.py
.venv/bin/python src/network.py
```
