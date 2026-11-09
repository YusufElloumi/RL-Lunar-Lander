from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

if __package__:
    from .agent import DQNAgent
else:
    from agent import DQNAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_best.pt"
EPISODES = 10


# Runs ten episodes with greedy actions and reports the mean and standard deviation.
def evaluate(model_path: Path = MODEL_PATH):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/train.py first."
        )

    agent = DQNAgent()
    agent.policy_network.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    env = gym.make("LunarLander-v3")
    episode_rewards = []

    try:
        for episode in range(1, EPISODES + 1):
            state, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f"Episode {episode}: reward = {total_reward:.2f}")
    finally:
        env.close()

    print(
        f"Mean reward: {np.mean(episode_rewards):.2f}; "
        f"standard deviation: {np.std(episode_rewards):.2f}"
    )


# Starts evaluation when this file is executed as a script.
def main():
    evaluate()


if __name__ == "__main__":
    main()