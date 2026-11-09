from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .agent import DQNAgent
else:
    from agent import DQNAgent


EPISODES = 1_000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SOLVED_REWARD = 200.0
REWARD_WINDOW = 100
REPORT_EVERY = 20
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Runs DQN training, saves the best policy, and stops when the solve threshold is reached.
def train():
    env = gym.make("LunarLander-v3")
    agent = DQNAgent()
    episode_rewards = []
    epsilon = EPSILON_START
    best_average = float("-inf")

    try:
        for episode in range(1, EPISODES + 1):
            state, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.add(state, action, reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            recent_average = float(np.mean(episode_rewards[-REWARD_WINDOW:]))
            if len(episode_rewards) >= REWARD_WINDOW and recent_average > best_average:
                best_average = recent_average
                model_path = PROJECT_ROOT / "models" / "dqn_best.pt"
                save_policy(agent, model_path)

            if episode % REPORT_EVERY == 0:
                print(
                    f"Episode {episode}, average reward over last "
                    f"{min(len(episode_rewards), REWARD_WINDOW)} episodes: "
                    f"{recent_average:.2f}, epsilon: {epsilon:.3f}"
                )

            if len(episode_rewards) >= REWARD_WINDOW and recent_average >= SOLVED_REWARD:
                print(
                    f"Solved: average reward {recent_average:.2f} over the "
                    f"last {REWARD_WINDOW} episodes."
                )
                break
    finally:
        env.close()

    save_training_plot(episode_rewards)


# Saves only the policy network weights to the requested checkpoint path.
def save_policy(agent: DQNAgent, model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.policy_network.state_dict(), model_path)


# Saves episode rewards and their rolling average as a training-curve image.
def save_training_plot(episode_rewards):
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(episode_rewards) + 1)
    rolling_averages = [
        np.mean(episode_rewards[max(0, index - REWARD_WINDOW):index])
        for index in range(1, len(episode_rewards) + 1)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, alpha=0.35, label="Episode reward")
    plt.plot(episodes, rolling_averages, label="Rolling average (up to 100)")
    plt.axhline(SOLVED_REWARD, color="red", linestyle="--", label="Solved threshold (200)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("LunarLander DQN training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curve.png")
    plt.close()


# Starts the training run when this file is executed as a script.
def main():
    train()


if __name__ == "__main__":
    main()