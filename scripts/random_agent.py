import gymnasium as gym


# Runs three episodes with uniformly random actions and reports each total reward.
def main():
    env = gym.make("LunarLander-v3", render_mode="human")

    try:
        for episode in range(1, 4):
            observation, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            print(f"Episode {episode}: total reward = {total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()