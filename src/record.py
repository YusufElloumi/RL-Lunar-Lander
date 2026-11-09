from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.error import DependencyNotInstalled

if __package__:
    from .agent import DQNAgent
else:
    from agent import DQNAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_best.pt"
VIDEO_DIR = PROJECT_ROOT / "videos"
SUCCESS_REWARD = 200.0
MAX_EPISODES = 100


# Records greedy-policy episodes until one reaches the 200-reward success benchmark.
def record(model_path: Path = MODEL_PATH, max_episodes: int = MAX_EPISODES):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/train.py first."
        )

    agent = DQNAgent()
    agent.policy_network.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        env = gym.wrappers.RecordVideo(
            base_env,
            video_folder=str(VIDEO_DIR),
            episode_trigger=lambda episode_id: True,
            name_prefix="dqn_lunar_lander",
        )
    except DependencyNotInstalled as error:
        base_env.close()
        raise RuntimeError(
            'Video recording requires MoviePy. Install it in the active '
            'environment with: python -m pip install "gymnasium[other]"'
        ) from error
    successful_reward = None

    try:
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            print(f"Recorded episode {episode}: reward = {total_reward:.2f}")
            if total_reward >= SUCCESS_REWARD:
                successful_reward = total_reward
                break
    finally:
        env.close()

    if successful_reward is None:
        raise RuntimeError(
            f"No episode reached reward {SUCCESS_REWARD:.0f} in "
            f"{max_episodes} attempts; clips were saved to {VIDEO_DIR}."
        )

    print(
        f"Successful landing recorded with reward {successful_reward:.2f}. "
        f"Clips are in {VIDEO_DIR}."
    )


# Starts video recording when this file is executed as a script.
def main():
    record()


if __name__ == "__main__":
    main()