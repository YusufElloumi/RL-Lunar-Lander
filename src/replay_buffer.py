import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Keep past transitions and return random minibatches for DQN."""

    # Creates a buffer that automatically discards its oldest items at capacity.
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    # Adds one state, action, reward, next-state, and done transition.
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Samples transitions uniformly and returns them as batches of tensors.
    def sample(self, batch_size: int):
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

# Runs a small test that adds transitions and checks sampled tensor shapes.
def main():
    buffer = ReplayBuffer(capacity=5)

    for index in range(3):
        state = [float(index)] * 8
        next_state = [float(index + 1)] * 8
        buffer.add(state, index % 4, float(index), next_state, index == 2)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=2)
    print("Tensor shapes:", states.shape, actions.shape, rewards.shape,
          next_states.shape, dones.shape)
    print("Sampled actions:", actions)


if __name__ == "__main__":
    main()