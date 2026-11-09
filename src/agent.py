import random
from collections.abc import Sequence

import torch
from torch import nn
from torch.optim import Adam

if __package__:
    from .network import QNetwork
    from .replay_buffer import ReplayBuffer
else:
    from network import QNetwork
    from replay_buffer import ReplayBuffer


class DQNAgent:
    """Choose actions and learn action values using the DQN algorithm."""

    # Creates policy and target networks, optimizer, and replay memory.
    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.0005,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_every: int = 1_000,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.policy_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.requires_grad_(False)
        self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    # Selects a random action with probability epsilon, otherwise the best estimated action.
    def select_action(self, state: Sequence[float], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(4)

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        return int(action_values.argmax(dim=1).item())

    # Updates the policy network from a sampled minibatch; returns None until enough data exists.
    def learn(self) -> float | None:
        if len(self.replay_buffer.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        predicted_values = self.policy_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_values * (1 - dones)

        loss = self.loss_function(predicted_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return float(loss.item())


# Runs a small smoke test of action selection and one DQN update.
def main():
    agent = DQNAgent(batch_size=2, buffer_size=4, target_update_every=1)
    state = [0.0] * 8

    for action in (0, 1):
        agent.replay_buffer.add(state, action, 1.0, state, False)

    print("Selected action:", agent.select_action(state, epsilon=0.0))
    print("One minibatch loss:", agent.learn())


if __name__ == "__main__":
    main()