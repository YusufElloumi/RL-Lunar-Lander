import torch
from torch import nn


class QNetwork(nn.Module):
    """Estimate the value of each action from a LunarLander state."""

    # Initializes the two hidden layers and the four-action output layer.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    # Maps one or more eight-number states to four action-value estimates.
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


# Runs a smoke test with one randomly generated LunarLander state.
def main():
    network = QNetwork()
    random_state = torch.randn(1, 8)
    action_values = network(random_state)
    print("Q-values for the four actions:", action_values.squeeze(0))


if __name__ == "__main__":
    main()