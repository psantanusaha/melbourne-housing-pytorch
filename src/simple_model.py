"""
Neural network architecture for housing price prediction.
"""

import torch
import torch.nn as nn
from torch.nn.functional import relu


class SimpleNN(nn.Module):
    """
    Deep neural network for regression.

    Architecture: Input -> 128 -> 64 -> 32 -> Output
    """

    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = relu(x)

        x = self.layer2(x)
        x = relu(x)

        x = self.layer3(x)
        x = relu(x)

        x = self.output(x)
        return x