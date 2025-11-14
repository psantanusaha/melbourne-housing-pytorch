"""
Neural network architectures for housing price prediction.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """Simple feedforward neural network - baseline model."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class DeepNN(nn.Module):
    """Deep neural network with dropout."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(DeepNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Test the models
    x = torch.randn(32, 10)  # batch_size=32, features=10
    
    model = SimpleNN(input_dim=10)
    print(f"SimpleNN output: {model(x).shape}")
    
    model = DeepNN(input_dim=10)
    print(f"DeepNN output: {model(x).shape}")
