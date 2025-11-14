"""
Training script.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import SimpleNN
from data_loader import load_data


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    """Main training function."""
    
    # Config
    DATA_PATH = "../data/raw/melbourne_housing.csv"
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Melbourne Housing Price Prediction - Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader, scaler = load_data(DATA_PATH)
    
    # Get input dimension from first batch
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]
    print(f"Input features: {input_dim}")
    
    # Create model
    print("\n2. Creating model...")
    model = SimpleNN(input_dim, hidden_dim=64).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\n3. Training...")
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path('../models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), '../models/best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Test
    print("\n4. Testing...")
    model.load_state_dict(torch.load('../models/best_model.pth'))
    test_loss = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest MSE: {test_loss:.4f}")
    print(f"Test RMSE: ${test_loss**0.5:,.0f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
