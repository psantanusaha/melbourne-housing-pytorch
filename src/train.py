"""
Training script for Melbourne housing price prediction.
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from simple_model import SimpleNN


def load_data():
    """Load and preprocess Melbourne housing data."""
    housingData = pd.read_csv("../data/raw/melb_data.csv")

    numeric_features = [
        'Rooms', 'Bedroom2', 'Bathroom', 'Car',
        'Landsize', 'BuildingArea', 'YearBuilt',
        'Lattitude', 'Longtitude', 'Propertycount', 'Distance'
    ]
    categorical_features = ['Type', 'Regionname']

    # Select and clean data
    data = housingData[numeric_features + categorical_features + ['Price']].dropna()

    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=False)

    # Separate features and target
    X = data_encoded.drop(columns=['Price'])
    y = data_encoded['Price']

    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_tensors(X, y):
    """Convert numpy arrays to PyTorch tensors."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
    return X_tensor, y_tensor


def train_model(model, train_loader, criterion, optimizer, epochs=200):
    """Train the model."""
    print("\nTraining...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_samples = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item() * len(X_batch)
            num_samples += len(X_batch)

        # Print progress
        if epoch % 10 == 0:
            avg_loss = epoch_loss / num_samples
            print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: ${avg_loss:,.2f}")


def evaluate_model(model, X_test, y_test, criterion):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            prediction = model(X_test[i])
            loss = criterion(prediction, y_test[i])
            total_loss += loss.item()

    return total_loss / len(X_test)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Melbourne Housing Price Prediction")
    print("=" * 60)

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_data()

    # Convert to tensors
    X_train_tensor, y_train_tensor = create_tensors(X_train, y_train)
    X_test_tensor, y_test_tensor = create_tensors(X_test, y_test)

    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Create model
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training setup
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, train_loader, criterion, optimizer, epochs=200)

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion)

    print("\n" + "=" * 60)
    print(f"Test Set Average Error: ${test_loss:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()