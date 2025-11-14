"""
Data loading and preprocessing.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class HousingDataset(Dataset):
    """PyTorch Dataset for housing data."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_data(filepath, test_size=0.2, val_size=0.1):
    """Load and preprocess data."""
    
    # Load
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Drop missing target
    df = df.dropna(subset=['Price'])
    
    # Select features (start simple)
    numeric_features = [
        'Rooms', 'Bedroom2', 'Bathroom', 'Car',
        'Landsize', 'BuildingArea', 'YearBuilt',
        'Lattitude', 'Longtitude', 'Propertycount'
    ]
    
    categorical_features = ['Type', 'Regionname']
    
    # Keep only relevant columns
    cols = ['Price'] + numeric_features + categorical_features
    df = df[[col for col in cols if col in df.columns]]
    
    # Fill missing values
    for col in numeric_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Encode categorical
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Remove outliers
    df = df[df['Price'] < 5_000_000]
    
    print(f"After preprocessing: {df.shape}")
    
    # Split features and target
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets and loaders
    train_dataset = HousingDataset(X_train, y_train.values)
    val_dataset = HousingDataset(X_val, y_val.values)
    test_dataset = HousingDataset(X_test, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler
