import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import pandas as pd
housingData = pd.read_csv('../data/raw/melb_data.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(housingData.shape)
print(housingData.columns)
print(housingData.columns[housingData.isnull().any()])


features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Create your X (features) and y (target)
X = housingData[features]
y = housingData['Price']

# Check missing values in selected features
print(X.isnull().sum())
data_clean = housingData[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']].dropna()
X = data_clean[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']].values
y = data_clean['Price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Feature means after scaling: {X_train.mean(axis=0)}")  # Should be close to 0
print(f"Feature stds after scaling: {X_train.std(axis=0)}")    # Should be close to 1

# Before scaling
print(f"Landsize range: {housingData['Landsize'].min()} to {housingData['Landsize'].max()}")
print(f"Rooms range: {housingData['Rooms'].min()} to {housingData['Rooms'].max()}")

# After scaling
print(f"\nAfter scaling:")
print(f"Landsize range: {X_train[:, 2].min():.2f} to {X_train[:, 2].max():.2f}")
print(f"Rooms range: {X_train[:, 0].min():.2f} to {X_train[:, 0].max():.2f}")



housingData = pd.read_csv("../data/raw/melb_data.csv")
print("Type:", housingData['Type'].nunique(), "unique values")
print("Regionname:", housingData['Regionname'].nunique(), "unique values")
print("Suburb:", housingData['Suburb'].nunique(), "unique values")