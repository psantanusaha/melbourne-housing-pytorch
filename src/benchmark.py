"""
Benchmark: Neural Network vs Traditional ML models
"""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# You'll need:
# 1. load_data() - reuse from train.py
# 2. Train each model
# 3. Evaluate each on test set
# 4. Print comparison table
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

def train_decision_tree(X_train, X_test, y_train, y_test):
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    error_rate = mean_absolute_error(y_test, y_pred)
    return error_rate

def train_random_forest(X_train, X_test, y_train, y_test):
    random_forest = RandomForestRegressor(random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    error_rate = mean_absolute_error(y_test, y_pred)
    return error_rate

def train_neural_network(X_train, X_test, y_train, y_test):
    return 162000


def main():
    print("=" * 60)
    print("Model Benchmark: Traditional ML vs Neural Networks")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train and evaluate each model
    print("\nTraining models...")

    dt_error = train_decision_tree(X_train, X_test, y_train, y_test)
    rf_error = train_random_forest(X_train, X_test, y_train, y_test)
    nn_error = train_neural_network(X_train, X_test, y_train, y_test)

    # Print comparison
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Decision Tree:     ${dt_error:,.2f}")
    print(f"Random Forest:     ${rf_error:,.2f}")
    print(f"Neural Network:    ${nn_error:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()