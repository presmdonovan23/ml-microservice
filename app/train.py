"""
XGBoost Model Training Script
Downloads handwritten digits dataset and trains a classifier
"""
from ucimlrepo import fetch_ucirepo
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data():
    """Download and load the optical recognition of handwritten digits dataset"""
    print("Fetching handwritten digits dataset from UCI ML Repository...")
    digits = fetch_ucirepo(id=80)

    X = digits.data.features
    y = digits.data.targets

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")

    return X, y


def preprocess_data(X, y):
    """Split data into train, validation, and test sets"""
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: separate train and validation (70% train, 15% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 H 0.15
    )

    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_model(X_train, X_val, y_train, y_val):
    """Train XGBoost classifier with early stopping"""
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': 10,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    print("\nTraining XGBoost model...")
    evals = [(dtrain, 'train'), (dval, 'validation')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=20
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def save_model(model, filepath="artifacts/model.json"):
    """Save trained model to JSON file"""
    model.save_model(filepath)
    print(f"\nModel saved to {filepath}")


def main():
    """Main training pipeline"""
    # Load data
    X, y = load_data()

    # Preprocess and split
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)

    # Train model
    model = train_model(X_train, X_val, y_train, y_val)

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    print("\nScaler saved to artifacts/scaler.pkl")


if __name__ == "__main__":
    main()
