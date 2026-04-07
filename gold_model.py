import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
from pathlib import Path


MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "gold_price_model.joblib"
FEATURE_COLUMNS = ["SPX", "USO", "SLV", "EUR/USD"]


def load_and_prepare_data(csv_path: str = "gld_price_data.csv"):
    """Load the gold price data and split into features/target and train/test.

    This mirrors the logic from gold.py so the Streamlit app and script
    stay consistent.
    """
    gold = pd.read_csv(csv_path)

    # split features and target (same as in gold.py)
    X = gold.drop(["Date", "GLD"], axis=1)
    y = gold["GLD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )

    return gold, X, y, X_train, X_test, y_train, y_test


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with the exact feature order expected by the model."""
    return df[FEATURE_COLUMNS]


def train_random_forest(X_train, y_train, random_state: int = 3):
    """Train a RandomForestRegressor similar to gold.py."""
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path: Path = MODEL_PATH):
    """Persist a trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path = MODEL_PATH):
    """Load a trained model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run train_model.py first."
        )
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """Compute basic evaluation metrics and predictions on the test set."""
    predictions = model.predict(X_test)

    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return {
        "predictions": predictions,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def get_feature_ranges(X):
    """Return per-feature min and max values for UI controls."""
    mins = X.min()
    maxs = X.max()
    return mins.to_dict(), maxs.to_dict()


def predict_price(model, feature_values: dict):
    """Predict GLD price from a feature dictionary."""
    input_df = pd.DataFrame([feature_values])
    input_df = ensure_feature_order(input_df)
    return float(model.predict(input_df)[0])
