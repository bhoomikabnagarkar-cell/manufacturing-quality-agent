"""
Anomaly Detection using Isolation Forest.
Trains on the manufacturing dataset and saves the model.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["temperature", "pressure", "vibration", "machine_speed", "production_rate"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "anomaly_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "anomaly_scaler.pkl")


def train(csv_path: str) -> dict:
    """Train Isolation Forest on the CSV data. Returns training summary."""
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.15,   # ~15% of data expected to be anomalous
        random_state=42,
    )
    model.fit(X_scaled)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate on training data (simple self-check)
    preds = model.predict(X_scaled)
    n_anomalies = int(np.sum(preds == -1))

    return {
        "status": "trained",
        "samples": len(df),
        "anomalies_detected": n_anomalies,
        "contamination": 0.15,
    }


def load_model():
    """Load saved model and scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Anomaly model not found. Run train() first or call /api/train."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(data: dict) -> dict:
    """
    Predict whether a single process reading is anomalous.
    data: dict with keys matching FEATURES.
    Returns: { "is_anomaly": bool, "anomaly_score": float }
    """
    model, scaler = load_model()
    row = np.array([[data[f] for f in FEATURES]], dtype=float)
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]          # 1 = normal, -1 = anomaly
    score = float(model.decision_function(row_scaled)[0])  # negative = more anomalous
    return {
        "is_anomaly": bool(pred == -1),
        "anomaly_score": round(score, 4),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly columns to a DataFrame."""
    model, scaler = load_model()
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    df = df.copy()
    df["anomaly_flag"] = (preds == -1).astype(int)
    df["anomaly_score"] = np.round(scores, 4)
    return df


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../backend/data/manufacturing_data.csv"
    result = train(csv_path)
    print("Training result:", result)
