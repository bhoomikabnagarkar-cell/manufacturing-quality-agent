"""
Defect Prediction using Random Forest Classifier.
Trains on the manufacturing dataset (uses 'defect' column as label).
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ["temperature", "pressure", "vibration", "machine_speed", "production_rate"]
TARGET = "defect"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "defect_scaler.pkl")


def train(csv_path: str) -> dict:
    """Train Random Forest on the CSV data. Returns training summary."""
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Feature importances
    importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))

    return {
        "status": "trained",
        "samples": len(df),
        "test_accuracy": round(report["accuracy"], 4),
        "feature_importances": importances,
    }


def load_model():
    """Load saved model and scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Defect model not found. Run train() first or call /api/train."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(data: dict) -> dict:
    """
    Predict defect for a single process reading.
    data: dict with keys matching FEATURES.
    Returns: { "defect_predicted": bool, "confidence": float, "top_features": list }
    """
    model, scaler = load_model()
    row = np.array([[data[f] for f in FEATURES]], dtype=float)
    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0]
    confidence = float(proba[1]) if pred == 1 else float(proba[0])

    # Top contributing features (based on model importances × deviation from normal)
    normal_means = {"temperature": 85, "pressure": 5.0, "vibration": 1.0,
                    "machine_speed": 1000, "production_rate": 50}
    normal_stds  = {"temperature": 5,  "pressure": 0.3, "vibration": 0.3,
                    "machine_speed": 30,  "production_rate": 3}

    feature_contributions = []
    importances = dict(zip(FEATURES, model.feature_importances_))
    for feat in FEATURES:
        val = data[feat]
        z = abs(val - normal_means[feat]) / (normal_stds[feat] + 1e-9)
        contribution = round(importances[feat] * z, 4)
        feature_contributions.append({"feature": feat, "value": val, "contribution": contribution})

    feature_contributions.sort(key=lambda x: x["contribution"], reverse=True)

    return {
        "defect_predicted": bool(pred == 1),
        "confidence": round(confidence, 4),
        "defect_probability": round(float(proba[1]), 4),
        "top_features": feature_contributions[:3],
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../backend/data/manufacturing_data.csv"
    result = train(csv_path)
    print("Training result:", result)
