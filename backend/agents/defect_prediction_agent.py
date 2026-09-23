"""
Defect Prediction Agent
Wraps the ML defect prediction model and provides structured output
about whether the current process is likely to produce a defective part.
"""
import sys
import os

# Make ml/ importable from anywhere inside the project
ML_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml")
)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

import defect_prediction as dp
import anomaly_detection as ad


def run(process_data: dict, monitoring_result: dict) -> dict:
    """
    Main entry point for the Defect Prediction Agent.

    Args:
        process_data: raw sensor readings (dict).
        monitoring_result: output from Process Monitoring Agent.

    Returns:
        {
          "defect_predicted": bool,
          "defect_probability": float,
          "confidence": float,
          "is_anomaly": bool,
          "anomaly_score": float,
          "top_features": [...],
          "summary": str,
        }
    """
    # Run defect prediction
    defect_result = dp.predict(process_data)

    # Run anomaly detection
    anomaly_result = ad.predict(process_data)

    defect_predicted = defect_result["defect_predicted"]
    defect_prob = defect_result["defect_probability"]
    confidence = defect_result["confidence"]
    top_features = defect_result["top_features"]

    is_anomaly = anomaly_result["is_anomaly"]
    anomaly_score = anomaly_result["anomaly_score"]

    # Build human-readable summary
    if defect_predicted:
        pct = round(defect_prob * 100, 1)
        top_names = [f["feature"].replace("_", " ") for f in top_features[:2]]
        summary = (
            f"DEFECT PREDICTED with {pct}% probability. "
            f"Main contributing factors: {', '.join(top_names)}. "
            f"Immediate quality inspection recommended."
        )
    else:
        pct = round((1 - defect_prob) * 100, 1)
        summary = (
            f"No defect predicted ({pct}% confidence the process is producing good parts). "
        )
        if is_anomaly:
            summary += (
                "However, an anomaly was detected in the sensor data. "
                "Close monitoring is advised."
            )

    return {
        "defect_predicted": defect_predicted,
        "defect_probability": defect_prob,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "top_features": top_features,
        "summary": summary,
    }
