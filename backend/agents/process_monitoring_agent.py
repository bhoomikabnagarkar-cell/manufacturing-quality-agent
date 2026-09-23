"""
Process Monitoring Agent
Reads manufacturing process data, evaluates each parameter against
defined thresholds, and returns a status: Normal / Warning / Critical.
"""
from typing import Any

# Thresholds for each parameter
THRESHOLDS = {
    "temperature": {
        "normal":   (75.0, 95.0),
        "warning":  (70.0, 105.0),
        # outside warning range → Critical
    },
    "pressure": {
        "normal":   (4.5, 5.5),
        "warning":  (4.0, 6.5),
    },
    "vibration": {
        "normal":   (0.5, 1.5),
        "warning":  (0.0, 3.4),
    },
    "machine_speed": {
        "normal":   (950.0, 1050.0),
        "warning":  (900.0, 1200.0),
    },
    "production_rate": {
        "normal":   (45.0, 55.0),
        "warning":  (38.0, 60.0),
    },
}


def _classify_parameter(value: float, param: str) -> str:
    """Return Normal / Warning / Critical for a single parameter value."""
    t = THRESHOLDS[param]
    lo_n, hi_n = t["normal"]
    lo_w, hi_w = t["warning"]

    if lo_n <= value <= hi_n:
        return "Normal"
    elif lo_w <= value <= hi_w:
        return "Warning"
    else:
        return "Critical"


def run(process_data: dict[str, Any]) -> dict:
    """
    Main entry point for the Process Monitoring Agent.

    Args:
        process_data: dict containing at minimum the sensor keys.

    Returns:
        {
          "overall_status": "Normal" | "Warning" | "Critical",
          "parameter_statuses": { param: status, ... },
          "abnormal_parameters": [ { "parameter", "value", "status" }, ... ],
          "summary": str
        }
    """
    parameter_statuses = {}
    abnormal = []

    for param in THRESHOLDS:
        if param not in process_data:
            continue
        value = float(process_data[param])
        status = _classify_parameter(value, param)
        parameter_statuses[param] = status
        if status != "Normal":
            abnormal.append({
                "parameter": param,
                "value": value,
                "status": status,
                "normal_range": THRESHOLDS[param]["normal"],
            })

    # Overall status: worst among all parameters
    if any(p["status"] == "Critical" for p in abnormal):
        overall = "Critical"
    elif any(p["status"] == "Warning" for p in abnormal):
        overall = "Warning"
    else:
        overall = "Normal"

    # Human-readable summary
    if overall == "Normal":
        summary = "All process parameters are within normal operating range."
    elif overall == "Warning":
        names = [p["parameter"] for p in abnormal]
        summary = (
            f"WARNING: {len(abnormal)} parameter(s) outside normal range — "
            f"{', '.join(names)}. Monitor closely and prepare to intervene."
        )
    else:
        crit = [p["parameter"] for p in abnormal if p["status"] == "Critical"]
        summary = (
            f"CRITICAL: {len(crit)} parameter(s) require immediate attention — "
            f"{', '.join(crit)}. Stop or slow the process immediately."
        )

    return {
        "overall_status": overall,
        "parameter_statuses": parameter_statuses,
        "abnormal_parameters": abnormal,
        "summary": summary,
    }
