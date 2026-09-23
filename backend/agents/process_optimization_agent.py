"""
Process Optimization Agent
Based on detected anomalies, quality issues, and predicted defects,
generates concrete corrective action recommendations. Distinguishes
between data-driven findings and general knowledge-based recommendations.
"""
from typing import Any

# Data-driven recommendation rules
# Each rule checks a condition and returns a recommendation
RECOMMENDATION_RULES = [
    # Temperature rules
    {
        "condition": lambda d: d.get("temperature", 85) > 105,
        "category": "Temperature",
        "priority": "Immediate",
        "action": "STOP or significantly reduce machine speed. "
                  "Reduce heater setpoint by 10–15 °C. "
                  "Check coolant flow and inspect cooling channels for blockage. "
                  "Inspect bearings for lubrication failure.",
        "basis": "data",
    },
    {
        "condition": lambda d: 95 < d.get("temperature", 85) <= 105,
        "category": "Temperature",
        "priority": "Soon",
        "action": "Reduce heater setpoint by 5–8 °C. "
                  "Verify coolant temperature and flow rate. "
                  "Reduce machine speed by 5–10% to decrease frictional heat.",
        "basis": "data",
    },
    {
        "condition": lambda d: d.get("temperature", 85) < 75,
        "category": "Temperature",
        "priority": "Soon",
        "action": "Increase heater setpoint by 5–10 °C. "
                  "Allow machine warm-up time (minimum 15 minutes). "
                  "Check heater element continuity.",
        "basis": "data",
    },
    # Pressure rules
    {
        "condition": lambda d: d.get("pressure", 5.0) > 6.5,
        "category": "Pressure",
        "priority": "Immediate",
        "action": "Reduce pressure setpoint by 1.0–1.5 bar immediately. "
                  "Inspect flow channel for blockage or solidified material. "
                  "Check pressure relief valve function.",
        "basis": "data",
    },
    {
        "condition": lambda d: 5.5 < d.get("pressure", 5.0) <= 6.5,
        "category": "Pressure",
        "priority": "Soon",
        "action": "Reduce pressure setpoint by 0.5 bar. "
                  "Inspect mold/die for flash buildup. "
                  "Check pressure transducer calibration.",
        "basis": "data",
    },
    {
        "condition": lambda d: d.get("pressure", 5.0) < 4.5,
        "category": "Pressure",
        "priority": "Soon",
        "action": "Increase pressure setpoint by 0.3–0.5 bar. "
                  "Inspect hydraulic lines for leaks. "
                  "Check pump output; service or replace if worn.",
        "basis": "data",
    },
    # Vibration rules
    {
        "condition": lambda d: d.get("vibration", 1.0) > 3.5,
        "category": "Vibration",
        "priority": "Immediate",
        "action": "STOP machine immediately. "
                  "Perform bearing inspection and replacement. "
                  "Check shaft alignment with a dial indicator. "
                  "Do not restart until vibration is below 1.5 mm/s.",
        "basis": "data",
    },
    {
        "condition": lambda d: 1.5 < d.get("vibration", 1.0) <= 3.5,
        "category": "Vibration",
        "priority": "Soon",
        "action": "Reduce machine speed by 10–20%. "
                  "Schedule bearing inspection within 24 hours. "
                  "Check and tighten all mounting fasteners. "
                  "Check coupling alignment.",
        "basis": "data",
    },
    # Machine speed rules
    {
        "condition": lambda d: d.get("machine_speed", 1000) > 1200,
        "category": "Machine Speed",
        "priority": "Immediate",
        "action": "Reduce machine speed to 1000–1050 RPM range. "
                  "High speed is generating excess heat and vibration. "
                  "Check speed controller setpoint.",
        "basis": "data",
    },
    {
        "condition": lambda d: 1050 < d.get("machine_speed", 1000) <= 1200,
        "category": "Machine Speed",
        "priority": "Monitor",
        "action": "Reduce machine speed to within 950–1050 RPM. "
                  "Monitor temperature and vibration trends.",
        "basis": "data",
    },
    # Production rate rules
    {
        "condition": lambda d: d.get("production_rate", 50) < 40,
        "category": "Production Rate",
        "priority": "Soon",
        "action": "Investigate cause of low production rate. "
                  "Check upstream material feed. "
                  "Inspect for machine jam or cycle-time increase. "
                  "Review reject/scrap rate.",
        "basis": "data",
    },
]

# General knowledge-based recommendations (always appended if defect is predicted)
GENERAL_RECOMMENDATIONS = [
    {
        "category": "Quality Inspection",
        "priority": "Immediate",
        "action": "Quarantine parts produced during the abnormal condition period. "
                  "Perform 100% inspection of quarantined parts before shipping.",
        "basis": "knowledge",
    },
    {
        "category": "Calibration",
        "priority": "Soon",
        "action": "Schedule sensor calibration check after resolving the process issue. "
                  "Verify temperature, pressure, and vibration sensor accuracy.",
        "basis": "knowledge",
    },
    {
        "category": "Maintenance Log",
        "priority": "Routine",
        "action": "Record this event in the machine maintenance log. "
                  "Update preventive maintenance schedule if wear is detected.",
        "basis": "knowledge",
    },
]


def run(
    process_data: dict[str, Any],
    monitoring_result: dict,
    quality_result: dict,
    defect_result: dict,
) -> dict:
    """
    Main entry point for the Process Optimization Agent.

    Returns:
        {
          "recommendations": [ { category, priority, action, basis }, ... ],
          "summary": str,
          "immediate_actions": [str, ...],
        }
    """
    recommendations = []

    # Apply data-driven rules
    for rule in RECOMMENDATION_RULES:
        try:
            if rule["condition"](process_data):
                recommendations.append({
                    "category": rule["category"],
                    "priority": rule["priority"],
                    "action": rule["action"],
                    "basis": rule["basis"],
                })
        except Exception:
            pass

    # Append general recommendations if defect was predicted or status is not Normal
    if defect_result.get("defect_predicted") or monitoring_result.get("overall_status") != "Normal":
        recommendations.extend(GENERAL_RECOMMENDATIONS)

    # Deduplicate by category
    seen = set()
    unique_recommendations = []
    for r in recommendations:
        key = r["category"]
        if key not in seen:
            seen.add(key)
            unique_recommendations.append(r)

    # Priority order for sorting
    priority_order = {"Immediate": 0, "Soon": 1, "Monitor": 2, "Routine": 3}
    unique_recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    # Build summary
    immediate = [r["action"] for r in unique_recommendations if r["priority"] == "Immediate"]

    if not unique_recommendations:
        summary = "Process is operating normally. No corrective actions required at this time."
    elif immediate:
        summary = (
            f"IMMEDIATE ACTION REQUIRED: {len(immediate)} critical issue(s) need urgent attention. "
            f"Total {len(unique_recommendations)} recommendation(s) generated."
        )
    else:
        summary = (
            f"{len(unique_recommendations)} corrective action recommendation(s) generated. "
            "No immediate emergency actions required."
        )

    return {
        "recommendations": unique_recommendations,
        "immediate_actions": immediate,
        "summary": summary,
    }
