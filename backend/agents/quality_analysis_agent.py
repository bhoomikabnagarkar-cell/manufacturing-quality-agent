"""
Quality Analysis Agent
Analyzes monitored process parameters and determines whether the current
conditions may affect product quality. Identifies contributing parameters
and provides an easy-to-understand explanation.
"""
from typing import Any

# Rules mapping parameter + direction to quality impact
QUALITY_RULES = {
    "temperature": {
        "high": {
            "impact": "High temperature can cause thermal degradation, burn marks, "
                      "warping, and reduced tensile strength in the product.",
            "severity": "high",
        },
        "low": {
            "impact": "Low temperature causes incomplete fill, poor bonding, "
                      "and brittleness in the finished product.",
            "severity": "medium",
        },
    },
    "pressure": {
        "high": {
            "impact": "High pressure causes flashing, cracking, tooling damage, "
                      "and oversized dimensional output.",
            "severity": "high",
        },
        "low": {
            "impact": "Low pressure results in voids, sink marks, short shots, "
                      "and weak weld lines.",
            "severity": "medium",
        },
    },
    "vibration": {
        "high": {
            "impact": "Excessive vibration leads to poor surface finish, "
                      "dimensional variation, and risk of tool breakage.",
            "severity": "high",
        },
    },
    "machine_speed": {
        "high": {
            "impact": "Over-speed generates excess heat, increases vibration, "
                      "and accelerates tool wear, reducing product quality.",
            "severity": "medium",
        },
        "low": {
            "impact": "Under-speed reduces throughput and may cause material "
                      "solidification issues in flow-dependent processes.",
            "severity": "low",
        },
    },
    "production_rate": {
        "low": {
            "impact": "Low production rate may indicate a process bottleneck, "
                      "material flow problem, or upstream equipment issue.",
            "severity": "medium",
        },
        "high": {
            "impact": "Unusually high production rate often signals skipped quality "
                      "checks or measurement errors.",
            "severity": "low",
        },
    },
}

NORMAL_MIDPOINTS = {
    "temperature": 85.0,
    "pressure": 5.0,
    "vibration": 1.0,
    "machine_speed": 1000.0,
    "production_rate": 50.0,
}


def run(process_data: dict[str, Any], monitoring_result: dict) -> dict:
    """
    Main entry point for the Quality Analysis Agent.

    Args:
        process_data: raw sensor values.
        monitoring_result: output from the Process Monitoring Agent.

    Returns:
        {
          "quality_risk": "Low" | "Medium" | "High",
          "quality_issues": [ { parameter, direction, impact, severity }, ... ],
          "contributing_parameters": [str, ...],
          "explanation": str,
        }
    """
    quality_issues = []

    for param in QUALITY_RULES:
        if param not in process_data:
            continue
        value = float(process_data[param])
        midpoint = NORMAL_MIDPOINTS[param]

        direction = "high" if value > midpoint else "low"
        rule = QUALITY_RULES[param].get(direction)

        # Only flag if parameter status is not Normal
        status = monitoring_result["parameter_statuses"].get(param, "Normal")
        if status != "Normal" and rule:
            quality_issues.append({
                "parameter": param,
                "value": value,
                "direction": direction,
                "impact": rule["impact"],
                "severity": rule["severity"],
                "status": status,
            })

    # Determine overall quality risk
    severities = [q["severity"] for q in quality_issues]
    if "high" in severities:
        quality_risk = "High"
    elif "medium" in severities:
        quality_risk = "Medium"
    elif "low" in severities:
        quality_risk = "Low"
    else:
        quality_risk = "Low"

    contributing_parameters = [q["parameter"] for q in quality_issues]

    # Build explanation
    if not quality_issues:
        explanation = (
            "All process parameters are within acceptable ranges. "
            "Product quality is not expected to be affected at this time."
        )
    else:
        parts = []
        for q in quality_issues:
            parts.append(
                f"- {q['parameter'].replace('_', ' ').title()} is {q['direction']} "
                f"({q['value']}): {q['impact']}"
            )
        explanation = (
            f"Quality risk is {quality_risk}. "
            f"The following process issues were identified:\n"
            + "\n".join(parts)
        )

    return {
        "quality_risk": quality_risk,
        "quality_issues": quality_issues,
        "contributing_parameters": contributing_parameters,
        "explanation": explanation,
    }
