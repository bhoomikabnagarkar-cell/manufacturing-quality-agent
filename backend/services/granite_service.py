"""
Gemini LLM Service
Connects to Google Gemini API to generate manufacturing quality explanations.
Falls back gracefully to a rule-based summary if GEMINI_API_KEY is not set.

Public interface (unchanged from the previous granite_service):
    generate_explanation(process_data, monitoring_result, quality_result,
                         defect_result, optimization_result, rag_context) -> str
"""
import os
from typing import Optional

# ── Optional Google GenAI SDK import ─────────────────────────────────────────
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ── Prompt builder (unchanged logic, updated header text) ────────────────────

def _build_prompt(
    process_data: dict,
    monitoring_result: dict,
    quality_result: dict,
    defect_result: dict,
    optimization_result: dict,
    rag_context: str,
) -> str:
    """Build the prompt to send to Gemini."""
    params = "\n".join(
        f"  {k}: {v}"
        for k, v in process_data.items()
        if k not in ("timestamp", "quality_status", "defect")
    )
    abnormal = monitoring_result.get("abnormal_parameters", [])
    abnormal_str = (
        ", ".join(f"{p['parameter']} ({p['status']})" for p in abnormal)
        if abnormal else "None"
    )
    top_features = defect_result.get("top_features", [])
    top_feat_str = (
        ", ".join(f["feature"] for f in top_features[:3])
        if top_features else "None"
    )
    recs = optimization_result.get("recommendations", [])
    recs_str = "\n".join(
        f"  [{r['priority']}] {r['category']}: {r['action']}" for r in recs[:5]
    )

    prompt = f"""You are a manufacturing quality control AI assistant.
Use the manufacturing knowledge below to generate a clear, concise quality report.

=== MANUFACTURING KNOWLEDGE (from knowledge base) ===
{rag_context}

=== CURRENT PROCESS READINGS ===
{params}

=== MONITORING STATUS ===
Overall Status: {monitoring_result.get('overall_status')}
Abnormal Parameters: {abnormal_str}

=== QUALITY ANALYSIS ===
Quality Risk: {quality_result.get('quality_risk')}
Explanation: {quality_result.get('explanation')}

=== DEFECT PREDICTION ===
Defect Predicted: {defect_result.get('defect_predicted')}
Defect Probability: {defect_result.get('defect_probability', 0) * 100:.1f}%
Top Contributing Factors: {top_feat_str}
Anomaly Detected: {defect_result.get('is_anomaly')}

=== RECOMMENDED ACTIONS ===
{recs_str}

=== INSTRUCTIONS ===
Write a concise manufacturing quality report (4-6 sentences) that:
1. States the overall process status and whether a defect is likely.
2. Explains the main cause(s) based on the abnormal parameters.
3. States which product quality issue(s) are expected if the process continues.
4. Recommends the most important corrective action.
Keep the language simple and suitable for a factory floor operator.
Do NOT repeat the raw numbers - explain the meaning instead.

Quality Report:"""
    return prompt


# ── Fallback (rule-based) explanation ────────────────────────────────────────

def _fallback_explanation(
    monitoring_result: dict,
    quality_result: dict,
    defect_result: dict,
    optimization_result: dict,
    reason: str = "",
) -> str:
    """Generate a rule-based explanation when Gemini is not available."""
    status   = monitoring_result.get("overall_status", "Normal")
    risk     = quality_result.get("quality_risk", "Low")
    defect   = defect_result.get("defect_predicted", False)
    prob     = defect_result.get("defect_probability", 0.0)
    abnormal = monitoring_result.get("abnormal_parameters", [])
    immediate = optimization_result.get("immediate_actions", [])

    lines = []
    lines.append(
        f"The manufacturing process is currently in {status} status "
        f"with {risk} quality risk."
    )

    if abnormal:
        param_list = ", ".join(p["parameter"].replace("_", " ") for p in abnormal)
        lines.append(
            f"The following parameters are outside their normal operating range: {param_list}."
        )

    if defect:
        lines.append(
            f"The defect prediction model estimates a {prob * 100:.1f}% probability "
            f"that the current process will produce a defective part."
        )
    else:
        lines.append(
            f"The defect prediction model indicates the process is unlikely to produce "
            f"defective parts (defect probability: {prob * 100:.1f}%)."
        )

    if immediate:
        lines.append(f"Immediate action required: {immediate[0]}")
    elif optimization_result.get("recommendations"):
        first_rec = optimization_result["recommendations"][0]["action"]
        lines.append(f"Recommended action: {first_rec}")
    else:
        lines.append("No corrective actions are required at this time.")

    if reason:
        lines.append(f"(Note: {reason})")

    return " ".join(lines)


# ── Public interface (same signature as the previous granite_service) ─────────

def generate_explanation(
    process_data: dict,
    monitoring_result: dict,
    quality_result: dict,
    defect_result: dict,
    optimization_result: dict,
    rag_context: str,
) -> str:
    """
    Generate a natural-language quality report using Google Gemini.
    Falls back to rule-based text if GEMINI_API_KEY is missing or the call fails.
    """
    api_key   = os.getenv("GEMINI_API_KEY", "")
    model_id  = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")

    if not GEMINI_AVAILABLE:
        return _fallback_explanation(
            monitoring_result, quality_result, defect_result, optimization_result,
            reason="google-genai package is not installed. Run: pip install -U google-genai",
        )

    if not api_key:
        return _fallback_explanation(
            monitoring_result, quality_result, defect_result, optimization_result,
            reason="Gemini is not configured. Set GEMINI_API_KEY in .env to enable AI-generated explanations.",
        )

    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(
            process_data, monitoring_result, quality_result,
            defect_result, optimization_result, rag_context,
        )
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text.strip()

    except Exception:
        # Gemini unavailable or call failed — return clean fallback, no error detail exposed.
        return _fallback_explanation(
            monitoring_result, quality_result, defect_result, optimization_result,
        )
