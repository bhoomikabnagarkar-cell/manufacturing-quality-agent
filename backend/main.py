"""
Manufacturing Process Quality Control Agent - FastAPI Backend
Main entry point. Runs the full 4-agent pipeline on demand.
"""
import sys
import os

# ── Path setup ───────────────────────────────────────────────────────────────
# BASE_DIR  = .../manufacturing-quality-agent/backend
# BACKEND_PARENT = .../manufacturing-quality-agent   (project root)
# ML_DIR    = .../manufacturing-quality-agent/ml
#
# When launched as  `python -m uvicorn backend.main:app`  from the project
# root, Python's package machinery resolves `backend` correctly but does NOT
# automatically put `backend/` on sys.path, so bare `from agents import …`
# and `from services import …` fail.
#
# Fix: insert the directory that *contains* the `agents/` and `services/`
# folders (i.e. BASE_DIR == backend/) so those sub-packages are importable.
# Also insert ml/ so the bare `import anomaly_detection / defect_prediction`
# statements resolve to the ML module files.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))          # .../backend
BACKEND_PARENT  = os.path.normpath(os.path.join(BASE_DIR, ".."))      # project root
ML_DIR          = os.path.join(BACKEND_PARENT, "ml")

for d in (BASE_DIR, ML_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

# ── Standard imports ─────────────────────────────────────────────────────────
import csv
import json
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Load .env from the project root (parent of this file's directory).
# Using an explicit path means the server works regardless of which
# directory uvicorn is started from.
_ENV_FILE = os.path.join(BACKEND_PARENT, ".env")
load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ── Agent / service imports ──────────────────────────────────────────────────
from backend.agents import (
    process_monitoring_agent,
    quality_analysis_agent,
    defect_prediction_agent,
    process_optimization_agent,
)
from services import rag_service, granite_service
import anomaly_detection as ad
import defect_prediction as dp

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Manufacturing Quality Control Agent",
    description="AI-powered manufacturing process quality control using multi-agent architecture.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
STATIC_DIR   = os.path.join(FRONTEND_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "manufacturing_data.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════

class ProcessReading(BaseModel):
    temperature:     float
    pressure:        float
    vibration:       float
    machine_speed:   float
    production_rate: float
    timestamp:       Optional[str] = None


class TrainRequest(BaseModel):
    csv_path: Optional[str] = None   # if None, uses default CSV


# ═══════════════════════════════════════════════════════════════════════════
# Helper: run full agent pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(reading: dict) -> dict:
    """Execute all four agents + RAG + Granite on a single process reading."""

    # 1. Process Monitoring Agent
    monitoring_result = process_monitoring_agent.run(reading)

    # 2. Quality Analysis Agent
    quality_result = quality_analysis_agent.run(reading, monitoring_result)

    # 3. Defect Prediction Agent
    defect_result = defect_prediction_agent.run(reading, monitoring_result)

    # 4. Process Optimization Agent
    optimization_result = process_optimization_agent.run(
        reading, monitoring_result, quality_result, defect_result
    )

    # 5. RAG retrieval
    rag_context = rag_service.retrieve_for_report(
        monitoring_result, quality_result, defect_result
    )

    # 6. IBM Granite explanation
    explanation = granite_service.generate_explanation(
        reading, monitoring_result, quality_result,
        defect_result, optimization_result, rag_context,
    )

    return {
        "process_data":        reading,
        "monitoring_result":   monitoring_result,
        "quality_result":      quality_result,
        "defect_result":       defect_result,
        "optimization_result": optimization_result,
        "ai_explanation":      explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>Frontend not found. Run from project root.</h2>", status_code=404)
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/train")
def train_models(req: TrainRequest = TrainRequest()):
    """Train (or retrain) the ML models on the manufacturing CSV."""
    csv_path = req.csv_path or DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise HTTPException(404, f"CSV not found: {csv_path}")

    anomaly_result = ad.train(csv_path)
    defect_result  = dp.train(csv_path)

    return {
        "status": "ok",
        "anomaly_detection": anomaly_result,
        "defect_prediction": defect_result,
    }


@app.post("/api/analyze")
def analyze_reading(reading: ProcessReading):
    """
    Run the full 4-agent pipeline on a single process reading.
    Returns the complete quality report.
    """
    reading_dict = reading.dict(exclude_none=True)
    try:
        result = run_pipeline(reading_dict)
    except FileNotFoundError as e:
        raise HTTPException(
            400,
            detail=str(e) + " — call POST /api/train first.",
        )
    return result


@app.post("/api/upload-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and run the pipeline on its LAST row.
    The last row represents the most recent process reading.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    contents = await file.read()
    lines = contents.decode("utf-8").splitlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    if not rows:
        raise HTTPException(400, "CSV file is empty.")

    last_row = rows[-1]
    reading = {
        "temperature":     float(last_row.get("temperature", 85)),
        "pressure":        float(last_row.get("pressure", 5.0)),
        "vibration":       float(last_row.get("vibration", 1.0)),
        "machine_speed":   float(last_row.get("machine_speed", 1000)),
        "production_rate": float(last_row.get("production_rate", 50)),
        "timestamp":       last_row.get("timestamp", ""),
    }

    try:
        result = run_pipeline(reading)
    except FileNotFoundError as e:
        raise HTTPException(400, detail=str(e) + " — call POST /api/train first.")
    return result


@app.get("/api/dataset")
def get_dataset():
    """Return the full manufacturing dataset as JSON (for dashboard charts)."""
    if not os.path.exists(DEFAULT_CSV):
        raise HTTPException(404, "Default dataset not found.")
    df = pd.read_csv(DEFAULT_CSV)
    return {"data": df.to_dict(orient="records"), "total": len(df)}


@app.get("/api/dataset/summary")
def get_dataset_summary():
    """Return basic statistics of the manufacturing dataset."""
    if not os.path.exists(DEFAULT_CSV):
        raise HTTPException(404, "Default dataset not found.")
    df = pd.read_csv(DEFAULT_CSV)
    features = ["temperature", "pressure", "vibration", "machine_speed", "production_rate"]
    summary = {}
    for col in features:
        summary[col] = {
            "mean":  round(float(df[col].mean()), 3),
            "std":   round(float(df[col].std()), 3),
            "min":   round(float(df[col].min()), 3),
            "max":   round(float(df[col].max()), 3),
        }
    status_counts = df["quality_status"].value_counts().to_dict()
    defect_rate   = round(float(df["defect"].mean()) * 100, 2)

    return {
        "total_records": len(df),
        "statistics": summary,
        "quality_status_counts": status_counts,
        "defect_rate_percent": defect_rate,
    }


@app.get("/api/analyze-latest")
def analyze_latest():
    """Run the pipeline on the latest row of the default dataset."""
    if not os.path.exists(DEFAULT_CSV):
        raise HTTPException(404, "Default dataset not found.")
    df = pd.read_csv(DEFAULT_CSV)
    last_row = df.iloc[-1].to_dict()
    reading = {
        "temperature":     float(last_row["temperature"]),
        "pressure":        float(last_row["pressure"]),
        "vibration":       float(last_row["vibration"]),
        "machine_speed":   float(last_row["machine_speed"]),
        "production_rate": float(last_row["production_rate"]),
        "timestamp":       str(last_row.get("timestamp", "")),
    }
    try:
        return run_pipeline(reading)
    except FileNotFoundError as e:
        raise HTTPException(400, detail=str(e) + " — call POST /api/train first.")


@app.get("/api/analyze-row/{row_index}")
def analyze_row(row_index: int):
    """Run the pipeline on a specific row index from the default dataset."""
    if not os.path.exists(DEFAULT_CSV):
        raise HTTPException(404, "Default dataset not found.")
    df = pd.read_csv(DEFAULT_CSV)
    if row_index < 0 or row_index >= len(df):
        raise HTTPException(400, f"row_index must be 0–{len(df)-1}")
    row = df.iloc[row_index].to_dict()
    reading = {
        "temperature":     float(row["temperature"]),
        "pressure":        float(row["pressure"]),
        "vibration":       float(row["vibration"]),
        "machine_speed":   float(row["machine_speed"]),
        "production_rate": float(row["production_rate"]),
        "timestamp":       str(row.get("timestamp", "")),
    }
    try:
        return run_pipeline(reading)
    except FileNotFoundError as e:
        raise HTTPException(400, detail=str(e) + " — call POST /api/train first.")


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Manufacturing Quality Control Agent"}
