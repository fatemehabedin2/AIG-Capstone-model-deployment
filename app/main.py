import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
MODEL_LOCAL_PATH = os.path.join("models", "energy_rf_pipeline.joblib")

# App-level API key (your FastAPI auth)
API_KEY_ENV = "API_KEY"

# GCS settings (for downloading the model artifact)
GCS_BUCKET_ENV = "GCS_BUCKET_NAME"          # e.g., energy-model-folio-451800-p5-4821
GCS_BLOB_ENV = "MODEL_BLOB_NAME"            # e.g., energy_rf_pipeline.joblib


# -----------------------------
# Request / Response Schemas
# -----------------------------
class EnergyRequest(BaseModel):
    # Option B: date string, API computes time features
    date: str = Field(..., description="ISO datetime like 2016-01-11 17:00:00 or 2016-01-11T17:00:00")

    lights: float
    T1: float
    RH_1: float
    T2: float
    RH_2: float
    T3: float
    RH_3: float
    T4: float
    RH_4: float
    T5: float
    RH_5: float
    RH_6: float
    T7: float
    RH_7: float
    T8: float
    RH_8: float
    T9: float
    RH_9: float
    T_out: float
    Press_mm_hg: float
    RH_out: float
    Windspeed: float
    Visibility: float
    Tdewpoint: float


class EnergyResponse(BaseModel):
    prediction: float


# -----------------------------
# Helpers
# -----------------------------
def require_api_key(x_api_key: str | None):
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set.")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def add_time_features_from_date(date_str: str) -> dict:
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use ISO datetime like '2016-01-11 17:00:00'."
        )

    return {
        "hour": int(dt.hour),
        "day": int(dt.day),
        "day_of_week": dt.day_name(),
        "month": dt.month_name(),
        "weekend": int(dt.dayofweek in [5, 6]),
    }


def ensure_model_present():
    """
    Ensures the model artifact exists locally. If not, downloads from GCS.
    Requires env vars:
      - GCS_BUCKET_NAME
      - MODEL_BLOB_NAME
    """
    if os.path.exists(MODEL_LOCAL_PATH):
        return

    bucket_name = os.getenv(GCS_BUCKET_ENV)
    blob_name = os.getenv(GCS_BLOB_ENV, "energy_rf_pipeline.joblib")

    if not bucket_name:
        raise RuntimeError("Model not found locally and GCS_BUCKET_NAME is not set.")

    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)

    print(f"⬇️  Downloading model from GCS: gs://{bucket_name}/{blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(MODEL_LOCAL_PATH)
    print(f"✅ Model downloaded to {MODEL_LOCAL_PATH}")


# -----------------------------
# Lifespan (startup)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model_present()

    app.state.model = joblib.load(MODEL_LOCAL_PATH)
    print("✅ Model loaded successfully.")

    yield


app = FastAPI(
    title="Energy Consumption Predictor",
    version="1.1.0",
    lifespan=lifespan
)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=EnergyResponse)
def predict(payload: EnergyRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)

    time_feats = add_time_features_from_date(payload.date)

    row = payload.model_dump()
    row.pop("date")
    row.update(time_feats)

    X = pd.DataFrame([row])

    pred = app.state.model.predict(X)[0]
    return EnergyResponse(prediction=float(pred))
