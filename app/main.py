import os
from contextlib import asynccontextmanager
from datetime import datetime
import pandas as pd
import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.path.join("models", "energy_rf_pipeline.joblib")
API_KEY_ENV = "API_KEY"  # we'll set this as an environment variable


# -----------------------------
# Request / Response Schemas
# -----------------------------
class EnergyRequest(BaseModel):
    # Option B: date string, API computes time features
    date: str = Field(..., description="ISO datetime like 2016-01-11 17:00:00")

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
# Lifespan Handler (Modern FastAPI)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run training first.")

    app.state.model = joblib.load(MODEL_PATH)
    print(" Model loaded successfully.")

    yield

    # Shutdown (optional)
    print(" Shutting down API...")


app = FastAPI(
    title="Energy Consumption Predictor",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------
# Security + Feature Engineering
# -----------------------------
def require_api_key(x_api_key: str | None):
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        # Safer default: if API_KEY isn't set, refuse to run "secure" endpoints
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set.")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def add_time_features_from_date(date_str: str) -> dict:
    # Accepts "2016-01-11 17:00:00" or ISO "2016-01-11T17:00:00"
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid date format. Use ISO datetime string.")

    return {
        "hour": int(dt.hour),
        "day": int(dt.day),
        "day_of_week": dt.day_name(),
        "month": dt.month_name(),
        "weekend": int(dt.dayofweek in [5, 6]),
    }

# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=EnergyResponse)
def predict(payload: EnergyRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)

    # Compute engineered time features from date
    time_feats = add_time_features_from_date(payload.date)

    # Build a single-row dataframe matching training features
    row = payload.model_dump()
    row.pop("date")
    row.update(time_feats)

    X = pd.DataFrame([row])

    pred = app.state.model.predict(X)[0]
    return EnergyResponse(prediction=float(pred))