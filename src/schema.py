# This defines what the API will accept later (Option B).
# We'll reuse this in FastAPI.

from pydantic import BaseModel, Field

class EnergyRequest(BaseModel):
    date: str = Field(..., description="ISO date-time string, e.g. 2016-01-11 17:00:00")

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
