from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime

class TripRequest(BaseModel):
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lon: float = Field(..., ge=-180, le=180)
    dropoff_lat: float = Field(..., ge=-90, le=90)
    dropoff_lon: float = Field(..., ge=-180, le=180)
    pickup_datetime: str

    @field_validator("pickup_datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            # Accept ISO-8601 (with or without timezone)
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                "pickup_datetime must be ISO-8601 format "
                "(e.g. 2016-01-15T18:30:00 or 2016-01-15T18:30:00Z)"
            )
        return v

class TripBatchRequest(BaseModel):
    trips: List[TripRequest]

class PredictionResponse(BaseModel):
    predicted_duration_seconds: float
    model_version: str
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    inference_time_ms: float
