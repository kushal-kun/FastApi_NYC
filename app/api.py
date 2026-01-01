import time
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import (
    TripRequest,
    TripBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
)
from app.config import (
    MODEL_NAME,
    MODEL_VERSION,
    FEATURE_COLUMNS,
    MODEL_TASK,
    PREDICTION_TARGET,
)
from app.preprocessing import build_features_from_input
from app.model import predict, predict_batch
from app.config import MODEL_VERSION


router = APIRouter()




@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }



@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_trip_duration(request: TripRequest):
    """
    Predict taxi trip duration for a single trip.
    """
    try:
        # Convert Pydantic model to dict
        input_data = request.model_dump()

        # Feature engineering
        features_df = build_features_from_input(input_data)

        # Model inference
        prediction, inference_time_ms = predict(features_df)

        return PredictionResponse(
            predicted_duration_seconds=prediction,
            model_version=MODEL_VERSION,
            inference_time_ms=inference_time_ms,
        )

    except ValueError as e:
        # For feature validation or preprocessing errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Catch-all for unexpected failures
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction",
        )



@router.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
)
def predict_trip_duration_batch(request: TripBatchRequest):
    """
    Predict taxi trip duration for a batch of trips.
    """
    try:
        # Convert list of TripRequest -> list of dicts
        records = [trip.model_dump() for trip in request.trips]

        features_list = [
            build_features_from_input(record) for record in records
        ]

        features_df = pd.concat(features_list, ignore_index=True)

        predictions, inference_time_ms = predict_batch(features_df)

        return BatchPredictionResponse(
            predictions=predictions,
            model_version=MODEL_VERSION,
            inference_time_ms=inference_time_ms,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction",
        )

@router.get("/model_info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "task": MODEL_TASK,
        "prediction_target": PREDICTION_TARGET,
        "num_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }
