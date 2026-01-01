from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.json"

MODEL_NAME = "nyc_taxi_xgb_regressor"
MODEL_VERSION = "v1.0"
MODEL_TASK = "regression"
PREDICTION_TARGET = "trip_duration_seconds"

FEATURE_COLUMNS: List[str] = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_minute_sin",
    "pickup_minute_cos",
    "wd_0",
    "wd_1",
    "wd_2",
    "wd_3",
    "wd_4",
    "wd_5",
    "wd_6",
    "haversine_km",
    "bearing_sin",
    "bearing_cos",
]
