import time
import pandas as pd
from typing import Tuple, List

import xgboost as xgb

from app.config import MODEL_PATH

_model: xgb.XGBRegressor | None = None


def load_model() -> None:
    global _model

    if _model is None:
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        _model = model



def predict(features_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Run inference on a single feature vector.

    Args:
        features_df (pd.DataFrame): 2D dataframe with correct feature order

    Returns:
        prediction (float): predicted trip duration (seconds)
        inference_time_ms (float): inference latency in milliseconds
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call load_model() first.")

    start_time = time.perf_counter()

    prediction = _model.predict(features_df)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return float(prediction[0]), inference_time_ms


def predict_batch(features_df: pd.DataFrame) -> Tuple[List[float], float]:
    """
    Run inference on a batch of feature vectors.

    Args:
        features_df (pd.DataFrame): 2D dataframe with correct feature order

    Returns:
        predictions (list): list of predicted durations
        inference_time_ms (float): batch inference latency in milliseconds
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call load_model() first.")

    start_time = time.perf_counter()

    predictions = _model.predict(features_df)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return predictions.tolist(), inference_time_ms
