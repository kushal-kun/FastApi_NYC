import numpy as np
import pandas as pd
from datetime import datetime

from app.config import FEATURE_COLUMNS



def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Compute haversine distance in kilometers.
    """
    R = 6371.0  # Earth radius (km)

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """
    Compute bearing angle in degrees.
    """
    lat1, lat2 = map(np.radians, [lat1, lat2])
    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(y, x)) + 360) % 360



def build_features_from_input(data: dict) -> pd.DataFrame:
    """
    Build inference-time features for NYC taxi trip duration prediction.

    Expected input:
    {
        "pickup_lat": float,
        "pickup_lon": float,
        "dropoff_lat": float,
        "dropoff_lon": float,
        "pickup_datetime": str (ISO-8601)
    }

    Returns:
        pd.DataFrame with columns ordered exactly as FEATURE_COLUMNS
    """


    pickup_dt = pd.to_datetime(
        data["pickup_datetime"],
        utc=True,
        errors="raise"
    ).tz_convert(None)

    pickup_hour = pickup_dt.hour
    pickup_minute = pickup_dt.minute
    weekday = pickup_dt.weekday()  # Monday = 0


    pickup_hour_sin = np.sin(2 * np.pi * pickup_hour / 24)
    pickup_hour_cos = np.cos(2 * np.pi * pickup_hour / 24)

    pickup_minute_sin = np.sin(2 * np.pi * pickup_minute / 60)
    pickup_minute_cos = np.cos(2 * np.pi * pickup_minute / 60)


    pickup_lat = data["pickup_lat"]
    pickup_lon = data["pickup_lon"]
    dropoff_lat = data["dropoff_lat"]
    dropoff_lon = data["dropoff_lon"]

    distance_km = haversine_km(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
    )

    bearing = bearing_deg(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
    )

    bearing_sin = np.sin(np.radians(bearing))
    bearing_cos = np.cos(np.radians(bearing))


    weekday_features = {
        f"wd_{i}": 1 if i == weekday else 0
        for i in range(7)
    }


    features = {
        # Raw coordinates
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,

        # Time features
        "pickup_hour_sin": pickup_hour_sin,
        "pickup_hour_cos": pickup_hour_cos,
        "pickup_minute_sin": pickup_minute_sin,
        "pickup_minute_cos": pickup_minute_cos,

        # Weekday one-hot
        **weekday_features,

        # Spatial engineered
        "haversine_km": distance_km,
        "bearing_sin": bearing_sin,
        "bearing_cos": bearing_cos,
    }


    df = pd.DataFrame([features])

    df = df.reindex(columns=FEATURE_COLUMNS)

    if df.isnull().any().any():
        raise ValueError("NaN detected in feature vector")

    return df
