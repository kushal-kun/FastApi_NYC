# NYC Taxi Trip Duration — ML Inference Service

A stateless, production-style machine learning inference service for predicting NYC taxi trip duration given pickup/dropoff coordinates and pickup time.

This project emphasizes applied ML engineering concerns—feature engineering, training–serving parity, inference latency, input validation, and system design—rather than model novelty or leaderboard optimization.

---

## Problem Overview

Accurately estimating taxi trip duration is a common real-world regression problem with applications in:

- ETA estimation and pricing
- Route simulation and mobility analytics
- Urban transportation planning

This project demonstrates how a trained ML model can be reliably packaged and served as a low-latency inference microservice for downstream systems.

---

## Model and Training Summary

- **Task:** Regression  
- **Target:** Trip duration (seconds)  
- **Model:** XGBoost Regressor  
- **Dataset:** NYC Taxi Trips (Kaggle)  
- **Training environment:** Kaggle (dataset scale >10M rows)  
- **Model artifact size:** ~8 MB  

The model was designed to be lightweight, prioritizing fast CPU inference, stability, and operational simplicity.

---

## Offline Model Performance

Evaluated on a held-out validation set:

- **RMSLE:** **0.29**

This significantly outperforms naive baselines such as constant-mean prediction and distance-only linear models.

Error analysis shows increased variance for longer trips, which is expected due to traffic dynamics and routing uncertainty in dense urban environments.

The accompanying notebook includes:
- residual distributions
- predicted vs actual duration plots
- feature importance visualizations

See `training/notebook.ipynb` for full EDA, training, and evaluation details.

---

## Feature Engineering

Feature ordering is explicitly frozen and enforced at inference time to prevent training–serving skew and silent inference errors.

### Input Fields (API Contract)

```json
{
  "pickup_lat": float,
  "pickup_lon": float,
  "dropoff_lat": float,
  "dropoff_lon": float,
  "pickup_datetime": "ISO-8601 string"
}

```
### Derived Features

| Category   | Features |
|-----------|----------|
| Spatial   | Haversine distance, bearing (sin / cos) |
| Temporal  | Pickup hour and minute (cyclical encoding) |
| Calendar  | One-hot weekday indicators |
| Raw       | Pickup and dropoff latitude / longitude |

Feature ordering is explicitly frozen and enforced at inference time to prevent silent model errors.

---

## System Architecture

| Step | Component | Description |
|------|----------|-------------|
| 1 | Client | Sends HTTP request with trip details (JSON payload) |
| 2 | FastAPI Service | Receives request and routes it to the appropriate endpoint |
| 3 | Input Validation | Pydantic validates schema and input ranges |
| 4 | Feature Engineering | Raw inputs transformed into model-ready features |
| 5 | Model Inference | XGBoost model predicts trip duration |
| 6 | Response | Prediction, model version, and latency returned as JSON |

**Design Characteristics**

- Stateless architecture for easy horizontal scaling
- Model initialized once during application startup
- Supports both single-record and batch inference
- Fail-fast input validation to prevent invalid inference requests

## Endpoints and Example Requests

### Health Check (GET/health)

Reports service and model health status.

**Response**
```json
{
  "status": "ok",
  "model_version": "v1.0"
}
```

### Info (GET/model_info)

Reports static metadata about the loaded model, including version, task type, and feature schema.

**Response**

```json
{
  "model_name": "nyc_taxi_xgb_regressor",
  "model_version": "v1.0",
  "task": "regression",
  "prediction_target": "trip_duration_seconds",
  "num_features": 18,
  "features": [
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
    "bearing_cos"
  ]
}
```

### Single Prediction (POST/predict)

Processes a single trip at a time.

**Request**
```json
{
  "pickup_lat": 40.758,
  "pickup_lon": -73.9855,
  "dropoff_lat": 40.7128,
  "dropoff_lon": -74.0060,
  "pickup_datetime": "2016-01-15T18:30:00"
}
```

**Response**
```json
{
  "predicted_duration_seconds": 845.2,
  "model_version": "v1.0",
  "inference_time_ms": 18.7
}
```

### Batch Prediction

Processes and predicts the duration for multiple trips at the same time.

**Request**
```json
{
  "trips": [
    {
      "pickup_lat": 40.758,
      "pickup_lon": -73.9855,
      "dropoff_lat": 40.7128,
      "dropoff_lon": -74.0060,
      "pickup_datetime": "2016-01-15T18:30:00"
    },
    {
      "pickup_lat": 40.73061,
      "pickup_lon": -73.935242,
      "dropoff_lat": 40.650002,
      "dropoff_lon": -73.949997,
      "pickup_datetime": "2016-01-15T08:15:00"
    }
  ]
}
```

**Response**
```json
{
  "predictions": [845.2, 1320.6],
  "model_version": "v1.0",
  "inference_time_ms": 21.4
}
```

### Error Handling

In the presence of invalid or malformed inputs, the service fails gracefully and returns structured validation errors with explicit failure reasons.

**example**

```json
{
  "detail": [
    {
      "msg": "Input should be less than or equal to 90",
      "loc": ["body", "pickup_lat"]
    },
    {
      "msg": "pickup_datetime must be ISO-8601 format",
      "loc": ["body", "pickup_datetime"]
    }
  ]
}
```

## Inference Performance

Performance measurements were collected after model warm-up to reflect steady-state inference latency.

Measured on CPU in a local development environment:

- **Single-request latency:** ~15–25 ms
- **Batch inference:** amortized overhead with near-linear scaling
- **Model load time:** negligible relative to application startup

Latency measurements include model execution, Python-to-native XGBoost calls, and minimal orchestration overhead. 


## Running the Service Locally

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn app.main:app --reload
```

Interactive API documentation available at:
```text
http://127.0.0.1:8000/docs
```

## Tech Stack

- **API Framework:** FastAPI
- **Model:** XGBoost (CPU inference)
- **Validation:** Pydantic v2
- **Feature Processing:** NumPy, Pandas
- **Serialization:** XGBoost native JSON format
- **Testing:** curl and PowerShell scripts

## Design Decisions

- The service is **stateless by design** to simplify horizontal scaling and deployment.
- The model is **loaded once at startup** to avoid per-request overhead.
- Feature ordering is **explicitly frozen** to prevent training–serving skew.
- No frontend or persistence layer is included; the service is intended for **programmatic consumption**.
- “Model retraining, monitoring, and drift detection are intentionally out of scope for this repository, but the service is designed to integrate with such systems in a production setting.”

## What This Project Demonstrates

This project demonstrates:
- How to take a trained ML model from notebook to a production-style inference service
- How to enforce training–serving parity in feature engineering
- How to design low-latency, stateless ML APIs suitable for horizontal scaling
- How to validate and fail fast on bad inputs in ML systems
