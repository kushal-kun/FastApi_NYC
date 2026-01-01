# app/main.py

from fastapi import FastAPI

from app.api import router
from app.model import load_model
from app.config import MODEL_NAME, MODEL_VERSION


def create_app() -> FastAPI:
    app = FastAPI(
        title="NYC Taxi Trip Duration Inference Service",
        description=(
            "Stateless ML inference API for predicting NYC taxi trip duration "
            "based on pickup/dropoff coordinates and pickup time."
        ),
        version=MODEL_VERSION,
    )

    # Register routes
    app.include_router(router)

    @app.on_event("startup")
    def startup_event():
        load_model()

    return app


app = create_app()
