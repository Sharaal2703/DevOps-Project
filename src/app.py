from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict


app = FastAPI(
    title="Breast Cancer Classification API",
    description="ML API using an MLflow registered model",
    version="1.0.0"
)


class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 breast cancer features"
    )


@app.get("/")
def home():

    return {
        "message": "Breast Cancer Classification API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):

    prediction, probability = predict(request.features)

    return {
        "prediction": prediction,
        "probability": probability
    }