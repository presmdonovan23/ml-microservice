import joblib
import xgboost as xgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional


model: Optional[xgb.Booster] = None
scaler: Optional[object] = None

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int

class HealthResponse(BaseModel):
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    print("Loading model and scaler...")
    model = xgb.Booster()
    model.load_model("artifacts/model.json")
    scaler = joblib.load("artifacts/scaler.pkl")
    print("Model and scaler loaded successfully.")

    yield  # Application runs here

    print("Shutting down application...")

#FastAPI app
app = FastAPI(
    title="Handwritten Digit Classifier API",
    description="API for classifying handwritten digits using a trained XGBoost model",
    version="1.0.0",
    lifespan=lifespan
)

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Preprocess input features
    features_array = np.array(request.features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(features_scaled)

    # Make prediction
    pred_class = int(model.predict(dmatrix)[0])

    return PredictResponse(prediction=pred_class)