# app/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import joblib
import os
import pandas as pd
from app.models.schemas import MembershipPredictorFeatures, PredictionResponse
from app.models.train import train_model  # Ajusta la ruta según tu estructura
model = train_model()

app = FastAPI(title="API de Predicción de Membresías de Supermercado")

model_path = os.getenv("MODEL_PATH")
model = joblib.load(model_path)

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="API Key Inválida")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MembershipPredictorFeatures, api_key: str = Depends(get_api_key)):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()

    return {
        "prediction": str(prediction),
        "probability": float(probability)
    }
