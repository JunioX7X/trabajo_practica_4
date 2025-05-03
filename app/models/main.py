# app/api/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import joblib
import os
import pandas as pd
from app.models.schemas import MembershipPredictorFeatures, PredictionResponse
import uuid
from datetime import datetime

app = FastAPI(title="API de Predicción de Membresías de Supermercado")

# Ruta del modelo entrenado
model_path = os.path.join(os.path.dirname(__file__), "grocery_membership_model.joblib")
model = joblib.load(model_path)

# Seguridad con API Key
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY")  # asegúrate de definir esta variable
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="API Key Inválida")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MembershipPredictorFeatures, api_key: str = Depends(get_api_key)):
    try:
        input_df = pd.DataFrame([features.dict()])
        proba = model.predict_proba(input_df)[0]
        prediction = int(proba[1] >= 0.5)  # o model.predict(input_df)[0]

        return PredictionResponse(
            auto_renew_prediction=prediction,
            probability_yes=round(float(proba[1]), 4),
            probability_no=round(float(proba[0]), 4),
            model_version="v1.0.0",  # puedes automatizar esto más adelante
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
            prediction_timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")