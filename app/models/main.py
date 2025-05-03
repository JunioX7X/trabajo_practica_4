# app/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import joblib
import os
import json
import pandas as pd
from app.models.schemas import MembershipPredictorFeatures, PredictionResponse
from uuid import uuid4
import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_DIR, "data", "membership_groceries_userprofile.csv")
MODEL_OUTPUT_PATH = os.path.join(ROOT_DIR, "app", "models", "grocery_membership_model.joblib")

app = FastAPI(title="API de Predicción de Membresías de Supermercado")

model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "grocery_membership_model.joblib"))
model = joblib.load(model_path)


# Cargar columnas esperadas
columns_path = model_path.replace(".joblib", "_columns.json")
with open(columns_path, "r") as f:
    expected_columns = json.load(f)


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

    # Aplicar dummies igual que en entrenamiento
    input_df = pd.get_dummies(input_df)

    # Asegurar que tenga las mismas columnas que el modelo espera
    print(input_df.head())

    # Ajustar columnas del input según modelo entrenado
    missing_cols = [col for col in expected_columns if col not in input_df.columns]
    if missing_cols:
        missing_df = pd.DataFrame([[0] * len(missing_cols)], columns=missing_cols)
        input_df = pd.concat([input_df, missing_df], axis=1)
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    return {
        "auto_renew_prediction": bool(prediction),
        "probability_yes": float(proba[1]),
        "probability_no": float(proba[0]),
        "model_version": "1.0.0",
        "prediction_id": str(uuid4()),
        "prediction_timestamp": datetime.datetime.now()
    }
