from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum
import numpy as np
import datetime


class FeatureImportance(BaseModel):
    """Schema for feature importance metadata."""
    feature_name: str
    importance_score: float
    cumulative_importance: Optional[float] = None


class ModelMetadata(BaseModel):
    """Schema for model versioning and lineage tracking."""
    model_id: str = Field(..., description="Unique identifier for the model version")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    training_dataset: str = Field(..., description="Path or identifier of training dataset")
    model_type: str
    framework_version: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    feature_importance: Optional[List[FeatureImportance]] = None
    hyperparameters: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_20240418_a1b2c3",
                "created_at": "2024-04-18T10:30:00Z",
                "training_dataset": "membership_groceries_userprofile.csv",
                "model_type": "RandomForestClassifier",
                "framework_version": "scikit-learn 1.3.0",
                "accuracy": 0.92,
                "f1_score": 0.89,
                "feature_importance": [
                    {"feature_name": "purchase_frequency", "importance_score": 0.23, "cumulative_importance": 0.23},
                    {"feature_name": "total_spend", "importance_score": 0.18, "cumulative_importance": 0.41}
                ],
                "hyperparameters": {"n_estimators": 100, "max_depth": 10}
            }
        }


class MembershipPredictorFeatures(BaseModel):
    """Input schema for membership prediction features."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    income: float = Field(..., ge=0, description="Annual income in USD")
    shopping_frequency: int = Field(..., ge=0, description="Number of shopping trips per month")
    avg_basket_value: float = Field(..., ge=0, description="Average transaction value in USD")
    months_active: int = Field(..., ge=1, description="Number of months as active customer")
    previous_renewals: int = Field(..., ge=0, description="Count of previous membership renewals")
    product_categories_purchased: int = Field(..., ge=0, description="Number of unique product categories")
    has_returned_items: bool = Field(..., description="Whether customer has returned items")
    distance_to_store: float = Field(..., ge=0, description="Distance to nearest store in km")

    @validator('avg_basket_value')
    def validate_basket_value(cls, v):
        if v > 10000:
            raise ValueError("Unusually high basket value detected")
        return v

    @model_validator(mode="after")
    def validate_activity_metrics(cls, values):
        if values.get('months_active', 0) > 0 and values.get('shopping_frequency', 0) == 0:
            raise ValueError("Invalid activity pattern: active months with zero shopping frequency")
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 65000.0,
                "shopping_frequency": 8,
                "avg_basket_value": 120.5,
                "months_active": 24,
                "previous_renewals": 2,
                "product_categories_purchased": 12,
                "has_returned_items": False,
                "distance_to_store": 5.3
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for membership renewal prediction."""
    auto_renew_prediction: int = Field(..., description="Binary prediction: 1=renewal, 0=no renewal")
    probability_yes: float = Field(..., ge=0.0, le=1.0, description="Probability of renewal")
    probability_no: float = Field(..., ge=0.0, le=1.0, description="Probability of non-renewal")
    model_version: str = Field(..., description="Version ID of model used for prediction")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    prediction_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @validator('probability_yes', 'probability_no')
    def validate_probabilities(cls, v):
        return round(float(v), 4)  # Ensure consistent decimal precision

    @model_validator(mode="after")
    def validate_probability_sum(cls, values):
        p_yes = values.get('probability_yes', 0)
        p_no = values.get('probability_no', 0)
        if not np.isclose(p_yes + p_no, 1.0, atol=1e-5):
            raise ValueError("Probabilities must sum to 1.0")
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "auto_renew_prediction": 1,
                "probability_yes": 0.8763,
                "probability_no": 0.1237,
                "model_version": "model_20240418_a1b2c3",
                "prediction_id": "pred_f7g8h9",
                "prediction_timestamp": "2024-04-18T14:25:36Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    features: List[MembershipPredictorFeatures]
    request_id: Optional[str] = None

    @validator('features')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size exceeds maximum limit of 1000 records")
        return v


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    predictions: List[PredictionResponse]
    batch_id: str
    processing_time_ms: float
    model_version: str
    records_processed: int

    @validator('records_processed')
    def validate_record_count(cls, v, values):
        if 'predictions' in values and len(values['predictions']) != v:
            raise ValueError("Records processed count mismatch")
        return v


class ModelDeploymentConfig(BaseModel):
    """Schema for model deployment configuration."""
    model_path: str
    version: str
    environment: str = Field(..., pattern='^(development|staging|production)$')
    replicas: int = Field(1, ge=1, le=10)
    resources: Dict[str, Dict[str, str]]
    autoscaling_enabled: bool = False
    monitoring_enabled: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "/app/models/grocery_membership_model.joblib",
                "version": "v1.2.3",
                "environment": "production",
                "replicas": 3,
                "resources": {
                    "requests": {"cpu": "500m", "memory": "512Mi"},
                    "limits": {"cpu": "1", "memory": "1Gi"}
                },
                "autoscaling_enabled": True,
                "monitoring_enabled": True
            }
        }


class PredictionMonitoringEvent(BaseModel):
    """Schema for prediction monitoring events."""
    prediction: PredictionResponse
    input_features: MembershipPredictorFeatures
    latency_ms: float
    environment: str
    client_id: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "auto_renew_prediction": 1,
                    "probability_yes": 0.8763,
                    "probability_no": 0.1237,
                    "model_version": "model_20240418_a1b2c3",
                    "prediction_id": "pred_f7g8h9",
                    "prediction_timestamp": "2024-04-18T14:25:36Z"
                },
                "input_features": {
                    "age": 35,
                    "income": 65000.0,
                    "shopping_frequency": 8,
                    "avg_basket_value": 120.5,
                    "months_active": 24,
                    "previous_renewals": 2,
                    "product_categories_purchased": 12,
                    "has_returned_items": False,
                    "distance_to_store": 5.3
                },
                "latency_ms": 12.43,
                "environment": "production",
                "client_id": "web-portal",
                "timestamp": "2024-04-18T14:25:36Z"
            }
        }