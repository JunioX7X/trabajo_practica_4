from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, model_validator
import numpy as np
import datetime

class FeatureImportance(BaseModel):
    """Schema for feature importance metadata."""
    feature_name: str
    importance_score: float
    cumulative_importance: Optional[float] = None

class MembershipPredictorFeatures(BaseModel):
    """Input schema for membership prediction features."""

    gender: str = Field(..., description="Customer gender, e.g., 'male', 'female', or 'other'")
    shared_account: bool = Field(..., description="Indicates if the account is shared")
    membership_tier: str = Field(..., description="Membership level: 'silver', 'gold', or 'platinum'")
    membership_fee: float = Field(..., ge=0, description="Monthly membership fee in USD")
    push_notification_enabled: bool = Field(..., description="True if push notifications are enabled")
    have_app: bool = Field(..., description="True if the customer has the store app installed")
    app_engagement_score: float = Field(..., ge=0, le=100, description="Engagement score with the app (0-100)")
    bought_store_brand: bool = Field(..., description="True if customer buys the store’s brand")
    promotion_participation_count: int = Field(..., ge=0, description="Number of promotions the customer participated in")
    average_basket_size: float = Field(..., ge=0, description="Average number of items per shopping trip")
    use_count: int = Field(..., ge=0, description="Total number of times the customer used the membership")
    reward_points_used: float = Field(..., ge=0, description="Total reward points redeemed")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "male",
                "shared_account": True,
                "membership_tier": "silver",
                "membership_fee": 29.99,
                "push_notification_enabled": True,
                "have_app": True,
                "app_engagement_score": 65.5,
                "bought_store_brand": False,
                "promotion_participation_count": 3,
                "average_basket_size": 55.2,
                "use_count": 18,
                "reward_points_used": 200.0
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

    @classmethod
    @model_validator(mode="after")
    def validate_probability_sum(cls, values):
        p_yes = values.probability_yes
        p_no = values.probability_no
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