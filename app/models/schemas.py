
from pydantic import BaseModel
class PredictionResponse(BaseModel):
    auto_renew_prediction: int
    probability_yes: float
    probability_no: float