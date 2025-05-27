from pydantic import BaseModel, Field
from typing import List


class NewsItem(BaseModel):
    content: str = Field(..., description="News content")
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$', description="Date in YYYY-MM-DD format")


class PredictionRequest(BaseModel):
    prediction_horizon: int = Field(..., gt=0, description="Number of days to predict")
    news_content: List[NewsItem] = Field(..., description="List of news items")


class PredictionItem(BaseModel):
    date: str = Field(..., description="Prediction date in YYYY-MM-DD format")
    value: float = Field(..., description="Predicted value")


class PredictionResponse(BaseModel):
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    predictions: List[PredictionItem] = Field(..., description="List of predictions")