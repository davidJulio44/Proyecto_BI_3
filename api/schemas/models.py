from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import datetime as dt

class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Crypto symbol, e.g. BTC, ETH, BNB", min_length=2, max_length=10)
    days: int = Field(7, ge=1, le=60, description="Forecast horizon in days (1-60)")

class ForecastPoint(BaseModel):
    date: dt.date
    value: float

class ForecastResponse(BaseModel):
    symbol: str
    model: str
    metrics: Dict[str, float]
    forecast: List[ForecastPoint]

class RecommendationResponse(BaseModel):
    symbol: str
    horizon_days: int
    trend: str
    recommendation: str

class HistoryItem(BaseModel):
    timestamp: dt.datetime
    request: Dict[str, Any]
    response: Dict[str, Any]

class Envelope(BaseModel):
    status: str = "ok"
    data: Any
    error: Optional[str] = None
