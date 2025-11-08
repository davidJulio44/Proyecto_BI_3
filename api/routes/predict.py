from fastapi import APIRouter, HTTPException
from api.schemas.models import PredictRequest, Envelope, ForecastResponse, ForecastPoint
from api.services.prediction import predict as predict_service
from api.services.history import append_history

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=Envelope)
def predict_price(body: PredictRequest):
    try:
        result = predict_service(body.symbol, body.days)
        response = ForecastResponse(
            symbol=result["symbol"],
            model=result["model"],
            metrics=result["metrics"],
            forecast=[ForecastPoint(date=fp["date"], value=fp["value"]) for fp in result["forecast"]]
        )
        env = Envelope(status="ok", data=response)
        try:
            append_history({
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                "request": body.dict(),
                "response": {"symbol": response.symbol, "model": response.model, "metrics": response.metrics}
            })
        except Exception:
            pass
        return env
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")