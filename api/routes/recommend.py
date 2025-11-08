from fastapi import APIRouter, HTTPException, Query
from api.schemas.models import Envelope, RecommendationResponse
from api.services.recommendation import recommend as recommend_service
from api.services.history import append_history

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("/", response_model=Envelope)
def recommend(symbol: str = Query("BTC"), days: int = Query(7, ge=1, le=60)):
    try:
        result = recommend_service(symbol, days)
        resp = RecommendationResponse(
            symbol=result["symbol"],
            horizon_days=days,
            trend=result["trend"],
            recommendation=result["recommendation"],
        )
        env = Envelope(status="ok", data=resp)
        try:
            append_history({
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                "request": {"symbol": symbol, "days": days},
                "response": resp.dict()
            })
        except Exception:
            pass
        return env
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")