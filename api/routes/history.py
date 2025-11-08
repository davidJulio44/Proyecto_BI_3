from fastapi import APIRouter, Query
from api.schemas.models import Envelope, HistoryItem
from api.services.history import read_history

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/", response_model=Envelope)
def get_history(limit: int = Query(25, ge=1, le=200)):
    entries = read_history(limit=limit)
    items = [HistoryItem(**e) for e in entries]
    return Envelope(status="ok", data=items)