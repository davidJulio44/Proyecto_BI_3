# src/api/routes/health.py
from fastapi import APIRouter
from api.schemas.models import Envelope

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", response_model=Envelope)
def health_check():
    return Envelope(status="ok", data={"message": "API running successfully"})
