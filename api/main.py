from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.predict import router as predict_router
from api.routes.recommend import router as recommend_router
from api.routes.health import router as health_router
from api.routes.history import router as history_router
from api.services.prediction import preload_models
from api.middleware import logging_middleware

app = FastAPI(title="Crypto Forecasting API", version="1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.middleware("http")(logging_middleware)

@app.on_event("startup")
def _startup():
	preload_models()

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(recommend_router)
app.include_router(history_router)
