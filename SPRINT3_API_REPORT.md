# Sprint 3 - API Técnica (FastAPI)

## Estructura
- `api/main.py`: App FastAPI, CORS, startup preload, routers
- `api/routes/health.py`: GET /health
- `api/routes/predict.py`: POST /predict
- `api/routes/recommend.py`: GET /recommendations
- `api/routes/history.py`: GET /history
- `api/schemas/models.py`: Pydantic (requests/responses/envelope)
- `api/services/prediction.py`: Precarga y predicción ARIMA
- `api/services/recommendation.py`: Reglas simples de recomendación
- `api/services/history.py`: Persistencia JSON de historial
- `api/middleware.py`: Logging y manejo genérico de errores

## Endpoints
- GET /health → {"status":"ok","data":{"message":"..."}}
- POST /predict → body {symbol, days} → Forecast ARIMA + métricas
- GET /recommendations → query {symbol, days} → tendencia y recomendación
- GET /history → últimos N registros

## Integración de modelos
- Se usa `timeseries/models_arima.py` para entrenar/predecir.
- Modelos se precargan al startup para BTC/ETH/BNB.

## Formato de respuesta
- Envoltura estándar: {status, data, error}

## Logs
- Log de peticiones/respuestas vía middleware (loguru).

## Pruebas
- `tests/test_api.py` valida health y casos de predict.

## Cómo ejecutar
- venv: `uvicorn api.main:app --reload --port 8000`
- Poetry: `poetry run uvicorn api.main:app --reload --port 8000`

## Próximas mejoras
- Persistencia en SQLite y /history con filtros
- Recomendación basada en clustering y señales técnicas
- Versionado de API `/api/v1`
- Seguridad (CORS restringido, auth opcional)
- Pruebas con dataset controlado (fixtures)
