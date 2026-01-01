# Copilot / AI agent instructions for Proyecto_BI_3

Purpose: give an AI agent targeted, actionable knowledge to be productive immediately in this repo.

## Big picture
- This project is a small data/ML stack for cryptocurrency analysis: ETL -> EDA -> clustering -> time-series models -> API + (planned) dashboard.
- Core folders:
  - `scripts/` : ETL, cleaning, EDA, clustering, and model training helpers used by notebooks and scripts.
  - `data/` : CSV artifacts (important: `data/crypto_clean_BTC_ETH_BNB.csv` is the canonical dataset).
  - `timeseries/` : ARIMA and RNN model implementations (`models_arima.py`, `models_rnn.py`).
  - `api/` : FastAPI app. Routes in `api/routes/`, logic in `api/services/`, Pydantic schemas in `api/schemas/models.py`.
  - `dashboard/` : Streamlit app (planned) entry at `dashboard/app.py`.

## Quick developer workflows (commands you should suggest/use)
- Install dependencies (preferred): `poetry install` then `poetry shell`.
- venv alternative: `python -m venv .venv` then `pip install -r requirements.txt`.
- Run API locally: `uvicorn api.main:app --reload --port 8000` (or `poetry run uvicorn ...`).
- Run tests: `pytest -q` (or `poetry run pytest -q`). Tests use `httpx.AsyncClient` and accept 200/400/500 for predict when dataset is missing.
- Regenerate dataset: `python scripts/extract_data.py` then `python scripts/clean_data.py`.

## Project-specific conventions & patterns (do this, not generic advice)
- API composition:
  - Add new endpoints under `api/routes/` (APIRouter instances). Put business logic in `api/services/` and models in `api/schemas/`.
  - Routes always return an `Envelope` (`api/schemas/models.py`) as top-level response; follow existing patterns (`status`, `data`, `error`).
  - Errors: services raise `ValueError` for client problems (mapped to 400 in routes); unexpected exceptions map to 500.
- Startup behavior: `api.main` registers a `startup` event that calls `preload_models()` (from `api/services/prediction.py`) to train/load ARIMA models into an in-memory `CACHE`. Be mindful that tests/CI may run without `data/` present.
- Data loader: `api.utils.loader.load_data()` returns an empty DataFrame on missing/corrupt data. Check for `df.empty` before assuming presence.
- History persistence: `api/services/history.py` appends to `data/recommend_history.json` (simple JSON list). Avoid writing to it in tests unless explicitly intended.
- Logging: `loguru` is used across the project; prefer `logger.info / logger.warning / logger.exception`.

## Models & Data expectations
- Canonical dataset: `data/crypto_clean_BTC_ETH_BNB.csv`.
- Required columns for API services: `date` (parseable), `symbol`, `price_usd`.
- Supported symbols are hard-coded: `BTC`, `ETH`, `BNB` (see `api/services/prediction.py` and `api/services/recommendation.py`).
- Forecasting: `prediction.predict()` uses preloaded ARIMA models (if available) and `forecast_arima()` from `timeseries/models_arima.py`.

## Testing hints for AI changes
- Use `pytest` and `httpx.AsyncClient` for endpoint tests (see `tests/test_api.py`).
- Tests tolerate missing dataset: predictions can return 400 (dataset missing) or 500; tests should assert for those statuses where appropriate.
- To add tests for `recommendations`, follow the existing async pattern and assert on `Envelope` structure.

## Where to change behavior safely
- Add new business logic in `api/services/` (keeping routes thin and focused on request/response conversion).
- Add validation and shapes in `api/schemas/models.py` (Pydantic models). Prefer adding typed models rather than ad-hoc dicts.
- To persist more complex history or provide richer queries, replace `data/recommend_history.json` with a small SQLite store and update `api/services/history.py` accordingly.

## Small but important notes
- Use Spanish for user-facing recommendation text (current messages like "Comprar", "Vender", "Mantener"). Keep translations consistent if added.
- Avoid hardcoding secrets; `COINGECKO_API_KEY` is read from env — instruct users to set `setx COINGECKO_API_KEY "KEY"` on Windows.
- When adding long-running training or heavy computations, avoid running them in FastAPI startup synchronously; prefer background tasks or lazy-loading to keep startup fast.

## Example quick edits an agent might do
- Fix: When adding a new endpoint, mirror the pattern in `api/routes/recommend.py` and add a corresponding `api/services/<name>.py` function that raises `ValueError` for client errors.
- Improve test: Add `tests/test_recommend.py` using `AsyncClient` and assert on `Envelope.status == 'ok'` and `data.symbol`.

---
If anything here is unclear or you'd like more detail about a specific area (e.g. tests, model retraining, or API extension examples), tell me which section to expand or give an example you want included.