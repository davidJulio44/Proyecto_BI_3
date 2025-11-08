from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
from loguru import logger

from api.utils.loader import load_data
from timeseries.models_arima import train_arima, forecast_arima

SUPPORTED = {"BTC", "ETH", "BNB"}
CACHE: Dict[str, dict] = {}

DATA_READY = False


def preload_models() -> None:
    global DATA_READY
    df = load_data()
    if df.empty:
        logger.error("Dataset is empty. Ensure data/crypto_clean_BTC_ETH_BNB.csv exists.")
        return
    for sym in SUPPORTED:
        try:
            sub = df[df["symbol"] == sym][["date", "price_usd"]]
            model, y_train, y_test, preds, metrics = train_arima(sub, "date", "price_usd")
            CACHE[sym] = {"model": model, "metrics": metrics}
            logger.info(f"Preloaded ARIMA for {sym} with metrics {metrics}")
        except Exception as e:
            logger.exception(f"Failed preloading model for {sym}: {e}")
    DATA_READY = True


def predict(symbol: str, days: int) -> Dict:
    sym = symbol.upper()
    if sym not in SUPPORTED:
        raise ValueError(f"Unsupported symbol: {sym}. Supported: {sorted(SUPPORTED)}")
    if sym not in CACHE:
        preload_models()
    model = CACHE[sym]["model"]
    metrics = CACHE[sym]["metrics"]
    series = forecast_arima(model, steps=days)
    # Convert to list of {date,value}
    start = series.index[0].date()
    out: List[dict] = []
    for idx, val in enumerate(series.tolist()):
        out.append({"date": (start + pd.Timedelta(days=idx)).date().isoformat(), "value": float(val)})
    return {"symbol": sym, "model": "ARIMA", "metrics": metrics, "forecast": out}
