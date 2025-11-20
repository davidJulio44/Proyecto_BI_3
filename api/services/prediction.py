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
    """Precarga modelos ARIMA para los símbolos soportados.

    Si los datos están vacíos o hay errores, deja logs pero no rompe el startup.
    """
    global DATA_READY
    df = load_data()
    if df.empty:
        logger.error(
            "Dataset is empty or invalid. Ensure data/crypto_clean_BTC_ETH_BNB.csv exists and is correct."
        )
        DATA_READY = False
        return

    for sym in SUPPORTED:
        try:
            sub = df[df["symbol"] == sym][["date", "price_usd"]].dropna()
            if len(sub) < 20:
                logger.warning(f"Not enough data to train ARIMA for {sym} (len={len(sub)})")
                continue
            model, y_train, y_test, preds, metrics = train_arima(sub, "date", "price_usd")
            CACHE[sym] = {"model": model, "metrics": metrics}
            logger.info(f"Preloaded ARIMA for {sym} with metrics {metrics}")
        except Exception as e:
            logger.exception(f"Failed preloading model for {sym}: {e}")

    DATA_READY = bool(CACHE)


def predict(symbol: str, days: int) -> Dict:
    """Realiza forecast para un símbolo.

    Lanza ValueError con mensajes claros cuando:
    - el símbolo no es soportado
    - no hay datos/modelo disponible
    """
    sym = symbol.upper()
    if sym not in SUPPORTED:
        raise ValueError(f"Unsupported symbol: {sym}. Supported: {sorted(SUPPORTED)}")
    if sym not in CACHE:
        preload_models()
    if sym not in CACHE:
        raise ValueError(
            f"No model available for {sym}. "
            "Check that data/crypto_clean_BTC_ETH_BNB.csv has enough rows for this symbol."
        )
    model = CACHE[sym]["model"]
    metrics = CACHE[sym]["metrics"]
    series = forecast_arima(model, steps=days)
    # Convertir a lista de {date,value}
    start = series.index[0].date()
    out: List[dict] = []
    for idx, val in enumerate(series.tolist()):
        out.append(
            {
                "date": (start + pd.Timedelta(days=idx)).date().isoformat(),
                "value": float(val),
            }
        )
    return {"symbol": sym, "model": "ARIMA", "metrics": metrics, "forecast": out}
