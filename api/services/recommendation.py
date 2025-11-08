from __future__ import annotations
from typing import Dict
import pandas as pd
from loguru import logger

from api.utils.loader import load_data

SUPPORTED = {"BTC", "ETH", "BNB"}


def recommend(symbol: str, horizon_days: int) -> Dict:
    sym = symbol.upper()
    if sym not in SUPPORTED:
        raise ValueError(f"Unsupported symbol: {sym}. Supported: {sorted(SUPPORTED)}")
    df = load_data()
    sub = df[df["symbol"] == sym].sort_values("date").tail(max(5, horizon_days))
    if sub.empty:
        raise ValueError(f"No data found for symbol {sym}")
    first = sub.iloc[0]["price_usd"]
    last = sub.iloc[-1]["price_usd"]
    change_pct = ((last - first) / first) * 100 if first else 0
    trend = "up" if last > first else "down"
    if trend == "up" and change_pct > 2:
        advice = "Comprar (tendencia alcista sostenida)"
    elif trend == "down" and change_pct < -2:
        advice = "Vender (tendencia bajista)"
    else:
        advice = "Mantener (movimiento lateral o leve)"
    logger.info(f"Recommendation for {sym}: trend={trend} change={change_pct:.2f}% advice={advice}")
    return {
        "symbol": sym,
        "horizon_days": horizon_days,
        "trend": trend,
        "recommendation": advice,
        "change_pct": round(change_pct, 2)
    }
