"""
extract_data.py (Binance, sin API key)

Descarga datos diarios (klines 1d) desde la API pública de Binance para
BTC, ETH y BNB contra USDT y los guarda en un CSV crudo: raw_crypto.csv.

Notas
-----
- No requiere API key, pero respeta límites de 1000 velas por petición.
- Para históricos largos, itera con startTime en milisegundos.
- Este archivo produce un dataset crudo; usa clean_data.py para limpiar.
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

PAIRS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT"
}

SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB"
}

def fetch_binance_klines(symbol: str):
    """Descarga klines 1d de Binance para un símbolo como "BTCUSDT".

    Devuelve la lista cruda de listas con el formato oficial de Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": 1000  # Binance máximo por llamada
    }

    all_data = []
    start_time = 0

    while True:
        params["startTime"] = start_time
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_data.extend(data)
        # último timestamp * 1000
        last_time = data[-1][0]
        start_time = last_time + 86400000  # avanzar 1 día

        if len(data) < 1000:
            break

    return all_data

def build_dataframe(coin: str, data) -> pd.DataFrame:
    """Convierte el payload de klines a un DataFrame con columnas útiles.

    Salida: [date, coin_id, symbol, price_usd, total_volume_usd]
    """
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price_usd"] = df["close"].astype(float)
    df["total_volume_usd"] = df["quote_asset_volume"].astype(float)

    df["coin_id"] = coin
    df["symbol"] = SYMBOLS[coin]

    return df[["date","coin_id","symbol","price_usd","total_volume_usd"]]

def run() -> None:
    """CLI simple: descarga los tres activos y escribe data/raw_crypto.csv.

    Notas de rutas
    --------------
    - Este script vive en scripts/. Guardaremos salidas en la carpeta raíz del
      proyecto bajo data/ para mantener consistencia.
    """
    # Descubrir raíz del proyecto (carpeta padre de scripts/)
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for coin, pair in PAIRS.items():
        print(f"📌 Descargando {coin} desde Binance...")
        kline_data = fetch_binance_klines(pair)
        df = build_dataframe(coin, kline_data)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    out_path = DATA_DIR / "raw_crypto.csv"
    raw.to_csv(out_path, index=False)
    print(f"\n✅ Archivo generado: {out_path.relative_to(ROOT)}")

if __name__ == "__main__":
    run()
