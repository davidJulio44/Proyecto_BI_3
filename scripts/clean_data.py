"""
clean_data.py

Lee el dataset crudo generado por extract_data.py (Binance) y aplica reglas
de limpieza y características derivadas para producir un CSV listo para modelar.

Entrada: raw_crypto.csv con columnas [date, coin_id, symbol, price_usd, total_volume_usd]
Salida: crypto_clean_BTC_ETH_BNB.csv con columnas base + [daily_return, log_return,
		roll_vol_30d, roll_mean_30d]
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Rutas de proyecto
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos crudos
in_path = DATA_DIR / "raw_crypto.csv"
df = pd.read_csv(in_path)

# Normalizar fecha y ordenar para operaciones temporales estables
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

# Eliminar duplicados por clave (coin, date)
df = df.drop_duplicates(subset=["coin_id", "date"])

# Retornos diarios y logarítmicos por activo
df["daily_return"] = df.groupby("coin_id")["price_usd"].pct_change()
df["log_return"] = np.log1p(df["daily_return"])

# Ventanas móviles de 30 días (volatilidad anualizada y media móvil)
df["roll_vol_30d"] = df.groupby("coin_id")["log_return"].rolling(30).std().reset_index(level=0, drop=True) * np.sqrt(365)
df["roll_mean_30d"] = df.groupby("coin_id")["price_usd"].rolling(30).mean().reset_index(level=0, drop=True)

# Guardar limpio
out_path = DATA_DIR / "crypto_clean_BTC_ETH_BNB.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Archivo generado: {out_path.relative_to(ROOT)} (LISTO PARA MODELOS)")
