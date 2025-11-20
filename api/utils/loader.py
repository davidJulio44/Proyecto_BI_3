from pathlib import Path
from typing import Set

import pandas as pd
from loguru import logger

DATA_PATH = Path("data/crypto_clean_BTC_ETH_BNB.csv")


def load_data() -> pd.DataFrame:
    """Carga el CSV limpio manejando errores comunes.

    - Si el archivo no existe o está corrupto, devuelve un DataFrame vacío.
    - Si faltan columnas clave, también devuelve vacío y deja log.
    """
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except Exception as e:
        logger.exception(f"Failed reading {DATA_PATH}: {e}")
        return pd.DataFrame()

    required_cols: Set[str] = {"date", "symbol", "price_usd"}
    if not required_cols.issubset(df.columns):
        logger.error(
            f"Data file {DATA_PATH} missing required columns. "
            f"Found={set(df.columns)}, required={required_cols}"
        )
        return pd.DataFrame()

    return df