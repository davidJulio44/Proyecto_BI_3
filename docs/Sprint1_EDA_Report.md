# Sprint 1 — Recolección, Limpieza y EDA (BTC, ETH, BNB)

Este informe documenta el proceso completo del Sprint 1: fuentes, extracción, reglas de limpieza, dataset entregable y análisis exploratorio con evidencias.

## 1) Fuentes de datos confiables
- CoinGecko — endpoint `/coins/{id}/market_chart` (históricos diarios). Requiere API key. Uso: `scripts/extract_real_data.py`.
- Binance — endpoint público `/api/v3/klines` (interval=1d). Alternativa sin API key. Uso: `scripts/extract_data.py`.

Referencias:
- CoinGecko API: https://www.coingecko.com/en/api/documentation
- Binance Klines: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data

## 2) Extracción: métodos y ejecución
Rutas soportadas para reproducibilidad:
- CoinGecko (recomendada, extracción + limpieza + features en un paso):
  - `python scripts/extract_real_data.py --days 3650 --vs_currency usd --out data/crypto_clean_BTC_ETH_BNB.csv`
- Binance (crudo) + limpieza:
  - `python scripts/extract_data.py` → `data/raw_crypto.csv`
  - `python scripts/clean_data.py` → `data/crypto_clean_BTC_ETH_BNB.csv`

Ambas rutas escriben en `data/` para un almacenamiento estructurado consistente.

## 3) Reglas de limpieza y features
Aplicadas en `scripts/clean_data.py` y `scripts/extract_real_data.py`:
1. Normalización de fechas y orden cronológico por activo
2. Eliminación de duplicados por clave `(coin_id, date)`
3. Manejo de nulos críticos: se eliminan filas con `price_usd` nulo
4. Features derivadas por activo:
   - `daily_return`, `log_return`
   - `roll_vol_30d` (volatilidad anualizada `std(30d)*sqrt(365)`)
   - `roll_mean_30d` (media móvil 30d del precio)

## 4) Dataset limpio (entregable)
- Archivo: `data/crypto_clean_BTC_ETH_BNB.csv`
- Columnas: `date, coin_id, symbol, price_usd, market_cap_usd, total_volume_usd, daily_return, log_return, roll_vol_30d, roll_mean_30d`
- Calidad asegurada: sin duplicados `(coin_id, date)`; fechas normalizadas; retornos y ventanas calculadas por activo.
- Entrega: subir a Google Drive y compartir enlace de solo lectura (añadir enlace aquí si se desea).

## 5) EDA — Estadísticas y gráficos
Ejecución: `python scripts/eda_report.py` (por defecto lee `data/` y escribe en `reports/eda`).

Salidas generadas en `reports/eda/`:
- `EDA_summary.csv` (por activo: min/max fecha, filas, medias/medianas/std de precio, media/std de retornos, nulos en retornos)
- Gráficas:
  - `price_history_BTC.png`, `price_history_ETH.png`, `price_history_BNB.png`
  - `rolling_vol_30d_BTC.png`, `rolling_vol_30d_ETH.png`, `rolling_vol_30d_BNB.png`
  - `returns_hist_BTC.png`, `returns_hist_ETH.png`, `returns_hist_BNB.png`

Lecturas clave del EDA (cualitativo):
- Las series de precio muestran ciclos marcados, con periodos de alta volatilidad.
- La volatilidad anualizada 30D captura bien los regímenes de riesgo.
- Las distribuciones de retornos tienen colas pesadas; considerar técnicas robustas u outlier handling en siguientes sprints.

## 6) Tecnologías
- Python 3.10+, pandas, numpy, requests, matplotlib, seaborn
- Próximo sprint: Streamlit (dashboard) y FastAPI (servir datos/modelos)

## Resultados y conclusiones
- Se obtuvo un dataset limpio y enriquecido listo para modelamiento y visualización.
- Se generó un informe EDA reproducible con estadísticas por activo y tres familias de gráficos.
- El pipeline es automatizable y parametrizable (días, moneda de referencia, rutas).

## Limitaciones y riesgos
- Dependencia de la disponibilidad y rate limits de la API de CoinGecko/Binance.
- Posibles huecos temporales si las APIs no entregan ciertos días (a validar en Sprint 2).
- Outliers y cambios de régimen requieren estrategias adicionales.

## Próximos pasos (Sprint 2)
- Validación de huecos temporales y control de outliers
- Indicadores técnicos adicionales y features de lags
- Dashboard en Streamlit y endpoint en FastAPI
- Pruebas automatizadas y linting (ruff/mypy) para calidad continua
