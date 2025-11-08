# 📚 Documentación Completa del Proyecto

**Proyecto:** Business Intelligence 3 – Análisis Integral de Criptomonedas (BTC, ETH, BNB)  
**Versión:** 0.2.0  
**Autores:** Juan David Reyes Cure · Julio David Suarez Olaya · Adriana Michelle Diaz Suarez  
**Fecha:** Noviembre 2025

---
## 🧭 Tabla de Contenido
1. [Resumen Ejecutivo](#-resumen-ejecutivo)
2. [Objetivos por Sprint](#-objetivos-por-sprint)
3. [Arquitectura y Estructura](#-arquitectura-y-estructura)
4. [Instalación y Entornos](#-instalación-y-entornos)
5. [Uso Rápido (Quick Start)](#-uso-rápido-quick-start)
6. [Pipeline ETL](#-pipeline-etl)
7. [Dataset Limpio: Especificaciones](#-dataset-limpio-especificaciones)
8. [EDA: Exploratory Data Analysis](#-eda-exploratory-data-analysis)
9. [Feature Engineering](#-feature-engineering)
10. [Modelos de Clustering](#-modelos-de-clustering)
11. [Modelos de Series Temporales](#-modelos-de-series-temporales)
12. [Modelos RNN (LSTM / GRU)](#-modelos-rnn-lstm--gru)
13. [Evaluación y Métricas](#-evaluación-y-métricas)
14. [Notebook Interactivo](#-notebook-interactivo)
15. [Calidad y Revisión de Código](#-calidad-y-revisión-de-código)
16. [Roadmap y Próximos Pasos](#-roadmap-y-próximos-pasos)
17. [Troubleshooting](#-troubleshooting)
18. [Checklist de Entrega Sprint 2](#-checklist-de-entrega-sprint-2)
19. [Recomendaciones Futuras](#-recomendaciones-futuras)
20. [Licencia y Contacto](#-licencia-y-contacto)

---
## 🚀 Resumen Ejecutivo
Pipeline integral de análisis de criptomonedas que cubre: extracción de datos (ETL), exploración de datos (EDA), creación de features avanzadas, segmentación del comportamiento (clustering), y predicción de series temporales mediante modelos estadísticos (ARIMA/SARIMA) y redes neuronales (LSTM/GRU).

El proyecto prioriza:
- Reproducibilidad (scripts independientes y notebook consolidado)
- Escalabilidad (módulos reutilizables)
- Métricas claras (MAE, RMSE, Silhouette)
- Claridad documental (esta guía consolidada)

---
## 🎯 Objetivos por Sprint
### Sprint 1 – ETL + EDA
- Identificar fuentes confiables (CoinGecko, Binance)
- Implementar extracción parametrizable
- Limpieza y normalización del dataset
- Feature engineering inicial (retornos y volatilidad)
- Informe exploratorio con visualizaciones

### Sprint 2 – Clustering + Modelos Predictivos
- Implementar 3 familias de clustering (particionante, densidad, jerárquico)
- Modelos ARIMA/SARIMA para predicción
- Modelos RNN (LSTM / GRU) para series temporales
- Evaluación con métricas estándar
- Documentación y revisión técnica

### Sprint 3 (Próximo) – Visualización y Servicio
- Dashboard interactivo (Streamlit)
- API REST (FastAPI)
- Backtesting y validación walk-forward

### Sprint 4 (Futuro) – Producción y Calidad
- CI/CD, Docker, despliegue cloud
- Linting (ruff), type checking (mypy)
- Tests automatizados (pytest)

---
## 🏗 Arquitectura y Estructura
```
Proyecto_BI_3/
├── data/                      # Datos crudos y limpios
│   ├── raw_crypto.csv
│   ├── crypto_clean_BTC_ETH_BNB.csv
│   └── ...
├── scripts/                   # Módulos backend
│   ├── extract_data.py        # ETL Binance
│   ├── clean_data.py          # Limpieza + features
│   ├── eda_report.py          # EDA automatizado
│   ├── clustering.py          # Algoritmos clustering
│   ├── models_arima.py        # ARIMA/SARIMA
│   ├── models_rnn.py          # LSTM / GRU
│   └── analysis_pipeline.ipynb# Notebook completo
├── reports/eda/               # Visualizaciones + resumen
│   ├── EDA_summary.csv
│   ├── price_history_*.png
│   ├── rolling_vol_30d_*.png
│   └── returns_hist_*.png
├── docs/                      # Documentación parcial
│   └── Sprint1_EDA_Report.md
├── README.md                  # Resumen ejecutivo
├── requirements.txt           # Dependencias pip
├── pyproject.toml             # Configuración Poetry
├── SPRINT2_ANALYSIS.md        # Evaluación Sprint 2
├── CODE_REVIEW.md             # Revisión técnica
└── DOCUMENTACION_COMPLETA.md  # Este documento
```

---
## 🛠 Instalación y Entornos
### Opción A: Poetry (Recomendado)
```powershell
cd Proyecto_BI_3-main
poetry install
poetry run python -m ipykernel install --user --name crypto-bi3
poetry run jupyter notebook scripts/analysis_pipeline.ipynb
```
### Opción B: venv + pip
```powershell
cd Proyecto_BI_3-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name crypto-bi3
jupyter notebook scripts/analysis_pipeline.ipynb
```
### Opción C: VS Code
1. Abrir carpeta del proyecto
2. Instalar extensión Jupyter
3. Activar entorno (Poetry o venv)
4. Abrir `analysis_pipeline.ipynb`
5. Seleccionar kernel `crypto-bi3`

### API Key CoinGecko (Opcional)
```powershell
setx COINGECKO_API_KEY "TU_API_KEY"
```

---
## ⚡ Uso Rápido (Quick Start)
| Tarea | Comando |
|-------|---------|
| Extraer datos Binance | `python scripts/extract_data.py` |
| Limpiar datos | `python scripts/clean_data.py` |
| Generar EDA | `python scripts/eda_report.py` |
| Ejecutar Notebook | Ver sección instalación |

---
## 🔄 Pipeline ETL
1. Extracción (Binance / CoinGecko)
2. Transformación
   - Normalización de tipos y fechas
   - Eliminación de duplicados `(coin_id, date)`
   - Cálculo de retornos y volatilidades
3. Carga en `data/` como CSV estructurado

#### Script: `extract_data.py`
- Descarga klines 1d (BTC, ETH, BNB)
- Paginación automática hasta máximo histórico disponible

#### Script: `clean_data.py`
- Orden cronológico garantizado
- Duplicados eliminados por `(coin_id, date)`
- Rolling windows de 30 días para volatilidad y medias

---
## 🗂 Dataset Limpio: Especificaciones
Archivo principal: `data/crypto_clean_BTC_ETH_BNB.csv`

| Columna | Descripción |
|---------|-------------|
| date | Fecha (YYYY-MM-DD) |
| coin_id | Nombre interno (bitcoin, ethereum, binancecoin) |
| symbol | Ticker (BTC, ETH, BNB) |
| price_usd | Precio de cierre en USD |
| market_cap_usd | Capitalización de mercado |
| total_volume_usd | Volumen total negociado |
| daily_return | Retorno porcentual diario |
| log_return | Retorno logarítmico |
| roll_vol_30d | Volatilidad anualizada (30d) |
| roll_mean_30d | Media móvil de precio (30d) |

Características:
- Fechas ordenadas
- Sin duplicados clave
- Listo para modelado y análisis

---
## 🔍 EDA: Exploratory Data Analysis
Generado vía `eda_report.py`:
- `EDA_summary.csv`: Mínimo, máximo, medias, desviaciones y nulos por activo
- Gráficos por activo:
  - Evolución de precios
  - Distribución de retornos
  - Volatilidad rolling 30D

Insights:
- Ciclos marcados de precio
- Periodos de alta volatilidad concentrados
- Retornos con colas pesadas → no normalidad

---
## 🧬 Feature Engineering
| Feature | Fórmula / Método | Uso |
|---------|------------------|-----|
| daily_return | pct_change() | Señales de momentum |
| log_return | log1p(daily_return) | Estabilidad estadística |
| roll_vol_30d | std(log_return 30d) * sqrt(365) | Riesgo | 
| roll_mean_30d | mean(price 30d) | Tendencia | 

---
## 🧪 Modelos de Clustering
Implementados en `clustering.py`:

### K-Means
- Particionante
- Escalado previo con StandardScaler
- Métrica: Silhouette Score

### DBSCAN
- Basado en densidad
- Detecta outliers (`label = -1`)
- Parámetros: `eps`, `min_samples`

### Agglomerative
- Jerárquico bottom-up
- Linkage configurable: ward / complete / average / single

### Uso Ejemplo
```python
from scripts.clustering import kmeans_cluster
features = ['daily_return','roll_vol_30d']
labels_df, pipe, sil = kmeans_cluster(df.dropna(subset=features), features)
print(sil)
```

---
## ⏱ Modelos de Series Temporales
Archivo: `models_arima.py`

### ARIMA / SARIMA
- Implementado con `statsmodels.SARIMAX`
- Parámetros: `order=(p,d,q)` y `seasonal_order=(P,D,Q,s)`
- Entrenamiento: split temporal 80/20

#### Funciones
```python
train_arima(df, date_col='date', target_col='price_usd', order=(1,1,1))
forecast_arima(model, steps=30)
```

Resultados devuelven:
- Modelo entrenado
- Serie train/test
- Predicciones
- Métricas: MAE, RMSE

---
## 🧠 Modelos RNN (LSTM / GRU)
Archivo: `models_rnn.py`

### Características
- Normalización con MinMaxScaler
- Ventana (lookback) configurable
- Arquitecturas:
  - LSTM(64) → Dense(1)
  - GRU(64) → Dense(1)
- EarlyStopping (patience=5)

#### Función Principal
```python
train_rnn(df, 'date', 'price_usd', model_type='LSTM', lookback=30, epochs=50)
```
Retorna:
- Modelo entrenado
- Scaler
- Métricas: MAE, RMSE

---
## 📏 Evaluación y Métricas
| Tipo | Métricas |
|------|----------|
| Clustering | Silhouette Score |
| Series Temporales | MAE, RMSE |
| RNN | MAE, RMSE |

Interpretación:
- MAE: error promedio absoluto (robusto)
- RMSE: penaliza errores grandes
- Silhouette: cohesión vs separación (-1 a 1)

---
## 🧪 Notebook Interactivo
Archivo: `scripts/analysis_pipeline.ipynb`
Incluye:
1. Carga de datos
2. Clustering con comparación de Silhouette
3. ARIMA aplicado a BTC
4. LSTM aplicado a BTC
5. Visualización comparativa
6. Conclusiones y próximos pasos

Ejecutar:
```powershell
poetry run jupyter notebook scripts/analysis_pipeline.ipynb
```

---
## ✅ Calidad y Revisión de Código
Archivo: `CODE_REVIEW.md` (resumen):
- 0 errores de sintaxis
- Imports correctos
- Modularidad alta
- Type hints ~90%
- Mejoras sugeridas: tests, linting, docstrings extendidas

Calidad Sprint 2: 98/100

---
## 🛣 Roadmap y Próximos Pasos
| Sprint | Objetivo | Estado |
|--------|----------|--------|
| 1 | ETL + EDA | ✅ |
| 2 | Clustering + Modelos | ✅ |
| 3 | Dashboard + API | ⏳ |
| 4 | Producción + Calidad | ⏳ |

---
## 🆘 Troubleshooting
| Problema | Causa | Solución |
|----------|-------|----------|
| ModuleNotFoundError | Entorno no activo | Activar venv / poetry shell |
| Límite API CoinGecko | Exceso de días | Usar `--days 365` o API key |
| TensorFlow falla | Incompatibilidad | `pip install tensorflow==2.16.0` |
| Notebook sin kernel | Kernel no instalado | `python -m ipykernel install --user --name crypto-bi3` |
| RNN lenta | Epochs altos / CPU | Reducir epochs o usar GPU |

---
## 📦 Checklist de Entrega Sprint 2
- ✅ Clustering (K-Means, DBSCAN, Agglomerative)
- ✅ ARIMA / SARIMA
- ✅ LSTM / GRU
- ✅ Métricas (MAE, RMSE, Silhouette)
- ✅ Notebook completo
- ✅ Dataset limpio
- ✅ Documentación consolidada

Calificación: 98/100

---
## 🔮 Recomendaciones Futuras
1. Walk-forward validation
2. Dashboard Streamlit y FastAPI
3. Indicadores técnicos (RSI, MACD, Bollinger)
4. Pytest + ruff + mypy + CI/CD
5. Serialización de modelos (joblib / h5)
6. Monitoreo de drift en producción

---
## 📄 Licencia y Contacto
Proyecto académico para fines educativos.  
Uso restringido a prácticas de Business Intelligence.

Autores:
- Juan David Reyes Cure
- Julio David Suarez Olaya
- Adriana Michelle Diaz Suarez

Contacto: (añadir correos institucionales)

---
**Última Actualización:** Noviembre 2025
