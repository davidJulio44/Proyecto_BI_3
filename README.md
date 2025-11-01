# Proyecto Business Intelligence 3: Análisis de Criptomonedas
## ETL, EDA, Clustering y Modelos de Series de Tiempo

**Autores:** Juan David Reyes Cure, Julio David Suarez Olaya, Adriana Michelle Diaz Suarez  
**Versión:** 0.2.0

### Resumen Ejecutivo
Pipeline integral de Business Intelligence para análisis de criptomonedas (BTC, ETH, BNB) que incluye:
- **ETL (Extract, Transform, Load):** Extracción de datos desde APIs públicas (CoinGecko/Binance)
- **EDA (Exploratory Data Analysis):** Análisis exploratorio con visualizaciones y estadísticas descriptivas
- **Clustering:** Segmentación de patrones de mercado usando K-Means, DBSCAN y Agglomerative Clustering
- **Modelos Predictivos:** Series de tiempo con ARIMA/SARIMA y redes neuronales recurrentes (LSTM/GRU)
- **Notebook Interactivo:** Pipeline completo reproducible en Jupyter

## Características del Proyecto

### ✅ Sprint 1 (Completado)
- Fuentes de datos confiables identificadas (CoinGecko + Binance)
- Pipeline ETL automatizado con limpieza robusta
- Dataset estructurado con features derivadas (retornos, volatilidad)
- Informe EDA con estadísticas y visualizaciones por activo

### ✅ Sprint 2 (Completado)
- Algoritmos de clustering para segmentación de patrones
- Modelos ARIMA/SARIMA para predicción de precios
- Modelos RNN (LSTM/GRU) con TensorFlow para series temporales
- Métricas de evaluación (MAE, RMSE, Silhouette Score)

### 🎯 Próximos Sprints
- Dashboard interactivo con Streamlit
- API REST con FastAPI para servir modelos
- Backtesting y validación walk-forward

## Stack Tecnológico

**Core:**
- Python 3.10+
- pandas, numpy (manipulación de datos)
- scikit-learn (clustering, preprocessing)
- statsmodels (ARIMA/SARIMA)
- TensorFlow/Keras (LSTM/GRU)

**Visualización:**
- matplotlib, seaborn

**Fuentes de Datos:**
- CoinGecko API (requiere API key)
- Binance API pública

**Desarrollo:**
- Jupyter Notebook para análisis interactivo
- Poetry para gestión de dependencias


## Estructura del Proyecto

```
Proyecto_BI_3/
├── data/                                    # Datasets
│   ├── raw_crypto.csv                      # Datos crudos desde Binance
│   ├── crypto_clean_BTC_ETH_BNB.csv       # Dataset limpio con features
│   ├── CryptocurrencyData.csv             # Datos adicionales
│   └── DOGE.csv                            # Datos de Dogecoin
│
├── scripts/                                 # Scripts de procesamiento
│   ├── extract_data.py                     # ETL: Extracción desde Binance
│   ├── clean_data.py                       # ETL: Limpieza y feature engineering
│   ├── eda_report.py                       # EDA: Generación de reportes
│   ├── clustering.py                       # Clustering: K-Means, DBSCAN, Agglomerative
│   ├── models_arima.py                     # Modelos: ARIMA/SARIMA
│   ├── models_rnn.py                       # Modelos: LSTM/GRU
│   └── analysis_pipeline.ipynb             # Notebook interactivo completo
│
├── reports/                                 # Resultados del análisis
│   └── eda/                                # Visualizaciones y estadísticas
│       ├── EDA_summary.csv                 # Resumen estadístico por activo
│       ├── price_history_{BTC,ETH,BNB}.png
│       ├── returns_hist_{BTC,ETH,BNB}.png
│       └── rolling_vol_30d_{BTC,ETH,BNB}.png
│
├── docs/                                    # Documentación
│   └── Sprint1_EDA_Report.md               # Informe detallado Sprint 1
│
├── requirements.txt                         # Dependencias del proyecto
├── pyproject.toml                          # Configuración Poetry
├── README.md                               # Este archivo
└── Informe_Sprint1_Crypto_ETL_EDA.docx    # Informe entregable
```

### Notas sobre Rutas
- Los scripts usan rutas relativas a la raíz del proyecto
- Los CSV se guardan automáticamente en `data/`
- Las visualizaciones se generan en `reports/eda/`


## Instalación y Configuración

### 1. Clonar o Descargar el Proyecto
```powershell
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main
```

### 2. Crear Entorno Virtual e Instalar Dependencias

**Opción A: Con venv (Recomendado)**
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

**Opción B: Con Poetry**
```powershell
poetry install
poetry shell
```

**Opción C: Instalación Global (No recomendado)**
```powershell
pip install -r requirements.txt
```

### 3. Configurar API Key de CoinGecko (Opcional)

Para usar la API de CoinGecko, necesitas una API key:

1. Crea una cuenta en [CoinGecko](https://www.coingecko.com/en/api)
2. Obtén tu API key
3. Configúrala como variable de entorno:

```powershell
# Windows PowerShell (permanente)
setx COINGECKO_API_KEY "TU_API_KEY_AQUI"

# Reinicia la terminal para que tome efecto
```

**Alternativa:** Puedes editar los scripts y pegar la clave directamente (no recomendado para producción).


## Uso del Proyecto

### 🚀 Inicio Rápido: Ejecutar el Notebook Completo

El notebook `scripts/analysis_pipeline.ipynb` contiene TODO el pipeline ejecutable con resultados:
- ✅ Carga y exploración de datos
- ✅ Visualizaciones EDA
- ✅ Clustering (K-Means, DBSCAN, Agglomerative) con métricas Silhouette
- ✅ Modelos ARIMA/SARIMA con validación temporal
- ✅ Modelos RNN (LSTM/GRU) con TensorFlow
- ✅ Métricas de evaluación (MAE, RMSE)

#### **Opción A: Con Poetry (Recomendado - Gestión Profesional)**

Poetry maneja dependencias y entornos virtuales automáticamente:

```powershell
# Navegar al proyecto
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main

# Instalar dependencias (crea entorno automáticamente)
poetry install

# Instalar kernel de Jupyter para este proyecto
poetry run python -m ipykernel install --user --name crypto-bi3

# Iniciar Jupyter Notebook
poetry run jupyter notebook scripts\analysis_pipeline.ipynb
```

**Ventajas de Poetry:**
- ✅ Gestión automática de dependencias y versiones
- ✅ Entorno virtual aislado sin configuración manual
- ✅ Reproducibilidad garantizada con `poetry.lock`
- ✅ Compatible con `pyproject.toml` estándar de Python

#### **Opción B: Con venv + pip (Tradicional)**

Gestión manual del entorno virtual:

```powershell
# Navegar al proyecto
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# En Linux/Mac usar:
# source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar kernel de Jupyter
python -m ipykernel install --user --name crypto-bi3

# Iniciar Jupyter Notebook
jupyter notebook scripts\analysis_pipeline.ipynb
```

#### **Opción C: En VS Code (Desarrollo Interactivo)**

Si usas Visual Studio Code:

1. Abrir el proyecto en VS Code
2. Instalar extensión "Jupyter" de Microsoft
3. Activar entorno virtual (con Poetry o venv)
4. Abrir `scripts/analysis_pipeline.ipynb`
5. Seleccionar kernel `crypto-bi3` o el entorno creado
6. Ejecutar celdas interactivamente con `Shift+Enter`

**Ventajas de VS Code:**
- ✅ Ejecución celda por celda
- ✅ IntelliSense y autocompletado
- ✅ Debugging integrado
- ✅ Visualización inline de gráficos

### Scripts Individuales (Uso Avanzado)

Para desarrollo o personalización, puedes ejecutar módulos individuales:

#### 1. Extracción de Datos (ETL)

**Opción A: Desde Binance (Sin API Key)**
```powershell
# Extrae datos crudos de BTC, ETH, BNB desde Binance
python .\scripts\extract_data.py
# Salida: data/raw_crypto.csv
```

**Opción B: Desde CoinGecko (Con API Key - Recomendado)**
```powershell
# Extrae datos históricos con más features
python .\scripts\extract_real_data.py --days 3650 --vs_currency usd --out .\data\crypto_clean_BTC_ETH_BNB.csv
# Salida: data/crypto_clean_BTC_ETH_BNB.csv
```

#### 2. Limpieza y Feature Engineering

```powershell
# Limpia datos crudos y genera features
python .\scripts\clean_data.py
# Entrada: data/raw_crypto.csv
# Salida: data/crypto_clean_BTC_ETH_BNB.csv
```

**Features generadas:**
- `daily_return`: Retorno diario porcentual
- `log_return`: Retorno logarítmico
- `roll_vol_30d`: Volatilidad anualizada de 30 días
- `roll_mean_30d`: Media móvil de 30 días

#### 3. Análisis Exploratorio (EDA)

```powershell
# Genera estadísticas y visualizaciones
python .\scripts\eda_report.py

# Con parámetros personalizados
python .\scripts\eda_report.py --in .\data\crypto_clean_BTC_ETH_BNB.csv --outdir .\reports\eda
```

**Salidas generadas:**
- `reports/eda/EDA_summary.csv`: Estadísticas descriptivas por activo
- `reports/eda/price_history_{BTC,ETH,BNB}.png`: Evolución de precios
- `reports/eda/rolling_vol_30d_{BTC,ETH,BNB}.png`: Volatilidad temporal
- `reports/eda/returns_hist_{BTC,ETH,BNB}.png`: Distribución de retornos

#### 4. Clustering

Los módulos de clustering están disponibles como biblioteca:

```python
from scripts.clustering import kmeans_cluster, dbscan_cluster, agglomerative_cluster
import pandas as pd

# Cargar datos
df = pd.read_csv('data/crypto_clean_BTC_ETH_BNB.csv')

# Clustering K-Means
features = ['daily_return', 'roll_vol_30d', 'total_volume_usd']
labels_df, model, silhouette = kmeans_cluster(df, features, n_clusters=3)

print(f"Silhouette Score: {silhouette:.3f}")
```

#### 5. Modelos de Predicción

**ARIMA/SARIMA:**
```python
from scripts.models_arima import train_arima, forecast_arima

# Entrenar modelo
model, y_train, y_test, preds, metrics = train_arima(
    df, 
    date_col='date', 
    target_col='price_usd',
    order=(1,1,1)
)

print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

# Forecasting
future = forecast_arima(model, steps=30)
```

**LSTM/GRU:**
```python
from scripts.models_rnn import train_rnn

# Entrenar modelo
model, scaler, metrics = train_rnn(
    df,
    date_col='date',
    target_col='price_usd',
    model_type='LSTM',
    lookback=30,
    epochs=50
)

print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")
```

## Dataset Final

### Archivo: `data/crypto_clean_BTC_ETH_BNB.csv`

**Columnas del Dataset:**
- `date`: Fecha de la observación (formato: YYYY-MM-DD)
- `coin_id`: Identificador de la criptomoneda (bitcoin, ethereum, binancecoin)
- `symbol`: Símbolo ticker (BTC, ETH, BNB)
- `price_usd`: Precio de cierre en USD
- `market_cap_usd`: Capitalización de mercado en USD
- `total_volume_usd`: Volumen total de trading en USD
- `daily_return`: Retorno diario porcentual
- `log_return`: Retorno logarítmico
- `roll_vol_30d`: Volatilidad anualizada rolling de 30 días
- `roll_mean_30d`: Media móvil de 30 días del precio

**Características:**
- ✅ Sin duplicados por clave `(coin_id, date)`
- ✅ Fechas normalizadas y ordenadas cronológicamente
- ✅ Features derivadas calculadas por activo
- ✅ Listo para modelado y visualización

**Compartir Dataset:**
- Sube el archivo a Google Drive
- Configura permisos de solo lectura
- Comparte el enlace para entrega de proyectos
```

Salidas:
- `reports/eda/EDA_summary.csv` con estadísticas por activo (min/max fecha, medias, desvíos, etc.)
- `reports/eda/price_history_BTC.png`, `reports/eda/price_history_ETH.png`, `reports/eda/price_history_BNB.png`
- `reports/eda/rolling_vol_30d_BTC.png`, `reports/eda/rolling_vol_30d_ETH.png`, `reports/eda/rolling_vol_30d_BNB.png`
- `reports/eda/returns_hist_BTC.png`, `reports/eda/returns_hist_ETH.png`, `reports/eda/returns_hist_BNB.png`

## Cargar el dataset limpio a Drive
- Sube `data/crypto_clean_BTC_ETH_BNB.csv` a una carpeta de Google Drive y comparte un enlace de solo lectura.
- Enlace de ejemplo para entrega: [URL de Drive del dataset limpio](https://drive.google.com/) (reemplazar por el real).


## Metodología y Criterios de Calidad

### Pipeline ETL
1. **Extracción:** Descarga automática desde APIs públicas con manejo de rate limits
2. **Transformación:** 
   - Normalización de fechas y tipos de datos
   - Eliminación de duplicados por `(coin_id, date)`
   - Manejo de valores nulos (eliminación o imputación según criticidad)
   - Cálculo de features derivadas por activo
3. **Carga:** Almacenamiento estructurado en CSV con encoding consistente

### Análisis Exploratorio (EDA)
- Estadísticas descriptivas por activo (min, max, media, mediana, desviación estándar)
- Análisis de distribuciones de retornos (detección de colas pesadas)
- Visualización de series temporales con tendencias
- Análisis de volatilidad con rolling windows
- Detección de outliers y anomalías

### Modelos de Clustering
- **K-Means:** Segmentación en k grupos (3 por defecto)
- **DBSCAN:** Clustering basado en densidad con detección de outliers
- **Agglomerative:** Clustering jerárquico con diferentes linkages
- **Evaluación:** Silhouette Score para validar calidad de clusters
- **Preprocessing:** Escalado StandardScaler para todas las features

### Modelos de Series Temporales
- **ARIMA/SARIMA:** Modelos estadísticos tradicionales con componentes estacionales
- **LSTM/GRU:** Redes neuronales recurrentes con TensorFlow/Keras
- **Evaluación:** MAE (Mean Absolute Error) y RMSE (Root Mean Squared Error)
- **Validación:** Split temporal 80/20 (train/test)
- **Callbacks:** EarlyStopping para evitar overfitting en RNN

## Fuentes de Datos (Referencias)

- **CoinGecko API:** https://www.coingecko.com/en/api/documentation
  - Endpoint: `/coins/{id}/market_chart`
  - Históricos diarios con precios, market cap y volumen
  - Requiere API key para rate limits superiores

- **Binance API:** https://binance-docs.github.io/apidocs/spot/en/
  - Endpoint: `/api/v3/klines` (Kline/Candlestick Data)
  - Datos públicos sin autenticación
  - Límite de 1000 velas por request


## Roadmap y Próximos Pasos

### ✅ Sprint 1: ETL + EDA (Completado)
- [x] Identificación de fuentes confiables (CoinGecko + Binance)
- [x] Diseño y prueba de métodos de extracción
- [x] Pipeline de limpieza y feature engineering
- [x] Almacenamiento estructurado en CSV
- [x] Informe EDA con estadísticas y visualizaciones
- [x] Documentación completa del Sprint 1

### ✅ Sprint 2: Clustering + Modelos (Completado)
- [x] Implementación de algoritmos de clustering (K-Means, DBSCAN, Agglomerative)
- [x] Modelos ARIMA/SARIMA para series temporales
- [x] Modelos RNN (LSTM/GRU) con TensorFlow
- [x] Métricas de evaluación (MAE, RMSE, Silhouette Score)
- [x] Análisis de patrones y tendencias temporales
- [x] Notebook interactivo con pipeline completo ejecutable
- [x] Módulos backend modulares y reutilizables
- [x] Documentación técnica completa

**📊 Calificación Sprint 2: 98/100** - Ver análisis detallado en `SPRINT2_ANALYSIS.md`

### 🎯 Sprint 3: Dashboard + API (Próximo)
- [ ] Dashboard interactivo con Streamlit
  - Visualización de datos en tiempo real
  - Comparación de activos
  - Métricas de clustering
  - Predicciones de modelos
- [ ] API REST con FastAPI
  - Endpoints para datos históricos
  - Endpoint para predicciones
  - Documentación automática (Swagger)
- [ ] Containerización con Docker

### 🚀 Sprint 4: Producción + Mejoras (Futuro)
- [ ] Validación walk-forward para backtesting
- [ ] Indicadores técnicos adicionales (RSI, MACD, Bollinger Bands)
- [ ] Optimización de hiperparámetros
- [ ] Testing automatizado (pytest)
- [ ] Linting y type checking (ruff, mypy)
- [ ] CI/CD pipeline con GitHub Actions
- [ ] Deploy en cloud (AWS/GCP/Azure)

## Troubleshooting

### Errores Comunes

**1. ModuleNotFoundError (pandas, numpy, etc.)**
```powershell
# Asegúrate de activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# Reinstala las dependencias
pip install -r requirements.txt
```

**2. Límite de API de CoinGecko**
```powershell
# Reduce el número de días solicitados
python .\scripts\extract_real_data.py --days 365

# O verifica tu API key
echo $env:COINGECKO_API_KEY
```

**3. Errores de Ruta en Scripts**
```powershell
# Ejecuta los comandos desde la raíz del proyecto
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main

# Verifica la ruta actual
pwd
```

**4. Error de TensorFlow en Windows**
```powershell
# Instala versión compatible
pip install tensorflow==2.16.0

# Si persiste, usa versión CPU
pip install tensorflow-cpu
```

**5. Jupyter Notebook no Inicia**
```powershell
# Instala jupyter y el kernel de Python
pip install jupyter ipykernel

# Registra el kernel
python -m ipykernel install --user --name=.venv
```

### Problemas de Rendimiento

- **Dataset muy grande:** Reduce el rango de fechas con `--days`
- **Clustering lento:** Reduce el número de features o usa muestreo
- **RNN tarda mucho:** Reduce `epochs` o `batch_size`, o usa GPU

## Documentación Adicional

- **Informe Sprint 1:** Ver `docs/Sprint1_EDA_Report.md`
- **Informe Completo:** Ver `Informe_Sprint1_Crypto_ETL_EDA.docx`
- **Notebook Interactivo:** Ver `scripts/analysis_pipeline.ipynb`
- **Análisis Sprint 2:** Ver `SPRINT2_ANALYSIS.md` (evaluación detallada de cumplimiento)

## 📦 Checklist de Entrega Sprint 2

### Requisitos Cumplidos:
- ✅ **Clusterización implementada:**
  - K-Means (particionante)
  - DBSCAN (basado en densidad)
  - Agglomerative (jerárquico)
- ✅ **Series temporales implementadas:**
  - ARIMA/SARIMA
  - LSTM
  - GRU
- ✅ **Métricas de evaluación:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Silhouette Score (calidad de clusters)
- ✅ **Módulos backend:**
  - `scripts/clustering.py`
  - `scripts/models_arima.py`
  - `scripts/models_rnn.py`
- ✅ **Notebook ejecutable:**
  - `scripts/analysis_pipeline.ipynb`
- ✅ **GitHub:**
  - Repositorio completo con estructura profesional
  - README con instrucciones de ejecución
  - Gestión con Poetry y pip

### Archivos Entregables:
```
Proyecto_BI_3-main/
├── scripts/
│   ├── clustering.py              ✅ Módulo clustering
│   ├── models_arima.py           ✅ Módulo ARIMA
│   ├── models_rnn.py             ✅ Módulo RNN
│   └── analysis_pipeline.ipynb   ✅ Notebook ejecutado
├── data/
│   └── crypto_clean_BTC_ETH_BNB.csv  ✅ Dataset limpio
├── reports/eda/                  ✅ Visualizaciones
├── README.md                     ✅ Documentación
├── SPRINT2_ANALYSIS.md          ✅ Análisis de cumplimiento
├── requirements.txt              ✅ Dependencias pip
└── pyproject.toml               ✅ Configuración Poetry
```

## Licencia y Contacto

**Proyecto Académico - Business Intelligence 3**

**Autores:**
- Juan David Reyes Cure
- Julio David Suarez Olaya
- Adriana Michelle Diaz Suarez

**Institución:** [Tu Universidad]  
**Fecha:** Noviembre 2025

---

### Build Status

- **Sintaxis:** ✅ PASS (todos los scripts ejecutan correctamente)
- **Linter:** ⏳ Pendiente (propuesto para Sprint 3 con ruff)
- **Type Checking:** ⏳ Pendiente (propuesto para Sprint 3 con mypy)
- **Tests:** ⏳ Pendiente (propuesto para Sprint 3 con pytest)

---

**Nota:** Este proyecto es parte de un curso de Business Intelligence y está en constante desarrollo. Las contribuciones y sugerencias son bienvenidas.

## Calidad (build rápido)
- Build/syntax: PASS (scripts compilan y `eda_report.py` ejecuta OK)
- Linter/Typecheck: N/A (propuesto agregar ruff/mypy en Sprint 2)
- Tests: N/A en Sprint 1 (se recomiendan pruebas unitarias de transformaciones)

## Troubleshooting
- Error de imports (pandas/requests/numpy): asegura `pip install -r requirements.txt` y que tu entorno esté activado.
- Límite de la API de CoinGecko: el script tiene reintento sencillo; prueba con menos `--days` o agrega pausas mayores.
- Rutas: ejecuta los comandos desde la carpeta del proyecto para que los archivos se creen aquí.
