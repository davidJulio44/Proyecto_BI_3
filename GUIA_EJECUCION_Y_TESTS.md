# Guía de Ejecución y Pruebas

Esta guía resume cómo preparar el entorno, generar datos, ejecutar análisis, lanzar la API y correr pruebas automatizadas del proyecto.

## 1. Preparar el Entorno

### Opción A: venv + pip (Windows PowerShell)
```powershell
# Posicionarse en la raíz del proyecto
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main

# Crear entorno virtual
python -m venv .venv

# Activar entorno
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Opción B: Poetry
```powershell
cd d:\Users\User\Downloads\Proyecto_BI_3-main\Proyecto_BI_3-main
poetry install
poetry shell
```

### Variables de Entorno (Opcional CoinGecko)
```powershell
setx COINGECKO_API_KEY "TU_API_KEY"
```
(Reinicia la terminal para tomar efecto.)

## 2. Generar / Actualizar Datos

1. Extraer datos crudos desde Binance:
```powershell
python .\scripts\extract_data.py
# Salida: data\raw_crypto.csv
```
2. Limpiar y crear features:
```powershell
python .\scripts\clean_data.py
# Entrada: data\raw_crypto.csv
# Salida: data\crypto_clean_BTC_ETH_BNB.csv
```

## 3. Ejecutar Análisis (Notebook)
```powershell
jupyter notebook scripts\analysis_pipeline.ipynb
```
O en VS Code: abrir el archivo `scripts/analysis_pipeline.ipynb` y ejecutar las celdas.

## 4. Lanzar la API (FastAPI)

### Con venv
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

### Con Poetry
```powershell
poetry run uvicorn api.main:app --reload --port 8000
```

Documentación interactiva: http://127.0.0.1:8000/docs

### Endpoints Disponibles
| Método | Ruta | Descripción | Parámetros |
|--------|------|-------------|------------|
| GET | /health/ | Estado del servicio | - |
| POST | /predict/ | Forecast ARIMA | body: symbol, days |
| GET | /recommendations/ | Recomendación heurística | query: symbol, days |
| GET | /history/ | Últimos registros guardados | query: limit |

### Ejemplos cURL (PowerShell)
```powershell
curl -X GET "http://127.0.0.1:8000/health/"

curl -X POST "http://127.0.0.1:8000/predict/" ^
  -H "Content-Type: application/json" ^
  -d '{"symbol":"BTC","days":7}'

curl -X GET "http://127.0.0.1:8000/recommendations/?symbol=ETH&days=5"

curl -X GET "http://127.0.0.1:8000/history/?limit=10"
```

## 5. Pruebas Automatizadas

### Ejecutar Tests

Con venv:
```powershell
pytest -q
```
Con Poetry:
```powershell
poetry run pytest -q
```

Archivo principal de pruebas: `tests/test_api.py` (verifica /health y /predict).

### Añadir Nuevas Pruebas (Ejemplo)
Crear `tests/test_recommend.py`:
```python
import pytest
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_recommend_btc():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/recommendations/?symbol=BTC&days=7")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["data"]["symbol"] == "BTC"
```

## 6. Logs y Historial
- Middleware registra cada petición (loguru).
- Historial persistente: `data/recommend_history.json` (se escribe en /predict y /recommendations).

## 7. Troubleshooting Rápido
| Problema | Causa | Solución |
|----------|-------|----------|
| ModuleNotFoundError | Entorno no activado | Activar venv / usar poetry shell |
| 400 Unsupported symbol | Símbolo inválido | Usar BTC, ETH o BNB |
| Forecast falla (500) | Dataset ausente | Ejecutar scripts de ETL (extract + clean) |
| No arranca API | Dependencias faltan | pip install -r requirements.txt |
| Notebook sin kernel | Kernel no instalado | python -m ipykernel install --user --name crypto-bi3 |

## 8. Flujo Completo Resumido
```text
(1) Crear entorno → (2) Instalar deps → (3) ETL (extract + clean) → (4) Notebook / API → (5) Tests → (6) Revisar logs/historial
```

## 9. Próximas Mejoras
- Versionado de API `/api/v1`.
- Recomendaciones basadas en clustering y señales técnicas (RSI, MACD).
- Persistencia en SQLite para el historial con filtros.
- CI (GitHub Actions): lint + tests + build imagen Docker.
- Seguridad: CORS restringido, auth opcional.
- Cache TTL para predicciones repetidas.

## 10. Referencias Clave
- Dataset principal: `data/crypto_clean_BTC_ETH_BNB.csv`
- Modelos ARIMA: `timeseries/models_arima.py`
- Modelos RNN: `timeseries/models_rnn.py`
- Servicios API: `api/services/`
- Rutas API: `api/routes/`
- Esquemas: `api/schemas/models.py`
- Tests: `tests/`

---
Guía lista para ejecutar y validar el proyecto end-to-end.
