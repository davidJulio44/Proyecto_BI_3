"""Simple Streamlit dashboard for Crypto BI3 project.

Features:
- Connects to FastAPI endpoints (/predict, /recommendations, /history, /health)
- Shows key statistics and interactive recommendations
- Uses Plotly for interactive charts
"""

import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("CRYPTO_API_BASE_URL", "http://127.0.0.1:8000")

SUPPORTED_SYMBOLS = ["BTC", "ETH", "BNB"]

DATA_PATH = "data/crypto_clean_BTC_ETH_BNB.csv"


def _api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict) or body.get("status") != "ok":
        raise RuntimeError(f"Unexpected API response: {body}")
    return body


def _api_post(path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    r = requests.post(url, json=json_body, timeout=30)
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict) or body.get("status") != "ok":
        raise RuntimeError(f"Unexpected API response: {body}")
    return body


def page_overview():
    st.title("📊 Dashboard Criptomonedas - Proyecto BI3")
    st.markdown("Interfaz para explorar predicciones ARIMA y recomendaciones heurísticas.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Símbolos soportados", len(SUPPORTED_SYMBOLS))
        st.write(", ".join(SUPPORTED_SYMBOLS))
    with col2:
        try:
            health = _api_get("/health/")
            st.success("API OK")
        except Exception as e:
            st.error(f"API no disponible: {e}")
    with col3:
        st.info("Usa las pestañas de la izquierda para navegar entre vistas.")

    st.markdown("### Historial reciente de consultas")
    try:
        body = _api_get("/history/", params={"limit": 50})
        items: List[Dict[str, Any]] = body.get("data", [])
        if not items:
            st.warning("No hay historial aún. Realiza una predicción o recomendación.")
            return
        # Normalizar a DataFrame amigable
        rows = []
        for it in items:
            ts = it.get("timestamp")
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else ts
            except Exception:
                ts_dt = ts
            kind = "predict" if "days" in it.get("request", {}) and it.get("response", {}).get("model") else "recommend"
            rows.append({
                "timestamp": ts_dt,
                "tipo": kind,
                "symbol": (it.get("request", {}).get("symbol") or "").upper(),
                "days": it.get("request", {}).get("days"),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp", ascending=False)
        st.dataframe(df, use_container_width=True)

        by_symbol = df.groupby("symbol")["tipo"].count().reset_index(name="n_consultas")
        fig = px.bar(by_symbol, x="symbol", y="n_consultas", title="Consultas por símbolo")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo cargar el historial: {e}")


def page_predict():
    st.title("📈 Predicción de precios (ARIMA)")

    with st.sidebar:
        st.header("Parámetros de predicción")
        symbol = st.selectbox("Símbolo", SUPPORTED_SYMBOLS, index=0)
        symbol = (symbol or "").upper()
        days = st.slider("Horizonte (días)", min_value=1, max_value=60, value=7)
        run_btn = st.button("Ejecutar predicción")

    if not run_btn:
        st.info("Configura los parámetros y pulsa 'Ejecutar predicción'.")
        return

    try:
        body = _api_post("/predict/", {"symbol": symbol, "days": days})
    except Exception as e:
        st.error(f"Error llamando a /predict: {e}")
        return

    data = body.get("data", {})
    metrics = data.get("metrics", {})
    forecast = data.get("forecast", [])

    st.subheader(f"Resultado para {symbol}")
    if metrics:
        m_cols = st.columns(len(metrics))
        for (name, value), col in zip(metrics.items(), m_cols):
            col.metric(name.upper(), f"{value:.4f}")

    if forecast:
        df_f = pd.DataFrame(forecast)
        df_f["date"] = pd.to_datetime(df_f["date"])
        fig = px.line(df_f, x="date", y="value", markers=True, title=f"Forecast {symbol} próximos {days} días")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_f, use_container_width=True)
    else:
        st.warning("La API no devolvió datos de forecast.")



def page_recommend():
    st.title("💡 Recomendaciones por tendencia")

    with st.sidebar:
        st.header("Parámetros de recomendación")
        symbol = st.selectbox("Símbolo", SUPPORTED_SYMBOLS, index=0, key="rec_symbol")
        symbol = (symbol or "").upper()
        days = st.slider("Horizonte (días)", min_value=1, max_value=60, value=7, key="rec_days")
        run_btn = st.button("Obtener recomendación")

    if not run_btn:
        st.info("Configura los parámetros y pulsa 'Obtener recomendación'.")
        return

    try:
        body = _api_get("/recommendations/", params={"symbol": symbol, "days": days})
    except Exception as e:
        st.error(f"Error llamando a /recommendations: {e}")
        return

    data = body.get("data", {})

    st.subheader(f"Recomendación para {symbol}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Símbolo", data.get("symbol", "-"))
    col2.metric("Horizonte (días)", data.get("horizon_days", days))
    col3.metric("Tendencia", data.get("trend", "-"))

    st.success(data.get("recommendation", "Sin recomendación"))

    # Mostrar historial filtrado por símbolo
    try:
        body_hist = _api_get("/history/", params={"limit": 100})
        items: List[Dict[str, Any]] = body_hist.get("data", [])
        rec_rows = []
        for it in items:
            resp = it.get("response", {})
            if resp.get("symbol") == symbol and "recommendation" in resp:
                ts = it.get("timestamp")
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else ts
                except Exception:
                    ts_dt = ts
                rec_rows.append({
                    "timestamp": ts_dt,
                    "symbol": resp.get("symbol"),
                    "trend": resp.get("trend"),
                    "recommendation": resp.get("recommendation"),
                })
        if rec_rows:
            df_rec = pd.DataFrame(rec_rows).sort_values("timestamp", ascending=False)
            st.markdown("### Historial de recomendaciones para este símbolo")
            st.dataframe(df_rec, use_container_width=True)
        else:
            st.info("No hay recomendaciones históricas para este símbolo.")
    except Exception:
        pass


def page_eda():
    st.title("📉 Análisis de precios y volatilidad")

    # Cargar datos locales
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        # Normalize symbol column to uppercase and strip whitespace so all symbols appear
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de datos: {DATA_PATH}. Ejecuta los scripts de ETL.")
        return
    except Exception as e:
        st.error(f"Error leyendo {DATA_PATH}: {e}")
        return

    # Filtros
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    if not symbols:
        st.warning("El dataset no contiene símbolos.")
        return

    with st.sidebar:
        st.header("Filtros EDA")
        sym = st.selectbox("Símbolo", symbols, index=0, key="eda_symbol")
        date_min = df["date"].min().date()
        date_max = df["date"].max().date()
        start_date, end_date = st.date_input(
            "Rango de fechas",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

    mask = (df["symbol"] == sym) & (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    sub = df.loc[mask].sort_values("date")
    if sub.empty:
        st.warning("No hay datos para ese rango.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precio mínimo", f"{sub['price_usd'].min():.2f} USD")
    with col2:
        st.metric("Precio máximo", f"{sub['price_usd'].max():.2f} USD")
    with col3:
        st.metric("Observaciones", len(sub))

    # Gráfico de precio
    st.markdown("### Precio diario")
    fig_price = px.line(sub, x="date", y="price_usd", title=f"Precio {sym} ({start_date} a {end_date})")
    st.plotly_chart(fig_price, use_container_width=True)

    # Gráfico de volumen
    if "total_volume_usd" in sub.columns:
        st.markdown("### Volumen diario")
        fig_vol = px.bar(sub, x="date", y="total_volume_usd", title=f"Volumen {sym}")
        st.plotly_chart(fig_vol, use_container_width=True)

    # Retornos diarios (si existen)
    if "daily_return" in sub.columns and not sub["daily_return"].isna().all():
        st.markdown("### Histograma de retornos diarios")
        fig_ret = px.histogram(sub.dropna(subset=["daily_return"]), x="daily_return", nbins=50,
                               title=f"Distribución de retornos diarios {sym}")
        st.plotly_chart(fig_ret, use_container_width=True)

    # Volatilidad rolling 30d si existe
    if "roll_vol_30d" in sub.columns and not sub["roll_vol_30d"].isna().all():
        st.markdown("### Volatilidad rolling 30 días")
        fig_vola = px.line(sub, x="date", y="roll_vol_30d", title=f"Volatilidad 30d {sym}")
        st.plotly_chart(fig_vola, use_container_width=True)

    st.markdown("### Datos filtrados")
    st.dataframe(sub[["date", "price_usd", "total_volume_usd", "daily_return", "roll_vol_30d"]].fillna(""),
                 use_container_width=True)


PAGES = {
    "Visión general": page_overview,
    "Predicción": page_predict,
    "Recomendaciones": page_recommend,
    "Análisis de precios": page_eda,
}


def main():
    st.set_page_config(page_title="Crypto BI3 Dashboard", layout="wide")

    with st.sidebar:
        st.title("Crypto BI3")
        st.markdown("Selecciona la vista:")
        page_name = st.radio(
            "Selecciona la vista principal",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("API base:")
        st.code(API_BASE_URL, language="text")

    PAGES[page_name]()


if __name__ == "__main__":
    main()
