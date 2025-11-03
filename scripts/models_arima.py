from __future__ import annotations
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_arima(df: pd.DataFrame, date_col: str, target_col: str, order=(1,1,1), seasonal_order=(0,0,0,0), train_ratio: float = 0.8):
    s = df[[date_col, target_col]].dropna().sort_values(date_col).set_index(date_col)[target_col].asfreq('D').interpolate()
    split = int(len(s)*train_ratio)
    y_train, y_test = s.iloc[:split], s.iloc[split:]
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(y_test)).predicted_mean
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return res, y_train, y_test, preds, {'MAE': mae, 'RMSE': rmse}

def forecast_arima(fitted_res, steps: int = 30):
    return fitted_res.get_forecast(steps=steps).predicted_mean
