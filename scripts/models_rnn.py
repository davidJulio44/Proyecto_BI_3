from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

def make_supervised(series: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    X = np.asarray(X)[..., np.newaxis]
    y = np.asarray(y)
    return X, y

def train_rnn(df: pd.DataFrame, date_col: str, target_col: str, model_type: str = 'LSTM', lookback: int = 30, train_ratio: float = 0.8, epochs: int = 50, batch_size: int = 32):
    s = df[[date_col, target_col]].dropna().sort_values(date_col).set_index(date_col)[target_col].astype(float)
    scaler = MinMaxScaler()
    s_scaled = scaler.fit_transform(s.values.reshape(-1,1)).ravel()
    split = int(len(s_scaled)*train_ratio)
    train, test = s_scaled[:split], s_scaled[split:]
    # ensure supervised split continuity
    import numpy as np
    X_train, y_train = make_supervised(train, lookback)
    X_test, y_test = make_supervised(np.concatenate([train[-lookback:], test]), lookback)
    model = Sequential()
    if model_type.upper() == 'GRU':
        model.add(GRU(64, input_shape=(lookback,1)))
    else:
        model.add(LSTM(64, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    preds = model.predict(X_test, verbose=0).ravel()
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1)).ravel()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    return model, scaler, {'MAE': mae, 'RMSE': rmse}
