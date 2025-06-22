limport numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def train_advanced_lstm(df, prediction_days=30, epochs=20, dropout_rate=0.2):
    if len(df) < 90:
        st.warning("Data historis terlalu pendek untuk pelatihan LSTM lanjutan.")
        return None, None, None

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    # Buat urutan data
    X, y = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/Test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None

    # Build model
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # Evaluasi
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    # Prediksi masa depan
    inputs = scaled_data[-seq_len:]
    future = []
    for _ in range(prediction_days):
        x = inputs[-seq_len:].reshape(1, seq_len, 1)
        pred = model.predict(x, verbose=0)[0][0]
        future.append(pred)
        inputs = np.append(inputs, pred)

    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    last_date = df['Date'].iloc[-1]
    dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    df_forecast = pd.DataFrame({'Date': dates, 'price': future_prices})

    return df_forecast, mae, rmse
