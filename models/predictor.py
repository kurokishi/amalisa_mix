import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import plotly.graph_objects as go
import yfinance as yf

# Fungsi ambil data harga
@st.cache_data

def get_stock_data(ticker, period='5y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        hist = hist.reset_index()[['Date', 'Close']]
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        return hist.rename(columns={'Close': 'price'})
    except:
        return None

# Prophet
@st.cache_data

def prophet_prediction(df, days):
    df_prophet = df.rename(columns={'Date': 'ds', 'price': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'price'})

# LSTM

def lstm_prediction(df, days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1,1))
    x_train, y_train = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled_data)):
        x_train.append(scaled_data[i-seq_len:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train = np.array(x_train).reshape(-1, seq_len, 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    inputs = scaled_data[-seq_len:]
    future = []
    for _ in range(days):
        x = inputs[-seq_len:].reshape(1, seq_len, 1)
        pred = model.predict(x, verbose=0)[0][0]
        future.append(pred)
        inputs = np.append(inputs, pred)
    future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1)).flatten()
    last_date = df['Date'].iloc[-1]
    dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Date': dates, 'price': future_prices})

# XGBoost

def xgboost_prediction(df, days):
    df = df.copy()
    df.set_index('Date', inplace=True)
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    for i in range(1, 61):
        df[f'lag_{i}'] = df['price'].shift(i)
    df.dropna(inplace=True)

    X = df.drop('price', axis=1)
    y = df['price']
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)

    last_date = df.index[-1]
    inputs = df.tail(60)['price'].values
    future = []
    for i in range(days):
        date = last_date + timedelta(days=i+1)
        row = {
            'day': date.day, 'month': date.month, 'year': date.year
        }
        for j in range(1, 61):
            idx = -j if i - j < 0 else -(j - i)
            row[f'lag_{j}'] = inputs[idx] if idx < 0 else future[idx]
        X_pred = pd.DataFrame([row])
        pred = model.predict(X_pred)[0]
        future.append(pred)
    dates = [last_date + timedelta(days=i+1) for i in range(days)]
    return pd.DataFrame({'Date': dates, 'price': future})

# UI tampilan utama

def show_price_prediction(portfolio_df, prediction_days):
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    st.header("Prediksi Harga Saham")
    stock = st.selectbox("Pilih Saham", portfolio_df['Stock'])
    row = portfolio_df[portfolio_df['Stock'] == stock].iloc[0]
    ticker = row['Ticker']

    hist_data = get_stock_data(ticker)
    if hist_data is None:
        st.error(f"Gagal mengambil data untuk {ticker}")
        return

    st.line_chart(hist_data.set_index('Date'))

    model_option = st.selectbox("Pilih Model", ["Prophet", "LSTM", "XGBoost", "Ensemble"])
    results = {}
    if model_option in ["Prophet", "Ensemble"]:
        results['Prophet'] = prophet_prediction(hist_data, prediction_days)
    if model_option in ["LSTM", "Ensemble"]:
        results['LSTM'] = lstm_prediction(hist_data, prediction_days)
    if model_option in ["XGBoost", "Ensemble"]:
        results['XGBoost'] = xgboost_prediction(hist_data, prediction_days)

    st.subheader("Hasil Prediksi")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['price'], mode='lines', name='Historis'))
    for name, df_pred in results.items():
        if df_pred is not None:
            fig.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['price'], mode='lines', name=f"{name}"))
    fig.update_layout(title=f"Prediksi Harga Saham {stock} ({ticker})", xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)
