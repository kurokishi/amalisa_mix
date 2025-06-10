# modules/analyzer_teknikal.py
import yfinance as yf
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

def fetch_history(ticker: str, period="1y"):
    df = yf.download(ticker + ".JK", period=period)
    df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return df

def predict_prophet(df: pd.DataFrame, periods: int = 90):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def predict_arima(df: pd.DataFrame, periods: int = 30):
    data = df['y'].values
    model = ARIMA(data, order=(5,1,0))  # Simple ARIMA model
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({'ds': future_dates, 'yhat': forecast})
