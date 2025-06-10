# modules/data_fetcher.py
import yfinance as yf
import pandas as pd
from typing import Tuple

def get_price_data(ticker: str, period="1y") -> pd.DataFrame:
    df = yf.download(ticker + '.JK', period=period)
    return df

def get_dividends(ticker: str) -> pd.Series:
    stock = yf.Ticker(ticker + '.JK')
    return stock.dividends
