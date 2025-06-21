import yfinance as yf
import streamlit as st

@st.cache_data

def get_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        sector_esg_scores = {
            "Technology": 7.5,
            "Healthcare": 8.0,
            "Financial Services": 6.0,
            "Consumer Defensive": 7.0,
            "Energy": 4.5,
            "Basic Materials": 5.0,
            "Industrials": 6.5,
            "Communication Services": 7.0
        }

        green_scores = {
            "Technology": 8.0,
            "Healthcare": 7.5,
            "Financial Services": 5.0,
            "Consumer Defensive": 6.0,
            "Energy": 3.0,
            "Basic Materials": 4.0,
            "Industrials": 5.5,
            "Communication Services": 7.0
        }

        sektor = info.get('sector', 'N/A')
        esg_score = sector_esg_scores.get(sektor, 5.0)
        green_score = green_scores.get(sektor, 5.0)

        return {
            'PER': info.get('trailingPE'),
            'PBV': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'EPS': info.get('trailingEps'),
            'DER': info.get('debtToEquity'),
            'Dividend Yield': info.get('dividendYield'),
            'Market Cap': info.get('marketCap'),
            'Sektor': sektor,
            'Industri': info.get('industry', 'N/A'),
            'ESG Score': esg_score,
            'Green Score': green_score
        }
    except:
        return None
