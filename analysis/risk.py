import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np


def show_risk_analysis(portfolio_df):
    st.header("Analisis Risiko Portofolio")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    st.subheader("Volatilitas Historis")
    returns = {}
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        data = yf.Ticker(ticker).history(period='6mo')
        data['Return'] = data['Close'].pct_change()
        returns[ticker] = data['Return'].dropna()

    df_returns = pd.DataFrame(returns)
    volatilities = df_returns.std() * np.sqrt(252)
    st.dataframe(volatilities.rename("Volatilitas (tahunan)"))

    st.subheader("Korelasi Saham")
    corr = df_returns.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Value at Risk (VaR) 95% Confidence")

    var_results = []
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        if ticker in df_returns.columns:
            percentile_loss = np.percentile(df_returns[ticker], 5)
            var_nominal = percentile_loss * row['Shares'] * row['Avg Price']
            var_results.append({'Saham': row['Stock'], 'Ticker': ticker, 'VaR (95%)': round(var_nominal, 2)})

    df_var = pd.DataFrame(var_results)
    st.write("(Simulasi kasar berdasarkan distribusi historis)")
    st.dataframe(df_var)

