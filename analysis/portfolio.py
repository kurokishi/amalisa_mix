import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from utils.formatter import format_currency_idr

def visualize_portfolio(portfolio):
    """
    Menampilkan ringkasan portofolio saham, metrik, dan visualisasi.
    """
    if portfolio is None or portfolio.empty:
        st.warning("Portofolio kosong atau belum dimuat.")
        return

    # Ambil harga terkini
    portfolio['Current Price'] = portfolio['Ticker'].apply(
        lambda x: yf.Ticker(x).history(period='1d')['Close'].iloc[-1] if not pd.isna(x) else 0
    )

    # Hitung nilai dan profit/loss
    portfolio['Value'] = portfolio['Shares'] * portfolio['Current Price']
    portfolio['Investment'] = portfolio['Shares'] * portfolio['Avg Price']
    portfolio['P/L'] = portfolio['Value'] - portfolio['Investment']
    portfolio['P/L %'] = (portfolio['P/L'] / portfolio['Investment']) * 100

    total_value = portfolio['Value'].sum()
    total_investment = portfolio['Investment'].sum()
    total_pl = total_value - total_investment
    total_pl_pct = (total_pl / total_investment) * 100 if total_investment > 0 else 0

    # Tampilkan metrik utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", format_currency_idr(total_value))
    col2.metric("Total Investment", format_currency_idr(total_investment))
    col3.metric("Total P/L", format_currency_idr(total_pl), f"{total_pl_pct:.2f}%")
    col4.metric("Jumlah Saham", len(portfolio))

    # Grafik komposisi portofolio
    st.subheader("Komposisi Portofolio")
    fig = px.pie(portfolio, values='Value', names='Stock', hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Grafik performa per saham
    st.subheader("Performa Saham (P/L %)")
    portfolio_sorted = portfolio.sort_values('P/L %', ascending=False)
    fig2 = px.bar(
        portfolio_sorted, x='Stock', y='P/L %', color='P/L %',
        color_continuous_scale='RdYlGn',
        text=portfolio_sorted['P/L %'].apply(lambda x: f"{x:.2f}%")
    )
    fig2.update_layout(xaxis_title='Saham', yaxis_title='Profit / Loss (%)')
    st.plotly_chart(fig2, use_container_width=True)

    # Tabel portofolio dengan format rupiah
    st.subheader("Detail Portofolio")
    portfolio_display = portfolio.copy()
    money_cols = ['Current Price', 'Avg Price', 'Value', 'Investment', 'P/L']
    for col in money_cols:
        portfolio_display[col] = portfolio_display[col].apply(format_currency_idr)

    st.dataframe(portfolio_display.sort_values('P/L %', ascending=False).reset_index(drop=True))
