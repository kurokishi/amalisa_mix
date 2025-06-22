import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from utils.formatter import format_currency_idr


def fetch_price_history(ticker, period="1y", interval="1mo"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        data = data.reset_index()[['Date', 'Close']]
        data = data.dropna()
        data.columns = ['Tanggal', 'Harga']
        return data
    except:
        return None


def show_dca_simulation(portfolio_df):
    st.header("📆 Simulasi Strategi DCA (Dollar Cost Averaging)")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    selected_stock = st.selectbox("📌 Pilih Saham", portfolio_df['Stock'])
    row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
    ticker = row['Ticker']

    dca_periode = st.radio("🔄 Frekuensi DCA", ["Bulanan", "Mingguan"], horizontal=True)
    dca_nominal = st.number_input("💸 Jumlah Investasi per Periode (Rp)", min_value=10000, step=10000, value=500000)
    duration = st.slider("⏳ Durasi Simulasi (bulan)", 3, 60, 12)

    interval = "1mo" if dca_periode == "Bulanan" else "1wk"
    period = f"{int(duration * 1.2)}mo"  # Ambil lebih panjang agar aman dari libur
    prices = fetch_price_history(ticker, period=period, interval=interval)

    if prices is None or len(prices) < duration:
        st.error("Data harga historis tidak cukup untuk simulasi. Coba dengan durasi lebih pendek atau saham lain.")
        return

    prices = prices.tail(duration)
    shares = []
    total_invested = []
    total_shares = 0
    total_cost = 0

    for _, rowp in prices.iterrows():
        price = rowp['Harga']
        if price == 0:
            shares.append(0)
            total_invested.append(total_cost)
            continue
        unit = dca_nominal / price
        total_shares += unit
        total_cost += dca_nominal
        shares.append(total_shares)
        total_invested.append(total_cost)

    prices['Total Saham'] = shares
    prices['Total Investasi'] = total_invested
    prices['Harga Rata-rata'] = prices['Total Investasi'] / prices['Total Saham']
    prices['Nilai Saham Sekarang'] = prices['Total Saham'] * prices['Harga']
    prices['Unrealized P/L'] = prices['Nilai Saham Sekarang'] - prices['Total Investasi']

    st.subheader("📌 Ringkasan Simulasi")
    last = prices.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Investasi", format_currency_idr(last['Total Investasi']))
    col2.metric("📈 Nilai Saham Sekarang", format_currency_idr(last['Nilai Saham Sekarang']))
    col3.metric("📊 Unrealized P/L", format_currency_idr(last['Unrealized P/L']),
                delta=f"{(last['Unrealized P/L'] / last['Total Investasi'] * 100):.2f}%")

    with st.expander("📈 Grafik Nilai Investasi vs Nilai Saham"):
        fig = px.line(prices, x='Tanggal', y=['Total Investasi', 'Nilai Saham Sekarang'], markers=True,
                     labels={"value": "Rupiah", "Tanggal": "Tanggal"})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabel Detail Periode DCA"):
        display = prices.copy()
        for col in ['Total Investasi', 'Nilai Saham Sekarang', 'Unrealized P/L']:
            display[col] = display[col].apply(format_currency_idr)
        st.dataframe(display.reset_index(drop=True), use_container_width=True)
