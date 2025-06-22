import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils.formatter import format_currency_idr


def simulate_dividend_reinvestment(ticker, shares, tahun):
    stock = yf.Ticker(ticker)
    try:
        df_price = stock.history(start=f"{datetime.now().year - tahun}-01-01")
        df_div = stock.dividends[df_price.index[0]:]
        if df_price.empty or df_div.empty:
            return None
    except:
        return None

    data = []
    total_shares = shares
    total_investasi = shares * df_price['Close'].iloc[0]

    for date, div_per_share in df_div.items():
        if date not in df_price.index:
            continue
        close_price = df_price.loc[date]['Close']
        div_total = div_per_share * total_shares
        tambahan_saham = div_total / close_price if close_price > 0 else 0
        total_shares += tambahan_saham
        nilai_portofolio = total_shares * close_price

        data.append({
            "Tanggal": date.date(),
            "Dividen/Share": round(div_per_share, 2),
            "Total Dividen": round(div_total, 2),
            "Harga Saham": round(close_price, 2),
            "Saham Tambahan": round(tambahan_saham, 4),
            "Total Saham": round(total_shares, 4),
            "Nilai Portofolio": round(nilai_portofolio, 2)
        })

    return pd.DataFrame(data)


def show_reinvest_dividen(portfolio_df):
    st.header("🔁 Simulasi Reinvestasi Dividen (DRIP)")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    selected_stock = st.selectbox("📌 Pilih Saham", portfolio_df['Stock'])
    row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
    ticker = row['Ticker']
    shares = row['Shares']
    tahun = st.slider("⏳ Periode Simulasi (tahun)", 1, 10, 5)

    df_simulasi = simulate_dividend_reinvestment(ticker, shares, tahun)
    if df_simulasi is None or df_simulasi.empty:
        st.error("Data dividen atau harga historis tidak tersedia.")
        return

    st.subheader("📈 Grafik Pertumbuhan Portofolio")
    fig = px.line(df_simulasi, x="Tanggal", y="Nilai Portofolio", markers=True,
                  title="Pertumbuhan Nilai Portofolio dari Reinvestasi Dividen")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Tabel Detail Transaksi Reinvestasi")
    df_display = df_simulasi.copy()
    for col in ["Total Dividen", "Harga Saham", "Nilai Portofolio"]:
        df_display[col] = df_display[col].apply(format_currency_idr)
    st.dataframe(df_display, use_container_width=True)

    st.subheader("📊 Ringkasan Akhir")
    akhir = df_simulasi.iloc[-1]
    nilai_awal = shares * df_simulasi['Harga Saham'].iloc[0]
    nilai_akhir = akhir['Nilai Portofolio']
    pertumbuhan = (nilai_akhir - nilai_awal) / nilai_awal * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Nilai Awal", format_currency_idr(nilai_awal))
    col2.metric("Nilai Akhir", format_currency_idr(nilai_akhir))
    col3.metric("Return (%)", f"{pertumbuhan:.2f}%")
