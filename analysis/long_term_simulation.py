import streamlit as st
import pandas as pd
import plotly.express as px
from models.predictor import get_stock_data
from models.advanced_lstm import train_advanced_lstm
from utils.formatter import format_currency_idr


def show_long_term_growth_simulation(portfolio_df):
    st.header("🚀 Simulasi Pertumbuhan Harga Jangka Panjang")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    duration = st.slider("⏳ Durasi Proyeksi (hari)", 30, 365, 180)
    st.info("Model menggunakan prediksi harga dari Advanced LSTM. Hasil bersifat estimatif.")

    results = []
    progress = st.progress(0, text="Memulai prediksi...")

    for idx, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        stock = row['Stock']
        shares = row['Shares']
        hist_data = get_stock_data(ticker)

        if hist_data is None or len(hist_data) < 90:
            continue

        pred_df, _, _ = train_advanced_lstm(hist_data, prediction_days=duration)
        if pred_df is not None:
            price_start = hist_data['price'].iloc[-1]
            price_end = pred_df['price'].iloc[-1]
            growth_pct = (price_end - price_start) / price_start * 100
            value_now = shares * price_start
            value_future = shares * price_end

            results.append({
                'Saham': stock,
                'Ticker': ticker,
                'Harga Sekarang': price_start,
                'Harga Prediksi': price_end,
                'Return (%)': growth_pct,
                'Nilai Sekarang': value_now,
                'Nilai Prediksi': value_future
            })

        progress.progress((idx + 1) / len(portfolio_df), text=f"Memproses {stock}...")

    progress.empty()

    if not results:
        st.error("Tidak ada saham yang bisa diprediksi karena data historis kurang.")
        return

    df = pd.DataFrame(results)

    with st.container():
        st.subheader("📊 Ringkasan Proyeksi Portofolio")
        col1, col2 = st.columns(2)
        col1.metric("💰 Nilai Saat Ini", format_currency_idr(df['Nilai Sekarang'].sum()))
        col2.metric("📈 Prediksi Nilai", format_currency_idr(df['Nilai Prediksi'].sum()),
                    delta=f"{((df['Nilai Prediksi'].sum() - df['Nilai Sekarang'].sum()) / df['Nilai Sekarang'].sum()) * 100:.2f}%")

    with st.expander("📈 Grafik Proyeksi Return Saham"):
        fig = px.bar(df, x='Saham', y='Return (%)', color='Return (%)', text_auto='.2f',
                     color_continuous_scale='RdYlGn', title="Return (%) per Saham")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabel Detail Proyeksi"):
        df_display = df.copy()
        df_display['Harga Sekarang'] = df_display['Harga Sekarang'].apply(format_currency_idr)
        df_display['Harga Prediksi'] = df_display['Harga Prediksi'].apply(format_currency_idr)
        df_display['Nilai Sekarang'] = df_display['Nilai Sekarang'].apply(format_currency_idr)
        df_display['Nilai Prediksi'] = df_display['Nilai Prediksi'].apply(format_currency_idr)
        st.dataframe(df_display.sort_values(by='Return (%)', ascending=False).reset_index(drop=True), use_container_width=True)
