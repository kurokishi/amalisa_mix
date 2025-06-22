import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from data.fetcher import get_fundamental_data
from utils.formatter import format_currency_idr


def show_rebalancing_recommendation(portfolio_df):
    st.header("⚖️ Rekomendasi Rebalancing Portofolio")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    st.markdown("""
    Tujuan dari rebalancing adalah menjaga diversifikasi sektor portofolio Anda agar tetap proporsional.
    Berikut ini adalah analisis alokasi saat ini dibandingkan dengan target ideal (rata-rata sektor).
    """)

    with st.status("📊 Menghitung komposisi sektor..."):
        sektor_map = {}
        sektor_value = {}
        total_value = 0
        sektor_detail = {}

        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']
            try:
                price_now = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
            except:
                continue
            total = shares * price_now

            fdata = get_fundamental_data(ticker)
            sektor = fdata.get('Sektor', 'Lainnya') if fdata else 'Lainnya'

            sektor_map[ticker] = sektor
            sektor_value[sektor] = sektor_value.get(sektor, 0) + total
            sektor_detail[ticker] = {'nilai': total, 'sektor': sektor, 'saham': row['Stock']}
            total_value += total

    if not sektor_value:
        st.error("Tidak ada data sektor yang tersedia untuk dihitung.")
        return

    st.subheader("📌 Komposisi Sektor Saat Ini")
    sektor_actual_df = pd.DataFrame([
        {'Sektor': s, 'Alokasi Saat Ini (%)': v / total_value * 100} for s, v in sektor_value.items()
    ])
    fig = px.pie(sektor_actual_df, names='Sektor', values='Alokasi Saat Ini (%)', hole=0.45,
                 title="Distribusi Sektor Portofolio", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Perbandingan Alokasi: Aktual vs Target")
    target_alloc = {s: 100 / len(sektor_value) for s in sektor_value.keys()}
    comp_df = sektor_actual_df.copy()
    comp_df['Target (%)'] = comp_df['Sektor'].map(target_alloc)
    comp_df['Selisih (%)'] = comp_df['Alokasi Saat Ini (%)'] - comp_df['Target (%)']
    st.dataframe(comp_df.round(2), use_container_width=True)

    st.subheader("🛠️ Saran Rebalancing")
    saran = []
    for ticker in portfolio_df['Ticker']:
        info = sektor_detail.get(ticker)
        if not info:
            continue
        sektor = info['sektor']
        nilai = info['nilai']
        ideal = total_value * target_alloc[sektor] / 100
        delta = nilai - ideal
        if abs(delta) / total_value > 0.01:
            saran.append({
                'Saham': info['saham'],
                'Sektor': sektor,
                'Tindakan': '📉 Kurangi' if delta > 0 else '📈 Tambah',
                'Selisih Nilai': format_currency_idr(abs(delta))
            })

    if saran:
        df_saran = pd.DataFrame(saran)
        st.dataframe(df_saran, use_container_width=True)
    else:
        st.success("✅ Portofolio Anda sudah seimbang berdasarkan sektor!")
