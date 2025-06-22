import streamlit as st
import pandas as pd
import random


def generate_fake_sentiment():
    return random.choice(['Positif', 'Netral', 'Negatif'])

def generate_fake_dividend_yield():
    return random.uniform(0, 10)  # dalam %

def generate_fake_risk_level():
    return random.choice(['Low', 'Medium', 'High'])

def generate_fake_growth():
    return random.uniform(-10, 30)  # dalam %


def show_strategy_recommendation(portfolio_df):
    st.header("🧠 Auto Strategy Recommendation")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    rekomendasi = []
    for _, row in portfolio_df.iterrows():
        saham = row['Stock']
        ticker = row['Ticker']

        sentiment = generate_fake_sentiment()
        div_yield = generate_fake_dividend_yield()
        risk = generate_fake_risk_level()
        pred_growth = generate_fake_growth()

        strategi = []
        if div_yield >= 4:
            strategi.append("🔁 Reinvest Dividen")
        if pred_growth > 10 and risk != 'High':
            strategi.append("📈 Long-Term Hold")
        if risk == 'Medium' and sentiment == 'Positif':
            strategi.append("📆 DCA Disarankan")
        if sentiment == 'Negatif' or pred_growth < 0:
            strategi.append("🚫 Tinjau Ulang Kepemilikan")

        if not strategi:
            strategi.append("❔ Belum ada strategi spesifik")

        rekomendasi.append({
            'Saham': saham,
            'Ticker': ticker,
            'Sentimen': sentiment,
            'Dividend Yield (%)': f"{div_yield:.2f}",
            'Prediksi Pertumbuhan (%)': f"{pred_growth:.2f}",
            'Risiko': risk,
            'Rekomendasi Strategi': ", ".join(strategi)
        })

    df_rek = pd.DataFrame(rekomendasi)

    st.subheader("📋 Rekomendasi per Saham")
    st.dataframe(df_rek, use_container_width=True)

    st.info("*Data dividen, prediksi, sentimen dan risiko pada tahap ini masih bersifat simulatif*")
