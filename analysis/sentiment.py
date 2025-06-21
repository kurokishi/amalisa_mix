import streamlit as st
import random
import pandas as pd

# Simulasi data berita dan sentimen (karena API belum tersedia)
FAKE_NEWS = [
    "Perusahaan umumkan ekspansi ke luar negeri tahun ini",
    "Laba bersih naik 25% dibanding tahun lalu",
    "Analis menyarankan BUY setelah laporan keuangan positif",
    "Perusahaan alami penurunan pendapatan karena biaya energi",
    "Investor asing masuk ke sektor terkait",
    "Harga komoditas utama turun, pengaruhi margin",
    "Manajemen umumkan rencana buyback saham",
    "Perusahaan digugat karena isu lingkungan",
    "Sentimen positif dari regulasi pemerintah baru",
    "Kinerja kuartal lebih baik dari ekspektasi analis"
]

SENTIMENT_LABELS = ["Positif", "Netral", "Negatif"]


def generate_fake_sentiment():
    news_items = random.sample(FAKE_NEWS, 5)
    sentiments = [random.choices(SENTIMENT_LABELS, weights=[0.5, 0.3, 0.2])[0] for _ in news_items]
    return pd.DataFrame({"Berita": news_items, "Sentimen": sentiments})


def show_sentiment_analysis(portfolio_df):
    st.header("Analisis Sentimen Berita")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    selected_stock = st.selectbox("Pilih Saham untuk Analisis Sentimen", portfolio_df['Stock'])

    st.subheader(f"Berita Terkait: {selected_stock}")
    news_df = generate_fake_sentiment()
    st.dataframe(news_df.style.applymap(
        lambda x: 'background-color: lightgreen' if x == 'Positif' else (
                  'background-color: lightcoral' if x == 'Negatif' else 'background-color: lightyellow'),
        subset=['Sentimen']
    ))
