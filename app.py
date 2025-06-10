# app.py
import streamlit as st
import pandas as pd
from modules.pdf_parser import parse_portfolio_pdf
from modules.analyzer_fundamental import analisa_saham
from modules.analyzer_teknikal import fetch_history, predict_prophet, predict_arima
from modules.diversification import analisa_diversifikasi
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer - Portofolio Ajaib")

# Navigasi
menu = st.sidebar.radio("Navigasi", [
    "📁 Upload Portofolio PDF",
    "📈 Analisa Fundamental",
    "📊 Prediksi Harga Saham",
    "🔁 Diversifikasi & Rekomendasi"
])

if menu == "📁 Upload Portofolio PDF":
    uploaded_file = st.file_uploader("Upload file PDF portofolio Ajaib", type="pdf")
    if uploaded_file:
        df = parse_portfolio_pdf(uploaded_file)
        st.success("📄 Data berhasil diparsing!")
        st.dataframe(df)
       
        import os
        if st.button("Simpan ke CSV"):
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/portfolio.csv", index=False)
            st.success("✅ Data disimpan ke data/portfolio.csv")

elif menu == "📈 Analisa Fundamental":
    st.subheader("Analisa Fundamental Saham")
    try:
        df = pd.read_csv("data/portfolio.csv")
        for kode in df['Kode Saham']:
            with st.expander(f"Analisa {kode}"):
                hasil = analisa_saham(kode)
                if hasil:
                    st.write(pd.DataFrame(hasil, index=[0]).T.rename(columns={0: 'Nilai'}))
                else:
                    st.warning(f"❌ Gagal mengambil data untuk {kode}")
    except FileNotFoundError:
        st.warning("⚠️ File CSV belum tersedia. Upload dulu PDF portofolio.")

elif menu == "📊 Prediksi Harga Saham":
    st.subheader("Prediksi Harga Saham dengan Prophet & ARIMA")
    try:
        df = pd.read_csv("data/portfolio.csv")
        pilihan = st.selectbox("Pilih saham", df['Kode Saham'].unique())
        if pilihan:
            data_hist = fetch_history(pilihan)
            st.line_chart(data_hist.set_index("ds")["y"])

            st.markdown("### 🔮 Prediksi Harga (Prophet - 90 hari)")
            prophet_result = predict_prophet(data_hist)
            st.line_chart(prophet_result.set_index("ds")["yhat"])

            st.markdown("### 🔮 Prediksi Harga (ARIMA - 30 hari)")
            arima_result = predict_arima(data_hist)
            st.line_chart(arima_result.set_index("ds")["yhat"])
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat data harga: {e}")

elif menu == "🔁 Diversifikasi & Rekomendasi":
    st.subheader("Analisa Diversifikasi Portofolio")

    try:
        df = pd.read_csv("data/portfolio.csv")
        hasil = analisa_diversifikasi(df)
        st.dataframe(hasil.style.format({"Bobot Portofolio (%)": "{:.2f}"}))

        st.markdown("📊 **Visualisasi Bobot Portofolio**")
        st.bar_chart(hasil.set_index("Kode Saham")["Bobot Portofolio (%)"])
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat data portofolio: {e}")

