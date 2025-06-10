# app.py
import streamlit as st
import pandas as pd
from modules.pdf_parser import parse_portfolio_pdf

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📊 Stock Analyzer - Portofolio Ajaib")

uploaded_file = st.file_uploader("Upload file PDF portofolio Ajaib", type="pdf")
if uploaded_file:
    df = parse_portfolio_pdf(uploaded_file)
    st.success("📄 Data berhasil diparsing!")
    st.dataframe(df)

    if st.button("Simpan ke CSV"):
        df.to_csv("data/portfolio.csv", index=False)
        st.success("✅ Data disimpan ke data/portfolio.csv")

from modules.analyzer_fundamental import analisa_saham

elif menu == "📈 Analisa Fundamental":
    st.subheader("Analisa Fundamental per Emiten")

    if 'df' in locals():
        for kode in df['Kode Saham']:
            with st.expander(f"Analisa {kode}"):
                hasil = analisa_saham(kode)
                if hasil:
                    st.write(pd.DataFrame(hasil, index=[0]).T.rename(columns={0: 'Nilai'}))
                else:
                    st.warning(f"Gagal mengambil data untuk {kode}")
    else:
        st.info("Upload file PDF portofolio dulu di tab awal.")
