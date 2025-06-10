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
