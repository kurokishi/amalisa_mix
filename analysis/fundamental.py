import streamlit as st
import pandas as pd
from data.fetcher import get_fundamental_data
from utils.formatter import format_currency_idr


def show_fundamental_analysis(portfolio_df):
    st.header("Analisis Fundamental Saham")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    fundamentals = []
    for _, row in portfolio_df.iterrows():
        data = get_fundamental_data(row['Ticker'])
        if data:
            fundamentals.append({
                'Saham': row['Stock'],
                'Kode': row['Ticker'],
                'PER': data.get('PER'),
                'PBV': data.get('PBV'),
                'ROE (%)': round(data.get('ROE', 0)*100, 2) if data.get('ROE') else None,
                'EPS': data.get('EPS'),
                'Div. Yield (%)': round(data.get('Dividend Yield', 0)*100, 2) if data.get('Dividend Yield') else None,
                'Market Cap': format_currency_idr(data.get('Market Cap')),
                'Sektor': data.get('Sektor')
            })

    if not fundamentals:
        st.info("Data fundamental tidak tersedia untuk saham dalam portofolio.")
        return

    df = pd.DataFrame(fundamentals)
    st.dataframe(df.sort_values(by='ROE (%)', ascending=False).reset_index(drop=True))