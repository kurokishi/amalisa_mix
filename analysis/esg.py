import streamlit as st
import pandas as pd
import plotly.express as px
from data.fetcher import get_fundamental_data


def show_esg_analysis(portfolio_df):
    st.header("Analisis ESG dan Green Score")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    results = []
    for _, row in portfolio_df.iterrows():
        data = get_fundamental_data(row['Ticker'])
        if data:
            results.append({
                'Saham': row['Stock'],
                'Sektor': data.get('Sektor'),
                'ESG Score': data.get('ESG Score', 5.0),
                'Green Score': data.get('Green Score', 5.0)
            })

    if not results:
        st.info("Data ESG tidak tersedia.")
        return

    df = pd.DataFrame(results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Skor ESG per Saham")
        fig1 = px.bar(df, x='Saham', y='ESG Score', color='Sektor', title="ESG Score")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Skor Green Score per Saham")
        fig2 = px.bar(df, x='Saham', y='Green Score', color='Sektor', title="Green Score")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detail ESG")
    st.dataframe(df.sort_values(by='ESG Score', ascending=False))
