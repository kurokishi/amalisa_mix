import streamlit as st
import pandas as pd
import plotly.express as px
from utils.formatter import format_currency_idr


def show_compound_projection(portfolio_df):
    st.header("Simulasi Pertumbuhan Bunga Majemuk")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    initial_value = portfolio_df['Shares'] * portfolio_df['Avg Price']
    total_initial = initial_value.sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        year_range = st.slider("Jangka Waktu (tahun)", 1, 30, 10)
    with col2:
        avg_return = st.slider("Estimasi Return Tahunan (%)", 5.0, 25.0, 12.0)
    with col3:
        reinvest_div = st.checkbox("Reinvest Dividen?", value=True)

    # Proyeksi bunga majemuk
    years = list(range(1, year_range + 1))
    growth = []
    value = total_initial
    for y in years:
        div_bonus = 0.02 if reinvest_div else 0.0
        rate = (avg_return / 100) + div_bonus
        value *= (1 + rate)
        growth.append(value)

    df = pd.DataFrame({'Tahun ke-': years, 'Proyeksi Nilai (Rp)': growth})

    st.subheader("Hasil Simulasi")
    final_value = growth[-1]
    st.metric("Nilai Akhir Portofolio", format_currency_idr(final_value),
              delta=f"{(final_value - total_initial) / total_initial * 100:.2f}%")

    fig = px.line(df, x='Tahun ke-', y='Proyeksi Nilai (Rp)', markers=True)
    fig.update_layout(yaxis_tickprefix='Rp')
    st.plotly_chart(fig, use_container_width=True)
