import streamlit as st
import pandas as pd
from analysis.portfolio import visualize_portfolio


def show_what_if_simulation(portfolio_df):
    st.header("Simulasi What-If")

    tab1, tab2 = st.tabs(["Tambahkan Saham Baru", "Simulasi Average Down"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portofolio Saat Ini")
            visualize_portfolio(portfolio_df)

        with col2:
            st.subheader("Tambah Investasi Baru")
            new_stock = st.text_input("Nama Saham")
            new_ticker = st.text_input("Kode Ticker")
            new_lots = st.number_input("Jumlah Lot", min_value=1, value=10)
            new_price = st.number_input("Harga per Saham (Rp)", min_value=10.0, value=100.0)

            if st.button("Simulasikan"):
                new_row = {
                    'Stock': new_stock,
                    'Ticker': new_ticker,
                    'Lot Balance': new_lots,
                    'Avg Price': new_price,
                    'Shares': new_lots * 100
                }
                new_df = portfolio_df.copy() if portfolio_df is not None else pd.DataFrame(columns=new_row.keys())
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                visualize_portfolio(new_df)

    with tab2:
        st.subheader("Simulasi Average Down (Sederhana)")

        if portfolio_df is None or portfolio_df.empty:
            st.warning("Silakan upload portofolio terlebih dahulu")
            return

        selected_stock = st.selectbox("Pilih Saham", portfolio_df['Stock'])
        row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
        ticker = row['Ticker']

        current_price = row['Avg Price'] * 0.8  # Simulasi penurunan 20%
        st.metric("Harga Saat Ini (simulasi)", f"Rp{current_price:,.0f}".replace(",", "."))

        additional_lots = st.slider("Berapa Lot akan Ditambah?", 1, 50, 10)
        if st.button("Simulasikan Average Down"):
            total_shares = row['Shares'] + additional_lots * 100
            new_avg_price = ((row['Shares'] * row['Avg Price']) + (additional_lots * 100 * current_price)) / total_shares

            st.success("Hasil Simulasi:")
            st.metric("Harga Rata-Rata Baru", f"Rp{new_avg_price:,.0f}".replace(",", "."))
            st.metric("Jumlah Saham Baru", f"{total_shares:,}")

            new_df = portfolio_df.copy()
            mask = new_df['Stock'] == selected_stock
            new_df.loc[mask, 'Avg Price'] = new_avg_price
            new_df.loc[mask, 'Lot Balance'] += additional_lots
            new_df.loc[mask, 'Shares'] = total_shares
            visualize_portfolio(new_df)
