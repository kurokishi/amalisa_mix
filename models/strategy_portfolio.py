import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.formatter import format_currency_idr
from analysis.strategy_simulation import simulate_dca, simulate_reinvest_dividen


def show_portfolio_strategy_simulation(portfolio_df):
    st.header("📊 Simulasi Strategi Portofolio Kombinasi")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
    dca_per_saham = st.number_input("💸 Nominal DCA per Saham / Bulan (Rp)", min_value=10000, step=10000, value=500000)

    total_awal = 0
    total_dca = 0
    total_dca_drip = 0
    total_investasi_dca = 0

    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100  # 1 lot = 100 lembar saham
        start_date = (datetime.now() - timedelta(days=durasi_tahun * 365)).strftime("%Y-%m-%d")

        try:
            hist = yf.download(ticker, start=start_date, interval="1mo", progress=False)
            if hist.empty or 'Close' not in hist.columns or hist['Close'].isnull().all():
                continue

            prices = hist['Close'].dropna()
            harga_akhir = prices.iloc[-1]

            # Strategi 1: Tanpa strategi
            nilai_awal = saham_awal * harga_akhir
            total_awal += nilai_awal

            # Strategi 2: DCA
            saham_dca, total_invested_dca = simulate_dca(prices, dca_per_saham)
            nilai_dca = saham_dca * harga_akhir
            total_dca += nilai_dca
            total_investasi_dca += total_invested_dca

            # Strategi 3: DCA + DRIP
            saham_dca_drip = simulate_reinvest_dividen(ticker, saham_dca, hist)
            nilai_dca_drip = saham_dca_drip * harga_akhir
            total_dca_drip += nilai_dca_drip

        except Exception as e:
            st.warning(f"Gagal memproses {ticker}: {e}")
            continue

    def calculate_return(total_now, invested):
        return ((total_now - invested) / invested * 100) if invested else 0

    result = pd.DataFrame({
        "Strategi": ["Tanpa Strategi", "📆 DCA", "📆 DCA + 🔁 DRIP"],
        "Nilai Akhir": [total_awal, total_dca, total_dca_drip],
        "Return (%)": [
            calculate_return(total_awal, total_investasi_dca),
            calculate_return(total_dca, total_investasi_dca),
            calculate_return(total_dca_drip, total_investasi_dca)
        ]
    })

    result_display = result.copy()
    result_display["Nilai Akhir"] = result_display["Nilai Akhir"].apply(format_currency_idr)
    result_display["Return (%)"] = result_display["Return (%)"].map(lambda x: f"{x:.2f}%")

    st.subheader("📈 Hasil Simulasi Portofolio")
    st.dataframe(result_display, use_container_width=True)
    st.caption("Simulasi dilakukan untuk semua saham dengan data dari yfinance.")
