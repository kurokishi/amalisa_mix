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

    # Hitung tanggal mulai
    start_date = (datetime.now() - timedelta(days=durasi_tahun * 365)).strftime("%Y-%m-%d")
    
    # Unduh data untuk semua ticker sekaligus
    tickers = portfolio_df['Ticker'].tolist()
    try:
        # Unduh data dengan auto_adjust=True untuk menghindari warning
        hist_data = yf.download(tickers, start=start_date, interval="1mo", progress=False, auto_adjust=True, group_by='ticker')
    except Exception as e:
        st.error(f"Gagal mengunduh data: {e}")
        return

    total_awal = 0
    total_nilai_awal = 0  # Untuk perhitungan return tanpa strategi
    total_dca = 0
    total_dca_drip = 0
    total_investasi_dca = 0

    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100  # 1 lot = 100 lembar saham

        try:
            # Ambil data untuk ticker spesifik
            if ticker in hist_data:
                ticker_data = hist_data[ticker]
            else:
                # Coba format ticker dengan .JK untuk IDX
                ticker_jk = f"{ticker}.JK"
                ticker_data = hist_data[ticker_jk] if ticker_jk in hist_data else None
            
            if ticker_data is None or ticker_data.empty or 'Close' not in ticker_data.columns:
                st.warning(f"Data tidak tersedia untuk {ticker}")
                continue

            prices = ticker_data['Close'].dropna()
            if len(prices) < 2:
                st.warning(f"Data historis tidak cukup untuk {ticker}")
                continue

            harga_awal = prices.iloc[0]
            harga_akhir = prices.iloc[-1]

            # 1. Strategi: Tanpa strategi (hold saja)
            nilai_awal = saham_awal * harga_awal
            nilai_akhir_tanpa_strategi = saham_awal * harga_akhir
            total_awal += nilai_akhir_tanpa_strategi
            total_nilai_awal += nilai_awal

            # 2. Strategi: DCA
            saham_dca, total_invested_dca = simulate_dca(prices, dca_per_saham)
            nilai_dca = saham_dca * harga_akhir
            total_dca += nilai_dca
            total_investasi_dca += total_invested_dca

            # 3. Strategi: DCA + DRIP
            saham_dca_drip = simulate_reinvest_dividen(ticker, saham_dca, prices.index)
            nilai_dca_drip = saham_dca_drip * harga_akhir
            total_dca_drip += nilai_dca_drip

        except Exception as e:
            st.warning(f"Gagal memproses {ticker}: {e}")
            continue

    # Fungsi perhitungan return
    def calculate_return(nilai_akhir, nilai_awal):
        return ((nilai_akhir - nilai_awal) / nilai_awal * 100) if nilai_awal else 0

    # Buat DataFrame hasil
    result = pd.DataFrame({
        "Strategi": ["Tanpa Strategi", "📆 DCA", "📆 DCA + 🔁 DRIP"],
        "Nilai Akhir": [total_awal, total_dca, total_dca_drip],
        "Total Investasi": [
            total_nilai_awal,
            total_investasi_dca,
            total_investasi_dca
        ],
        "Return (%)": [
            calculate_return(total_awal, total_nilai_awal),
            calculate_return(total_dca, total_investasi_dca),
            calculate_return(total_dca_drip, total_investasi_dca)
        ]
    })

    # Format tampilan
    result_display = result.copy()
    result_display["Nilai Akhir"] = result_display["Nilai Akhir"].apply(format_currency_idr)
    result_display["Total Investasi"] = result_display["Total Investasi"].apply(format_currency_idr)
    result_display["Return (%)"] = result_display["Return (%)"].map(lambda x: f"{x:.2f}%")

    st.subheader("📈 Hasil Simulasi Portofolio")
    st.dataframe(result_display, use_container_width=True)
    st.caption("Simulasi dilakukan untuk semua saham dengan data dari yfinance.")
