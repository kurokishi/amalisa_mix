import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import traceback
from utils.formatter import format_currency_idr

def safe_yfinance_download(ticker, start_date, max_retries=3):
    for i in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, interval="1mo", progress=False)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"⚠️ Percobaan {i+1}/{max_retries} gagal: {str(e)}")
            time.sleep(2)  # Tunggu sebelum coba lagi
    return pd.DataFrame()

def simulate_dca(prices, dca_nominal):
    try:
        total_shares = 0
        total_invested = 0
        valid_prices = [p for p in prices if pd.notnull(p) and p > 0]
        
        if not valid_prices:
            return 0, 0
            
        for price in valid_prices:
            shares = dca_nominal / price
            total_shares += shares
            total_invested += dca_nominal
        return total_shares, total_invested
    except Exception:
        return 0, 0

def safe_dividend_calculation(ticker, base_shares, price_history):
    if base_shares <= 0:
        return base_shares

    try:
        # Batasi waktu eksekusi
        start_time = time.time()
        timeout = 20  # detik
        
        stock = yf.Ticker(ticker)
        
        # Dapatkan tanggal mulai dan akhir dengan aman
        if price_history.empty:
            return base_shares
            
        start_date = price_history.index[0]
        end_date = price_history.index[-1]
        
        # Dapatkan dividen dengan penanganan error
        try:
            dividends = stock.dividends
            if dividends is None:
                return base_shares
                
            dividends = dividends.loc[start_date:end_date]
        except Exception:
            return base_shares

        total_shares = base_shares
        for date, div_per_share in dividends.items():
            # Cek timeout
            if time.time() - start_time > timeout:
                st.warning("⏱️ Timeout saat menghitung dividen")
                return total_shares
                
            if date in price_history.index:
                try:
                    close_price = price_history.loc[date, 'Close']
                    if pd.isna(close_price) or close_price <= 0:
                        continue
                        
                    div_total = div_per_share * total_shares
                    additional_shares = div_total / close_price
                    total_shares += additional_shares
                except Exception:
                    continue
        return total_shares
    except Exception:
        return base_shares

def show_strategy_simulation(portfolio_df):
    try:
        st.header("🧪 Simulasi Strategi Terintegrasi per Saham")

        if portfolio_df is None or portfolio_df.empty:
            st.warning("Silakan upload portofolio terlebih dahulu.")
            return

        # Pastikan kolom yang diperlukan ada
        required_columns = ['Stock', 'Ticker']
        if not all(col in portfolio_df.columns for col in required_columns):
            st.error("Format portofolio tidak valid. Pastikan ada kolom 'Stock' dan 'Ticker'.")
            return

        selected_stock = st.selectbox("📌 Pilih Saham", portfolio_df['Stock'])
        row = portfolio_df[portfolio_df['Stock'] == selected_stock]
        
        if row.empty:
            st.error("Saham yang dipilih tidak ditemukan dalam portofolio")
            return
            
        row = row.iloc[0]
        ticker = row['Ticker'] + ".JK"  # Tambahkan .JK untuk saham Indonesia
        
        durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
        dca_nominal = st.number_input("💸 Nominal DCA / Bulan (Rp)", min_value=10000, step=10000, value=500000)
        saham_awal_input = st.number_input("📦 Jumlah Saham Awal yang Dimiliki", min_value=0.0, step=0.01, value=0.0)

        start_date = (datetime.now() - timedelta(days=durasi_tahun*365)).strftime("%Y-%m-%d")
        
        # Download data dengan spinner dan penanganan error
        with st.spinner("🔄 Mengunduh data historis..."):
            try:
                hist = safe_yfinance_download(ticker, start_date)
            except Exception as e:
                st.error(f"⛔ Gagal mengunduh data: {str(e)}")
                return

        # Pengecekan data historis
        if hist is None:
            st.error("⛔ Tidak ada data historis")
            return
            
        if hist.empty:
            st.error("⛔ Data historis kosong")
            return
            
        if 'Close' not in hist.columns:
            st.error("⛔ Kolom 'Close' tidak tersedia di data")
            return

        # Pastikan ada cukup data
        if len(hist) < 6:
            st.warning("⚠️ Data historis terbatas, hasil mungkin kurang akurat")

        # Tangani harga
        try:
            prices = hist['Close'].dropna()
            if prices.empty:
                st.error("⛔ Tidak ada data harga yang valid")
                return
                
            harga_awal = prices.iloc[0]
            harga_akhir = prices.iloc[-1]
            st.info(f"📉 Harga awal: {format_currency_idr(harga_awal)}, harga akhir: {format_currency_idr(harga_akhir)}")
        except Exception as e:
            st.error(f"⚠️ Gagal mendapatkan harga: {str(e)}")
            harga_awal = 0
            harga_akhir = 0

        # Gunakan saham user sebagai dasar semua strategi
        saham_awal = saham_awal_input

        # 1. Strategi: Tanpa Strategi
        nilai_akhir_baseline = saham_awal * harga_akhir
        investasi_awal_baseline = saham_awal * harga_awal if harga_awal > 0 else 0
        
        if investasi_awal_baseline > 0:
            return_baseline = (nilai_akhir_baseline - investasi_awal_baseline) / investasi_awal_baseline * 100
        else:
            return_baseline = 0.0

        # 2. Strategi: DCA
        with st.spinner("🔄 Menghitung strategi DCA..."):
            saham_dari_dca, total_invested_dca = simulate_dca(prices, dca_nominal)
            total_saham_dca = saham_awal + saham_dari_dca
            nilai_dca = total_saham_dca * harga_akhir
            total_investasi_dca = investasi_awal_baseline + total_invested_dca
            
            if total_investasi_dca > 0:
                return_dca = (nilai_dca - total_investasi_dca) / total_investasi_dca * 100
            else:
                return_dca = 0.0

        # 3. Strategi: DCA + DRIP
        with st.spinner("🔄 Menghitung strategi DRIP..."):
            total_saham_dca_drip = safe_dividend_calculation(ticker, total_saham_dca, hist)
            nilai_dca_drip = total_saham_dca_drip * harga_akhir
            
            if total_investasi_dca > 0:
                return_dca_drip = (nilai_dca_drip - total_investasi_dca) / total_investasi_dca * 100
            else:
                return_dca_drip = 0.0

        # Hasil akhir
        result = pd.DataFrame({
            "Strategi": ["Tanpa Strategi", "📆 DCA", "📆 DCA + 🔁 DRIP"],
            "Total Saham": [saham_awal, total_saham_dca, total_saham_dca_drip],
            "Nilai Akhir": [nilai_akhir_baseline, nilai_dca, nilai_dca_drip],
            "Return (%)": [return_baseline, return_dca, return_dca_drip]
        })

        st.subheader("📊 Hasil Simulasi Strategi")
        
        # Coba buat grafik, jika gagal tampilkan tabel saja
        try:
            fig = px.bar(result, x="Strategi", y="Nilai Akhir", text_auto='.2s', color="Strategi",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title="Perbandingan Nilai Akhir per Strategi", hover_data=["Return (%)"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("⚠️ Tidak dapat membuat grafik, menampilkan tabel saja")

        # Format hasil untuk ditampilkan
        result_display = result.copy()
        result_display['Nilai Akhir'] = result_display['Nilai Akhir'].apply(
            lambda x: format_currency_idr(x) if pd.notnull(x) else "N/A"
        )
        result_display['Total Saham'] = result_display['Total Saham'].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
        )
        result_display['Return (%)'] = result_display['Return (%)'].apply(
            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"
        )
        
        st.dataframe(result_display, use_container_width=True)
        st.caption("Simulasi ini menggunakan data historis dan dividen aktual dari yfinance.")

    except Exception as e:
        st.error(f"Terjadi kesalahan fatal: {str(e)}")
        st.text(traceback.format_exc())  # Tampilkan traceback untuk debugging
