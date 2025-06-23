import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.formatter import format_currency_idr
from dateutil.relativedelta import relativedelta
import traceback

# Asumsi biaya dan inflasi
ASUMSI_BIAYA_TRANSAKSI_PERSEN = 0.15  # 0.15% biaya transaksi
ASUMSI_INFLASI_TAHUNAN = 3.5  # 3.5% per tahun

def simulate_dca(prices, monthly_investment, start_date=None, end_date=None):
    """
    Simulasi DCA dengan penyesuaian tanggal dan biaya transaksi
    """
    if start_date is None:
        start_date = prices.index[0]
    if end_date is None:
        end_date = prices.index[-1]
    
    # Generate tanggal pembelian bulanan
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    total_shares = 0
    total_invested = 0
    
    for date in dates:
        if date < prices.index[0] or date > prices.index[-1]:
            continue
            
        # Cari harga terdekat
        price_date = prices.index[prices.index.searchsorted(date)]
        price = prices.loc[price_date]
        
        # Hitung biaya transaksi
        biaya_transaksi = monthly_investment * (ASUMSI_BIAYA_TRANSAKSI_PERSEN / 100)
        net_investment = monthly_investment - biaya_transaksi
        
        # Beli saham
        shares_bought = net_investment / price
        total_shares += shares_bought
        total_invested += monthly_investment
    
    return total_shares, total_invested

def get_dividend_data(ticker, start_date, end_date):
    """
    Ambil data dividen dan tanggal ex-dividend
    """
    try:
        # Format ticker yang benar untuk yfinance
        if ticker.endswith('.JK'):
            clean_ticker = ticker
        else:
            clean_ticker = ticker + '.JK'
            
        stock = yf.Ticker(clean_ticker)
        div_data = stock.dividends
        
        if div_data.empty:
            return pd.Series(dtype=float)
        
        # Filter berdasarkan rentang tanggal simulasi
        div_data = div_data[(div_data.index >= start_date) & (div_data.index <= end_date)]
        return div_data
    except Exception as e:
        st.warning(f"Gagal mengambil dividen {ticker}: {str(e)}")
        return pd.Series(dtype=float)

def simulate_reinvest_dividen(ticker, shares, prices, start_date, end_date):
    """
    Simulasi reinvestment dividen dengan tanggal ex-dividend yang akurat
    """
    # Dapatkan data dividen
    dividends = get_dividend_data(ticker, start_date, end_date)
    
    if dividends.empty:
        return shares
    
    total_shares = shares
    
    for ex_date, amount in dividends.items():
        # Cari harga pada tanggal ex-dividend
        if ex_date in prices.index:
            price = prices.loc[ex_date]
        else:
            # Cari harga terdekat setelah ex-date
            next_date = prices.index[prices.index.searchsorted(ex_date)]
            price = prices.loc[next_date]
        
        # Hitung dividen yang diterima
        dividend_received = total_shares * amount
        
        # Hitung biaya transaksi
        biaya_transaksi = dividend_received * (ASUMSI_BIAYA_TRANSAKSI_PERSEN / 100)
        net_dividend = dividend_received - biaya_transaksi
        
        # Beli tambahan saham
        shares_bought = net_dividend / price
        total_shares += shares_bought
    
    return total_shares

def adjust_for_inflation(amount, start_year, end_year):
    """
    Sesuaikan nilai dengan inflasi
    """
    years = end_year - start_year
    inflation_factor = (1 + ASUMSI_INFLASI_TAHUNAN / 100) ** years
    return amount / inflation_factor

def show_portfolio_strategy_simulation(portfolio_df):
    st.header("📊 Simulasi Strategi Portofolio Kombinasi (Akurasi Tinggi)")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    with st.expander("⚙️ Pengaturan Lanjutan"):
        col1, col2 = st.columns(2)
        with col1:
            global ASUMSI_BIAYA_TRANSAKSI_PERSEN
            ASUMSI_BIAYA_TRANSAKSI_PERSEN = st.number_input("Biaya Transaksi (%)", 
                                            min_value=0.0, 
                                            max_value=1.0, 
                                            value=0.15, 
                                            step=0.05)
        with col2:
            global ASUMSI_INFLASI_TAHUNAN
            ASUMSI_INFLASI_TAHUNAN = st.number_input("Asumsi Inflasi Tahunan (%)", 
                                            min_value=0.0, 
                                            max_value=10.0, 
                                            value=3.5, 
                                            step=0.5)
    
    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 30, 10)
    dca_per_saham = st.number_input("💸 Nominal DCA per Saham / Bulan (Rp)", 
                                   min_value=10000, 
                                   step=10000, 
                                   value=500000)

    # Hitung tanggal
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=durasi_tahun)
    start_year = start_date.year
    end_year = end_date.year
    
    # Simpan hasil per saham
    results = []
    
    total_awal = 0
    total_nilai_awal = 0
    total_dca = 0
    total_dca_drip = 0
    total_investasi_dca = 0

    progress_bar = st.progress(0)
    total_stocks = len(portfolio_df)
    
    for idx, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100  # 1 lot = 100 lembar saham

        try:
            # Format ticker yang benar untuk yfinance
            if '.' in ticker:
                # Jika sudah ada titik, gunakan langsung
                yf_ticker = ticker
            else:
                # Jika tidak ada titik, tambahkan .JK
                yf_ticker = ticker + '.JK'
            
            # Unduh data untuk satu ticker
            hist_data = yf.download(
                yf_ticker, 
                start=start_date, 
                end=end_date, 
                interval="1mo", 
                progress=False, 
                auto_adjust=True
            )
            
            if hist_data.empty:
                st.warning(f"⚠️ Data tidak tersedia untuk {ticker} (mencoba {yf_ticker})")
                progress_bar.progress((idx+1) / total_stocks)
                continue
                
            if 'Close' not in hist_data.columns:
                st.warning(f"⚠️ Kolom 'Close' tidak ada untuk {ticker} (mencoba {yf_ticker})")
                progress_bar.progress((idx+1) / total_stocks)
                continue
                
            prices = hist_data['Close'].dropna()
            
            if len(prices) < 2:
                st.warning(f"⚠️ Data historis tidak cukup untuk {ticker} (mencoba {yf_ticker})")
                progress_bar.progress((idx+1) / total_stocks)
                continue
                
            # Tanggal pertama dan terakhir
            actual_start_date = prices.index[0]
            actual_end_date = prices.index[-1]
            
            harga_awal = prices.iloc[0]
            harga_akhir = prices.iloc[-1]

            # 1. Strategi: Tanpa strategi (hold saja)
            nilai_awal = saham_awal * harga_awal
            nilai_akhir_tanpa_strategi = saham_awal * harga_akhir
            
            # 2. Strategi: DCA
            saham_dca, total_invested_dca = simulate_dca(
                prices, 
                dca_per_saham,
                start_date=actual_start_date,
                end_date=actual_end_date
            )
            nilai_dca = saham_dca * harga_akhir
            
            # 3. Strategi: DCA + DRIP
            saham_dca_drip = simulate_reinvest_dividen(
                yf_ticker, 
                saham_dca, 
                prices,
                start_date=actual_start_date,
                end_date=actual_end_date
            )
            nilai_dca_drip = saham_dca_drip * harga_akhir
            
            # Simpan hasil per saham
            results.append({
                'Ticker': ticker,
                'Tanpa Strategi': nilai_akhir_tanpa_strategi,
                'DCA': nilai_dca,
                'DCA+DRIP': nilai_dca_drip,
                'Investasi DCA': total_invested_dca
            })
            
            # Akumulasi hasil
            total_awal += nilai_akhir_tanpa_strategi
            total_nilai_awal += nilai_awal
            total_dca += nilai_dca
            total_dca_drip += nilai_dca_drip
            total_investasi_dca += total_invested_dca

        except Exception as e:
            st.warning(f"⚠️ Gagal memproses {ticker}: {str(e)}")
        finally:
            progress_bar.progress((idx+1) / total_stocks)

    # Jika tidak ada data yang berhasil diunduh
    if total_nilai_awal == 0:
        st.error("❌ Tidak ada data yang berhasil diunduh. Simulasi tidak dapat dilakukan.")
        return

    # Fungsi perhitungan return
    def calculate_return(nilai_akhir, nilai_awal):
        return ((nilai_akhir - nilai_awal) / nilai_awal * 100) if nilai_awal > 0 else 0

    # Fungsi perhitungan CAGR
    def calculate_cagr(end_value, start_value, years):
        if start_value <= 0 or years <= 0:
            return 0
        return ((end_value / start_value) ** (1/years) - 1) * 100

    # Hitung CAGR untuk setiap strategi
    cagr_tanpa_strategi = calculate_cagr(total_awal, total_nilai_awal, durasi_tahun)
    cagr_dca = calculate_cagr(total_dca, total_investasi_dca, durasi_tahun)
    cagr_dca_drip = calculate_cagr(total_dca_drip, total_investasi_dca, durasi_tahun)
    
    # Buat DataFrame hasil utama
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
        ],
        "CAGR (%)": [
            cagr_tanpa_strategi,
            cagr_dca,
            cagr_dca_drip
        ]
    })

    # Hitung nilai setelah inflasi
    result["Nilai Riil"] = result.apply(
        lambda row: adjust_for_inflation(row["Nilai Akhir"], start_year, end_year),
        axis=1
    )

    # Format tampilan
    result_display = result.copy()
    result_display["Nilai Akhir"] = result_display["Nilai Akhir"].apply(format_currency_idr)
    result_display["Nilai Riil"] = result_display["Nilai Riil"].apply(format_currency_idr)
    result_display["Total Investasi"] = result_display["Total Investasi"].apply(format_currency_idr)
    result_display["Return (%)"] = result_display["Return (%)"].map(lambda x: f"{x:.2f}%")
    result_display["CAGR (%)"] = result_display["CAGR (%)"].map(lambda x: f"{x:.2f}%")

    # Tampilkan hasil
    st.subheader("📈 Hasil Simulasi Portofolio")
    st.dataframe(result_display, use_container_width=True)
    
    # Tampilkan detail per saham
    if results:
        st.subheader("📊 Detail Per Saham")
        detail_df = pd.DataFrame(results)
        detail_df["Tanpa Strategi"] = detail_df["Tanpa Strategi"].apply(format_currency_idr)
        detail_df["DCA"] = detail_df["DCA"].apply(format_currency_idr)
        detail_df["DCA+DRIP"] = detail_df["DCA+DRIP"].apply(format_currency_idr)
        detail_df["Investasi DCA"] = detail_df["Investasi DCA"].apply(format_currency_idr)
        st.dataframe(detail_df, use_container_width=True)
    
    st.caption("""
    **Catatan:**
    - **CAGR**: Compound Annual Growth Rate
    - **Nilai Riil**: Nilai setelah disesuaikan dengan inflasi
    - Biaya transaksi diasumsikan 0.15%
    - Inflasi diasumsikan 3.5% per tahun
    - Simulasi menggunakan harga yang sudah disesuaikan (adjusted close)
    """)
