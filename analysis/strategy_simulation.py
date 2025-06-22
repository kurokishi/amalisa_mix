import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import threading
import time
from utils.formatter import format_currency_idr

# Cache data historis untuk mengurangi beban
@st.cache_data(ttl=3600, show_spinner=False)
def download_hist_data(ticker, start_date):
    try:
        return yf.download(ticker, start=start_date, interval="1mo", progress=False)
    except Exception as e:
        st.error(f"⛔ Gagal mengunduh data: {str(e)}")
        return pd.DataFrame()

# Thread-safe DCA simulation
def simulate_dca_thread(prices, dca_nominal, result_container):
    total_shares = 0
    total_invested = 0
    for price in prices:
        try:
            price = float(price)
            if price > 0 and not pd.isna(price):
                shares = dca_nominal / price
                total_shares += shares
                total_invested += dca_nominal
        except Exception:
            continue
    result_container['dca'] = (total_shares, total_invested)

# Thread-safe dividend simulation
def simulate_dividend_thread(ticker, base_shares, price_history, result_container):
    if base_shares <= 0:
        result_container['drip'] = base_shares
        return

    try:
        stock = yf.Ticker(ticker)
        start = price_history.index[0].tz_localize(None)
        end = price_history.index[-1].tz_localize(None)
        
        # Get dividends with timeout
        def get_dividends():
            try:
                return stock.dividends.tz_localize(None)[start:end]
            except Exception:
                return pd.Series()
        
        divs = get_dividends()
        if divs.empty:
            result_container['drip'] = base_shares
            return

        total_shares = base_shares
        for date, div_per_share in divs.items():
            if date in price_history.index:
                try:
                    close_price = price_history.loc[date]['Close']
                    if pd.isna(close_price) or close_price <= 0:
                        continue
                    
                    div_total = div_per_share * total_shares
                    tambahan_saham = div_total / close_price
                    total_shares += tambahan_saham
                except Exception:
                    continue
        result_container['drip'] = total_shares
    except Exception:
        result_container['drip'] = base_shares

def show_strategy_simulation(portfolio_df):
    st.header("🧪 Simulasi Strategi Terintegrasi per Saham")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    selected_stock = st.selectbox("📌 Pilih Saham", portfolio_df['Stock'])
    row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
    ticker = row['Ticker']

    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
    dca_nominal = st.number_input("💸 Nominal DCA / Bulan (Rp)", min_value=10000, step=10000, value=500000)
    saham_awal_input = st.number_input("📦 Jumlah Saham Awal yang Dimiliki", min_value=0.0, step=0.01, value=0.0)

    start_date = (datetime.now() - timedelta(days=durasi_tahun*365)).strftime("%Y-%m-%d")
    
    # Download data dengan spinner
    with st.spinner("🔄 Mengunduh data historis..."):
        hist = download_hist_data(ticker, start_date)
        time.sleep(0.5)  # Beri waktu untuk spinner

    if hist is None or hist.empty:
        st.error("⛔ Data historis tidak tersedia")
        return

    # Pengecekan data historis
    if len(hist) < 6:
        st.error("⛔ Data historis terlalu pendek untuk simulasi")
        return
        
    if 'Close' not in hist.columns:
        st.error("⛔ Kolom 'Close' tidak tersedia di data")
        return

    prices = hist['Close'].dropna()
    
    if prices.empty:
        st.error("⛔ Tidak ada data harga yang valid")
        return

    try:
        harga_awal = prices.iloc[0]
        harga_akhir = prices.iloc[-1]
        st.info(f"📉 Harga awal: {format_currency_idr(harga_awal)}, harga akhir: {format_currency_idr(harga_akhir)}")
    except Exception as e:
        st.warning(f"⚠️ Gagal menampilkan harga: {e}")
        harga_awal = harga_akhir = 0

    # Gunakan saham user sebagai dasar semua strategi
    saham_awal = saham_awal_input

    # 1. Strategi: Tanpa Strategi
    nilai_akhir_baseline = saham_awal * harga_akhir
    investasi_awal_baseline = saham_awal * harga_awal if harga_awal > 0 else 0
    
    if investasi_awal_baseline > 0:
        return_baseline = (nilai_akhir_baseline - investasi_awal_baseline) / investasi_awal_baseline * 100
    else:
        return_baseline = 0.0

    # Container untuk hasil thread
    result_container = {'dca': None, 'drip': None}
    
    # Jalankan simulasi DCA di thread terpisah
    dca_thread = threading.Thread(
        target=simulate_dca_thread, 
        args=(prices, dca_nominal, result_container)
    )
    dca_thread.start()
    
    # 2. Strategi: DCA
    with st.spinner("🔄 Menghitung strategi DCA..."):
        dca_thread.join(timeout=30)  # Timeout 30 detik
        
        if result_container['dca'] is None:
            st.error("⏱️ Timeout saat menghitung DCA")
            saham_dari_dca, total_invested_dca = (0, 0)
        else:
            saham_dari_dca, total_invested_dca = result_container['dca']
        
        total_saham_dca = saham_awal + saham_dari_dca
        nilai_dca = total_saham_dca * harga_akhir
        total_investasi_dca = investasi_awal_baseline + total_invested_dca
        
        if total_investasi_dca > 0:
            return_dca = (nilai_dca - total_investasi_dca) / total_investasi_dca * 100
        else:
            return_dca = 0.0

    # 3. Strategi: DCA + DRIP
    drip_thread = threading.Thread(
        target=simulate_dividend_thread, 
        args=(ticker, total_saham_dca, hist, result_container)
    )
    drip_thread.start()
    
    with st.spinner("🔄 Menghitung strategi DRIP..."):
        drip_thread.join(timeout=45)  # Timeout 45 detik
        
        if result_container['drip'] is None:
            st.error("⏱️ Timeout saat menghitung DRIP")
            total_saham_dca_drip = total_saham_dca
        else:
            total_saham_dca_drip = result_container['drip']
        
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
    
    # Pastikan nilai akhir valid
    if not result['Nilai Akhir'].isnull().any():
        fig = px.bar(result, x="Strategi", y="Nilai Akhir", text_auto='.2s', color="Strategi",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="Perbandingan Nilai Akhir per Strategi", hover_data=["Return (%)"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Tidak dapat menampilkan grafik karena data tidak lengkap")

    result_display = result.copy()
    result_display['Nilai Akhir'] = result_display['Nilai Akhir'].apply(format_currency_idr)
    result_display['Total Saham'] = result_display['Total Saham'].apply(lambda x: f"{x:.4f}")
    
    # Format return dengan 2 desimal dan simbol persen
    result_display['Return (%)'] = result_display['Return (%)'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "-"
    )
    
    st.dataframe(result_display, use_container_width=True)
    st.caption("Simulasi ini menggunakan data historis dan dividen aktual dari yfinance.")
