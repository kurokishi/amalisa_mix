import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.formatter import format_currency_idr


def simulate_dca(prices, dca_nominal):
    total_shares = 0
    total_invested = 0
    for price in prices:
        try:
            # Pastikan harga valid dan numerik
            price = float(price)
            if price > 0 and not pd.isna(price):
                shares = dca_nominal / price
                total_shares += shares
                total_invested += dca_nominal
        except (TypeError, ValueError):
            continue
    return total_shares, total_invested


def simulate_reinvest_dividen(ticker, base_shares, price_history):
    if base_shares <= 0:
        return base_shares  # Tidak ada dividen jika tidak ada saham

    try:
        stock = yf.Ticker(ticker)
        start = price_history.index[0].tz_localize(None)
        end = price_history.index[-1].tz_localize(None)
        divs = stock.dividends.tz_localize(None)[start:end]
    except Exception:
        return base_shares  # Kembalikan saham awal jika error

    total_shares = base_shares
    for date, div_per_share in divs.items():
        if date in price_history.index:
            try:
                close_price = price_history.loc[date]['Close']
                # Pastikan harga valid
                if pd.isna(close_price) or close_price <= 0:
                    continue
                    
                div_total = div_per_share * total_shares
                tambahan_saham = div_total / close_price
                total_shares += tambahan_saham
            except Exception:
                continue
    return total_shares


def strategy_simulation(portfolio_df):
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
    hist = yf.download(ticker, start=start_date, interval="1mo", progress=False)

    # Pengecekan data historis
    if hist.empty or len(hist) < 6 or 'Close' not in hist.columns or hist['Close'].isnull().all():
        st.error("⛔ Data historis tidak mencukupi untuk simulasi")
        return

    prices = hist['Close'].dropna()
    
    if prices.empty:
        st.error("⛔ Tidak ada data harga yang valid")
        return

    try:
        harga_awal = prices.iloc[0]
        harga_akhir = prices.iloc[-1]
        st.info(f"📉 Harga awal: {harga_awal:.2f}, harga akhir: {harga_akhir:.2f}")
    except Exception as e:
        st.warning(f"⚠️ Gagal menampilkan harga: {e}")
        harga_awal = harga_akhir = 0

    # PERBAIKAN 1: Gunakan saham user sebagai dasar semua strategi
    saham_awal = saham_awal_input

    # 1. Strategi: Tanpa Strategi
    nilai_akhir_baseline = saham_awal * harga_akhir
    investasi_awal_baseline = saham_awal * harga_awal if harga_awal > 0 else 0
    return_baseline = ((nilai_akhir_baseline - investasi_awal_baseline) / investasi_awal_baseline * 100 
    if investasi_awal_baseline <= 0:
        return_baseline = 0.0

    # 2. Strategi: DCA
    saham_dari_dca, total_invested_dca = simulate_dca(prices, dca_nominal)
    total_saham_dca = saham_awal + saham_dari_dca
    nilai_dca = total_saham_dca * harga_akhir
    total_investasi_dca = investasi_awal_baseline + total_invested_dca
    return_dca = ((nilai_dca - total_investasi_dca) / total_investasi_dca * 100 
    if total_investasi_dca <= 0:
        return_dca = 0.0

    # PERBAIKAN 2: DRIP menggunakan saham awal + DCA
    # 3. Strategi: DCA + DRIP
    total_saham_dca_drip = simulate_reinvest_dividen(ticker, total_saham_dca, hist)
    nilai_dca_drip = total_saham_dca_drip * harga_akhir
    return_dca_drip = ((nilai_dca_drip - total_investasi_dca) / total_investasi_dca * 100 
    if total_investasi_dca <= 0:
        return_dca_drip = 0.0

    # Hasil akhir
    result = pd.DataFrame({
        "Strategi": ["Tanpa Strategi", "📆 DCA", "📆 DCA + 🔁 DRIP"],
        "Total Saham": [saham_awal, total_saham_dca, total_saham_dca_drip],
        "Nilai Akhir": [nilai_akhir_baseline, nilai_dca, nilai_dca_drip],
        "Return (%)": [return_baseline, return_dca, return_dca_drip]
    })

    st.subheader("📊 Hasil Simulasi Strategi")
    fig = px.bar(result, x="Strategi", y="Nilai Akhir", text_auto='.2s', color="Strategi",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Perbandingan Nilai Akhir per Strategi", hover_data=["Return (%)"])
    st.plotly_chart(fig, use_container_width=True)

    result_display = result.copy()
    result_display['Nilai Akhir'] = result_display['Nilai Akhir'].apply(format_currency_idr)
    result_display['Total Saham'] = pd.to_numeric(result_display['Total Saham'], errors='coerce').round(4)
    result_display['Return (%)'] = result_display['Return (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
    
    st.dataframe(result_display, use_container_width=True)
    st.caption("Simulasi ini menggunakan data historis dan dividen aktual dari yfinance.")
