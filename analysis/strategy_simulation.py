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
            price = float(price)
            if price > 0:
                shares = dca_nominal / price
                total_shares += shares
                total_invested += dca_nominal
        except:
            continue
    return total_shares, total_invested


def simulate_reinvest_dividen(ticker, base_shares, price_history):
    stock = yf.Ticker(ticker)
    try:
        start = price_history.index[0].tz_localize(None)
        end = price_history.index[-1].tz_localize(None)
        divs = stock.dividends.tz_localize(None)[start:end]
    except:
        return base_shares

    total_shares = base_shares
    for date, div_per_share in divs.items():
        if date in price_history.index:
            try:
                div_total = div_per_share * total_shares
                close_price = price_history.loc[date]['Close']
                tambahan_saham = div_total / close_price if close_price > 0 else 0
                total_shares += tambahan_saham
            except:
                continue
    return total_shares


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

    start_date = (datetime.now() - timedelta(days=durasi_tahun*365)).strftime("%Y-%m-%d")
    hist = yf.download(ticker, start=start_date, interval="1mo", progress=False)

    if hist.empty or len(hist) < 6 or hist['Close'].isna().all():
        st.error("Harga historis tidak tersedia atau kosong.")
        return

    prices = hist['Close'].dropna()
    st.info(f"📉 Harga awal: {prices.iloc[0]:.2f}, harga akhir: {prices.iloc[-1]:.2f}")

    try:
        harga_awal = prices.iloc[0]
        saham_awal = dca_nominal / harga_awal
        nilai_akhir_baseline = saham_awal * prices.iloc[-1]
    except:
        saham_awal = 0
        nilai_akhir_baseline = 0

    saham_dca, total_invested_dca = simulate_dca(prices, dca_nominal)
    nilai_dca = saham_dca * prices.iloc[-1] if saham_dca > 0 else 0

    saham_dca_drip = simulate_reinvest_dividen(ticker, saham_dca, hist)
    nilai_dca_drip = saham_dca_drip * prices.iloc[-1] if saham_dca_drip > 0 else 0

    result = pd.DataFrame({
        "Strategi": ["Tanpa Strategi", "📆 DCA", "📆 DCA + 🔁 DRIP"],
        "Total Saham": [saham_awal, saham_dca, saham_dca_drip],
        "Nilai Akhir": [nilai_akhir_baseline, nilai_dca, nilai_dca_drip],
        "Return (%)": [
            (nilai_akhir_baseline - dca_nominal)/dca_nominal*100 if dca_nominal else 0,
            (nilai_dca - total_invested_dca)/total_invested_dca*100 if total_invested_dca else 0,
            (nilai_dca_drip - total_invested_dca)/total_invested_dca*100 if total_invested_dca else 0
        ]
    })

    st.subheader("📊 Hasil Simulasi Strategi")
    fig = px.bar(result, x="Strategi", y="Nilai Akhir", text_auto='.2s', color="Strategi",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Perbandingan Nilai Akhir per Strategi", hover_data=["Return (%)"])
    st.plotly_chart(fig, use_container_width=True)

    result_display = result.copy()
    result_display['Nilai Akhir'] = result_display['Nilai Akhir'].apply(format_currency_idr)
    result_display['Total Saham'] = pd.to_numeric(result_display['Total Saham'], errors='coerce').round(4)
    result_display['Return (%)'] = pd.to_numeric(result_display['Return (%)'], errors='coerce')
    result_display['Return (%)'] = result_display['Return (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
    st.dataframe(result_display, use_container_width=True)

    st.caption("Simulasi ini menggunakan data historis dan dividen aktual dari yfinance.")
