import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.formatter import format_currency_idr
from analysis.strategy_simulation import simulate_dca


def show_auto_reallocation_simulation(portfolio_df):
    st.header("🔄 Simulasi Rotasi Modal dari Penjualan Saham")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
    dca_per_saham = st.number_input("💸 Nominal DCA per Saham / Bulan (Rp)", min_value=10000, step=10000, value=500000)

    saham_dijual = st.selectbox("📤 Pilih Saham untuk Dijual", portfolio_df['Stock'])
    bulan_jual = st.slider("📅 Bulan Ke-berapa Dilakukan Penjualan Saham?", 1, durasi_tahun * 12, 6)

    hasil_penjualan = 0
    alokasi_ke = []
    nilai_akhir_normal = 0
    nilai_akhir_realokasi = 0

    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100
        start_date = (datetime.now() - timedelta(days=durasi_tahun * 365)).strftime("%Y-%m-%d")

        try:
            hist = yf.download(ticker, start=start_date, interval="1mo", progress=False)
            if hist.empty or 'Close' not in hist.columns or hist['Close'].isnull().all():
                continue
            prices = hist['Close'].dropna()
            harga_akhir = prices.iloc[-1]

            # Tanpa rotasi
            nilai_normal = saham_awal * harga_akhir
            nilai_akhir_normal += nilai_normal

            # Jika saham ini yang dijual
            if row['Stock'] == saham_dijual and len(prices) >= bulan_jual:
                harga_jual = prices.iloc[bulan_jual - 1]
                hasil_penjualan = saham_awal * harga_jual
                continue  # saham dijual, tidak disimpan

            # Saham lain: tambah alokasi jika ada hasil penjualan
            tambahan_dana = hasil_penjualan / (len(portfolio_df) - 1) if hasil_penjualan > 0 else 0
            saham_dca, _ = simulate_dca(prices, dca_per_saham + tambahan_dana)
            nilai_realokasi = saham_dca * harga_akhir
            nilai_akhir_realokasi += nilai_realokasi

            alokasi_ke.append({
                "Ticker": ticker,
                "Harga Akhir": harga_akhir,
                "Nilai Akhir": nilai_realokasi
            })

        except Exception as e:
            st.warning(f"Gagal memproses {ticker}: {e}")
            continue

    # Ringkasan
    st.subheader("📊 Ringkasan Simulasi")
    st.markdown(f"💰 **Dana hasil penjualan**: {format_currency_idr(hasil_penjualan)}")

    hasil = pd.DataFrame({
        "Strategi": ["📈 Tanpa Rotasi", "🔄 Dengan Realokasi"],
        "Nilai Akhir": [nilai_akhir_normal, nilai_akhir_realokasi],
        "Return (%)": [
            ((nilai_akhir_normal - hasil_penjualan) / hasil_penjualan * 100) if hasil_penjualan else 0,
            ((nilai_akhir_realokasi - hasil_penjualan) / hasil_penjualan * 100) if hasil_penjualan else 0
        ]
    })

    hasil_display = hasil.copy()
    hasil_display['Nilai Akhir'] = hasil_display['Nilai Akhir'].apply(format_currency_idr)
    hasil_display['Return (%)'] = hasil_display['Return (%)'].map(lambda x: f"{x:.2f}%")

    st.dataframe(hasil_display, use_container_width=True)

    if alokasi_ke:
        st.markdown("---")
        st.subheader("📦 Alokasi Dana ke Saham Lain")
        df_alokasi = pd.DataFrame(alokasi_ke)
        df_alokasi['Nilai Akhir'] = df_alokasi['Nilai Akhir'].apply(format_currency_idr)
        st.dataframe(df_alokasi, use_container_width=True)
