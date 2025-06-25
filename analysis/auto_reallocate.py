import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

    # Validasi pilihan saham
    if 'Stock' not in portfolio_df.columns:
        st.error("Kolom 'Stock' tidak ditemukan dalam portofolio.")
        return
        
    saham_dijual = st.selectbox("📤 Pilih Saham untuk Dijual", portfolio_df['Stock'])
    bulan_jual = st.slider("📅 Bulan Ke-berapa Dilakukan Penjualan Saham?", 1, durasi_tahun * 12, 6)

    # Inisialisasi variabel
    hasil_penjualan = 0
    alokasi_ke = []
    nilai_akhir_normal = 0
    nilai_akhir_realokasi = 0
    total_bulan = durasi_tahun * 12
    start_date = (datetime.now() - timedelta(days=durasi_tahun * 365)).strftime("%Y-%m-%d")
    data_harga = {}

    # Pre-download harga saham untuk performa
    with st.spinner("Mengambil data harga saham..."):
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            try:
                hist = yf.download(ticker, start=start_date, interval="1mo", progress=False)
                if not hist.empty and 'Close' in hist.columns:
                    data_harga[ticker] = hist['Close'].dropna()
            except Exception as e:
                st.warning(f"Gagal mengambil data {ticker}: {str(e)}")

    # Hitung nilai portofolio tanpa rotasi
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100
        
        if ticker not in data_harga or data_harga[ticker].empty:
            continue
            
        prices = data_harga[ticker]
        harga_akhir = prices.iloc[-1] if len(prices) > 0 else 0
        nilai_akhir_normal += saham_awal * harga_akhir

    # Proses penjualan dan realokasi
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100
        
        if ticker not in data_harga or data_harga[ticker].empty:
            continue
            
        prices = data_harga[ticker]
        harga_akhir = prices.iloc[-1] if len(prices) > 0 else 0

        # Penjualan saham terpilih
        if row['Stock'] == saham_dijual:
            if len(prices) >= bulan_jual:
                harga_jual = prices.iloc[bulan_jual - 1]
                hasil_penjualan = saham_awal * harga_jual
            continue

        # Simulasi DCA untuk saham lainnya
        if hasil_penjualan > 0:
            sisa_bulan = max(1, total_bulan - bulan_jual + 1)
            tambahan_dana = hasil_penjualan / (len(portfolio_df) - 1) / sisa_bulan
        else:
            tambahan_dana = 0

        # Simulasi DCA dengan tambahan dana setelah penjualan
        try:
            # Split DCA periode
            dca_normal = simulate_dca(prices[:bulan_jual], dca_per_saham)[0] if bulan_jual > 1 else 0
            dca_tambahan = simulate_dca(prices[bulan_jual-1:], dca_per_saham + tambahan_dana)[0] if sisa_bulan > 0 else 0
            
            total_saham = saham_awal + dca_normal + dca_tambahan
            nilai_akhir = total_saham * harga_akhir
            nilai_akhir_realokasi += nilai_akhir

            alokasi_ke.append({
                "Ticker": ticker,
                "Harga Akhir": harga_akhir,
                "Nilai Akhir": nilai_akhir
            })
        except Exception as e:
            st.warning(f"Gagal memproses DCA {ticker}: {str(e)}")

    # Ringkasan hasil
    st.subheader("📊 Ringkasan Simulasi")
    st.markdown(f"💰 **Dana hasil penjualan**: {format_currency_idr(hasil_penjualan)}")

    # Hitung return relatif
    hasil = pd.DataFrame({
        "Strategi": ["📈 Tanpa Rotasi", "🔄 Dengan Realokasi"],
        "Nilai Akhir": [nilai_akhir_normal, nilai_akhir_realokasi],
    })
    
    # Tambahkan kolom perbandingan
    hasil['Selisih'] = hasil['Nilai Akhir'] - nilai_akhir_normal
    hasil['Peningkatan (%)'] = (hasil['Nilai Akhir'] / nilai_akhir_normal - 1) * 100

    # Format tampilan
    hasil_display = hasil.copy()
    hasil_display['Nilai Akhir'] = hasil_display['Nilai Akhir'].apply(format_currency_idr)
    hasil_display['Selisih'] = hasil_display['Selisih'].apply(format_currency_idr)
    hasil_display['Peningkatan (%)'] = hasil_display['Peningkatan (%)'].map(lambda x: f"{x:.2f}%")

    st.dataframe(hasil_display.set_index('Strategi'), use_container_width=True)

    # Tampilkan alokasi dana
    if alokasi_ke:
        st.markdown("---")
        st.subheader("📦 Alokasi Dana ke Saham Lain")
        df_alokasi = pd.DataFrame(alokasi_ke)
        df_alokasi['Nilai Akhir'] = df_alokasi['Nilai Akhir'].apply(format_currency_idr)
        st.dataframe(df_alokasi.set_index('Ticker'), use_container_width=True)
