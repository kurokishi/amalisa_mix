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

    # Validasi kolom
    if 'Stock' not in portfolio_df.columns or 'Ticker' not in portfolio_df.columns:
        st.error("Format portofolio tidak valid. Pastikan ada kolom 'Stock' dan 'Ticker'.")
        return
        
    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
    dca_per_saham = st.number_input("💸 Nominal DCA per Saham / Bulan (Rp)", min_value=10000, step=10000, value=500000)
    
    saham_dijual = st.selectbox("📤 Pilih Saham untuk Dijual", portfolio_df['Stock'])
    bulan_jual = st.slider("📅 Bulan Ke-berapa Dilakukan Penjualan Saham?", 1, durasi_tahun * 12, 6)

    # Inisialisasi variabel
    hasil_penjualan = 0.0  # Pastikan sebagai float
    alokasi_ke = []
    nilai_akhir_normal = 0.0
    nilai_akhir_realokasi = 0.0
    total_bulan = durasi_tahun * 12
    start_date = (datetime.now() - timedelta(days=durasi_tahun * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    data_harga = {}

    # Pre-download harga saham untuk semua ticker
    with st.spinner("Mengambil data harga saham..."):
        tickers = portfolio_df['Ticker'].unique().tolist()
        for ticker in tickers:
            try:
                hist = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    interval="1mo",
                    auto_adjust=True,
                    actions=False,
                    progress=False
                )
                if not hist.empty and 'Close' in hist.columns:
                    data_harga[ticker] = hist['Close'].dropna()
            except Exception as e:
                st.warning(f"Gagal mengambil data {ticker}: {str(e)}")

    # 1. Hitung hasil penjualan saham terpilih
    try:
        saham_dijual_row = portfolio_df[portfolio_df['Stock'] == saham_dijual].iloc[0]
        ticker_dijual = saham_dijual_row['Ticker']
        
        if ticker_dijual in data_harga:
            prices = data_harga[ticker_dijual]
            if len(prices) >= bulan_jual:
                harga_jual = prices.iloc[bulan_jual - 1]
                saham_awal = float(saham_dijual_row.get('Lots', 0)) * 100
                hasil_penjualan = saham_awal * harga_jual
            else:
                st.warning(f"Data harga tidak cukup untuk {saham_dijual} pada bulan {bulan_jual}")
        else:
            st.warning(f"Data harga tidak tersedia untuk {saham_dijual}")
    except Exception as e:
        st.error(f"Error menghitung penjualan saham: {str(e)}")
        hasil_penjualan = 0.0

    # Konversi hasil_penjualan ke float
    hasil_penjualan = float(hasil_penjualan) if not isinstance(hasil_penjualan, pd.Series) else hasil_penjualan.item()

    # 2. Hitung nilai akhir portofolio
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        saham_awal = float(row.get('Lots', 0)) * 100
        
        if ticker not in data_harga or data_harga[ticker].empty:
            continue
            
        prices = data_harga[ticker]
        harga_akhir = prices.iloc[-1] if len(prices) > 0 else 0

        # A. Nilai akhir tanpa rotasi (semua saham dipertahankan)
        nilai_akhir_normal += saham_awal * harga_akhir

        # B. Untuk realokasi (skip saham yang dijual)
        if row['Stock'] == saham_dijual:
            continue

        # C. Hitung realokasi untuk saham lainnya
        try:
            # Periode sebelum penjualan (jika ada)
            dca_normal = 0
            if bulan_jual > 1:
                dca_normal, _ = simulate_dca(prices[:bulan_jual], dca_per_saham)

            # Periode setelah penjualan
            sisa_bulan = max(1, total_bulan - bulan_jual + 1)
            tambahan_dana = hasil_penjualan / (len(portfolio_df) - 1) / sisa_bulan
            
            dca_tambahan = 0
            if sisa_bulan > 0 and len(prices) >= bulan_jual:
                dca_tambahan, _ = simulate_dca(
                    prices[bulan_jual-1:], 
                    dca_per_saham + tambahan_dana
                )

            total_saham = saham_awal + dca_normal + dca_tambahan
            nilai_akhir = total_saham * harga_akhir
            nilai_akhir_realokasi += nilai_akhir

            alokasi_ke.append({
                "Ticker": ticker,
                "Harga Akhir": harga_akhir,
                "Nilai Akhir": nilai_akhir,
                "Saham Tambahan": dca_tambahan
            })
        except Exception as e:
            st.warning(f"Gagal memproses DCA {ticker}: {str(e)}")

    # 3. Tampilkan hasil
    st.subheader("📊 Ringkasan Simulasi")
    st.markdown(f"💰 **Dana hasil penjualan**: {format_currency_idr(hasil_penjualan)}")

    # PERBAIKAN UTAMA: Pastikan nilai_akhir_normal adalah float, bukan Series
    nilai_akhir_normal = float(nilai_akhir_normal) if not isinstance(nilai_akhir_normal, pd.Series) else nilai_akhir_normal.item()
    
    # Hitung return
    hasil = pd.DataFrame({
        "Strategi": ["📈 Tanpa Rotasi", "🔄 Dengan Realokasi"],
        "Nilai Akhir": [nilai_akhir_normal, nilai_akhir_realokasi],
    })
    
    # Tambahan metrik perbandingan
    hasil['Selisih'] = hasil['Nilai Akhir'] - nilai_akhir_normal
    
    # PERBAIKAN: Gunakan nilai referensi yang sudah di-convert ke float
    nilai_ref = max(1, nilai_akhir_normal)
    hasil['Peningkatan (%)'] = (hasil['Nilai Akhir'] / nilai_ref - 1) * 100

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
        df_alokasi['Harga Akhir'] = df_alokasi['Harga Akhir'].apply(lambda x: f"Rp{x:,.2f}")
        st.dataframe(df_alokasi.set_index('Ticker'), use_container_width=True)
