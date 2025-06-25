import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.formatter import format_currency_idr

def show_auto_reallocation_simulation(portfolio_df):
    st.header("🔄 Simulasi Rotasi Modal dari Penjualan Saham")

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu.")
        return

    # Identifikasi kolom yang tersedia
    available_columns = portfolio_df.columns.tolist()
    
    # Cari kolom untuk jumlah saham
    quantity_col = None
    possible_quantity_cols = ['Lots', 'Jumlah', 'Quantity', 'Shares', 'Saham']
    for col in possible_quantity_cols:
        if col in available_columns:
            quantity_col = col
            break
    
    if not quantity_col:
        st.error("Format portofolio tidak valid. Tidak ditemukan kolom untuk jumlah saham (cari: Lots, Jumlah, Quantity, Shares, Saham).")
        st.write("Kolom yang tersedia:", ", ".join(available_columns))
        return
        
    # Validasi kolom ticker
    if 'Ticker' not in available_columns:
        st.error("Kolom 'Ticker' tidak ditemukan dalam portofolio.")
        return
        
    # Jika tidak ada kolom Stock, gunakan Ticker sebagai nama saham
    if 'Stock' not in available_columns:
        portfolio_df['Stock'] = portfolio_df['Ticker']
    
    durasi_tahun = st.slider("⏳ Durasi Simulasi (tahun)", 1, 10, 5)
    saham_dijual = st.selectbox("📤 Pilih Saham untuk Dijual", portfolio_df['Stock'])
    bulan_jual = st.slider("📅 Bulan Ke-berapa Dilakukan Penjualan Saham?", 1, durasi_tahun * 12, 6)

    # Inisialisasi variabel
    hasil_penjualan = 0.0
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
                else:
                    st.warning(f"Data harga tidak tersedia untuk {ticker}")
            except Exception as e:
                st.warning(f"Gagal mengambil data {ticker}: {str(e)}")

    # 1. Hitung hasil penjualan saham terpilih
    try:
        saham_dijual_row = portfolio_df[portfolio_df['Stock'] == saham_dijual].iloc[0]
        ticker_dijual = saham_dijual_row['Ticker']
        
        if ticker_dijual in data_harga:
            prices = data_harga[ticker_dijual]
            if len(prices) >= bulan_jual:
                harga_jual = float(prices.iloc[bulan_jual - 1])  # Konversi ke float
                
                # Konversi jumlah saham
                quantity = float(saham_dijual_row[quantity_col])
                saham_awal = quantity * (100 if quantity_col == 'Lots' else 1)
                
                hasil_penjualan = saham_awal * harga_jual
                st.success(f"Penjualan {saham_dijual}: {saham_awal:.0f} saham @ {harga_jual:.2f}")
            else:
                st.warning(f"Data harga tidak cukup untuk {saham_dijual} pada bulan {bulan_jual}")
        else:
            st.warning(f"Data harga tidak tersedia untuk {saham_dijual}")
    except Exception as e:
        st.error(f"Error menghitung penjualan saham: {str(e)}")
        hasil_penjualan = 0.0

    # 2. Hitung nilai akhir portofolio
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        
        # Konversi jumlah saham
        quantity = float(row[quantity_col])
        saham_awal = quantity * (100 if quantity_col == 'Lots' else 1)
        
        if ticker not in data_harga or data_harga[ticker].empty:
            continue
            
        prices = data_harga[ticker]
        if len(prices) > 0:
            harga_akhir = float(prices.iloc[-1])  # Konversi ke float
        else:
            harga_akhir = 0.0

        # A. Nilai akhir tanpa rotasi (semua saham dipertahankan)
        nilai_saham = saham_awal * harga_akhir
        nilai_akhir_normal += nilai_saham

        # B. Untuk realokasi (skip saham yang dijual)
        if row['Stock'] == saham_dijual:
            continue

        # C. Hitung realokasi untuk saham lainnya
        try:
            # Sederhanakan: alokasikan hasil penjualan secara merata
            tambahan_dana = hasil_penjualan / (len(portfolio_df) - 1)
            
            # Hitung jumlah saham tambahan yang bisa dibeli
            if len(prices) >= bulan_jual:
                harga_beli = float(prices.iloc[bulan_jual - 1])
                saham_tambahan = tambahan_dana / harga_beli if harga_beli > 0 else 0
            else:
                saham_tambahan = 0
                harga_beli = 0.0
                
            total_saham = saham_awal + saham_tambahan
            nilai_akhir = total_saham * harga_akhir
            nilai_akhir_realokasi += nilai_akhir

            alokasi_ke.append({
                "Ticker": ticker,
                "Harga Beli": harga_beli,
                "Saham Tambahan": saham_tambahan,
                "Nilai Akhir": nilai_akhir
            })
        except Exception as e:
            st.warning(f"Gagal memproses realokasi {ticker}: {str(e)}")

    # Konversi hasil akhir ke float
    nilai_akhir_normal = float(nilai_akhir_normal)
    nilai_akhir_realokasi = float(nilai_akhir_realokasi)

    # 3. Tampilkan hasil
    st.subheader("📊 Ringkasan Simulasi")
    st.markdown(f"💰 **Dana hasil penjualan**: {format_currency_idr(hasil_penjualan)}")

    # Hitung return sederhana
    hasil_df = pd.DataFrame({
        "Strategi": ["📈 Tanpa Rotasi", "🔄 Dengan Realokasi"],
        "Nilai Akhir": [nilai_akhir_normal, nilai_akhir_realokasi],
    })
    
    # Hitung selisih dan persentase
    hasil_df['Selisih'] = hasil_df['Nilai Akhir'] - nilai_akhir_normal
    hasil_df['Peningkatan (%)'] = (hasil_df['Selisih'] / max(1.0, nilai_akhir_normal)) * 100

    # Format tampilan
    hasil_display = hasil_df.copy()
    hasil_display['Nilai Akhir'] = hasil_display['Nilai Akhir'].apply(lambda x: format_currency_idr(float(x)))
    hasil_display['Selisih'] = hasil_display['Selisih'].apply(lambda x: format_currency_idr(float(x)))
    hasil_display['Peningkatan (%)'] = hasil_display['Peningkatan (%)'].apply(lambda x: f"{float(x):.2f}%")

    st.dataframe(hasil_display.set_index('Strategi'), use_container_width=True)

    # Tampilkan alokasi dana
    if alokasi_ke:
        st.markdown("---")
        st.subheader("📦 Alokasi Dana ke Saham Lain")
        df_alokasi = pd.DataFrame(alokasi_ke)
        
        # Format kolom nilai
        df_alokasi['Nilai Akhir'] = df_alokasi['Nilai Akhir'].apply(lambda x: format_currency_idr(float(x)))
        df_alokasi['Harga Beli'] = df_alokasi['Harga Beli'].apply(
            lambda x: f"Rp{x:,.2f}" if x > 0 else "N/A"
        )
        df_alokasi['Saham Tambahan'] = df_alokasi['Saham Tambahan'].apply(
            lambda x: f"{x:,.2f}" if x > 0 else "0"
        )
        
        st.dataframe(df_alokasi.set_index('Ticker'), use_container_width=True)
