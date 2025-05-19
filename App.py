import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Fungsi untuk mengambil data saham dari yfinance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    beta = info.get("beta")
    return {
        "Ticker": ticker,
        "Nama Perusahaan": info.get("shortName"),
        "Harga Saat Ini": info.get("currentPrice"),
        "PER": info.get("trailingPE"),
        "PBV": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "Dividen Yield": info.get("dividendYield"),
        "EPS": info.get("trailingEps"),
        "Book Value": info.get("bookValue"),
        "Debt to Equity": info.get("debtToEquity"),
        "Market Cap": info.get("marketCap"),
        "Beta": beta
    }

# Fungsi untuk menghitung harga wajar menggunakan Graham Number
def graham_number(eps, book_value):
    if eps is not None and book_value is not None and eps > 0 and book_value > 0:
        return round((22.5 * eps * book_value) ** 0.5, 2)
    return None

# Fungsi untuk memberikan rekomendasi sederhana
def rekomendasi_beli(harga_saat_ini, harga_wajar):
    if harga_wajar is None:
        return "Data tidak cukup"
    if harga_saat_ini < 0.8 * harga_wajar:
        return "BELI (Undervalued)"
    elif harga_saat_ini <= harga_wajar:
        return "Tahan"
    else:
        return "Hindari (Overvalued)"

# Fungsi simulasi compound dividen
def simulasi_dividen_compound(div_yield, tahun, modal_awal=1000000):
    if div_yield is None:
        return 0
    hasil = modal_awal * ((1 + div_yield) ** tahun)
    return round(hasil, 2)

# Fungsi simulasi average down
def simulasi_avg_down(harga_beli_awal, jumlah_awal, harga_beli_baru, jumlah_baru):
    total_saham = jumlah_awal + jumlah_baru
    total_biaya = (harga_beli_awal * jumlah_awal) + (harga_beli_baru * jumlah_baru)
    avg_price = total_biaya / total_saham
    return round(avg_price, 2)

# Fungsi grafik historis
def tampilkan_grafik(ticker):
    stock = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=365)
    hist = stock.history(start=start, end=end)
    plt.figure(figsize=(10, 4))
    plt.plot(hist['Close'], label='Harga Penutupan')
    plt.title(f'Tren Harga Saham {ticker} (1 Tahun Terakhir)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    st.pyplot(plt)

# Analisis risiko sederhana (BlackRock-style) menggunakan beta

def analisis_risiko(beta):
    if beta is None:
        return "Data beta tidak tersedia"
    if beta < 0.8:
        return "Risiko rendah"
    elif beta <= 1.2:
        return "Risiko sedang"
    else:
        return "Risiko tinggi"

# Value at Risk (VaR) dengan pendekatan historis sederhana
def hitung_var(ticker, confidence_level=0.95):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")['Close']
        returns = np.log(data / data.shift(1)).dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return round(var * 100, 2)
    except:
        return None

# Diversifikasi risiko sederhana berdasarkan korelasi

def diversifikasi_risiko(tickers):
    harga = pd.DataFrame()
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="6mo")['Close']
            harga[t] = data
        except:
            continue
    returns = harga.pct_change().dropna()
    korelasi = returns.corr()
    return korelasi

st.title("Aplikasi Analisis Saham ala Lo Kheng Hong + BlackRock-style Risk Tools")

uploaded_file = st.file_uploader("Upload file CSV portofolio Anda", type=["csv"])

ticker_input = st.text_input("Atau masukkan kode saham (Contoh: UNVR.JK)")

if uploaded_file:
    df_portfolio = pd.read_csv(uploaded_file)
    st.write("Portofolio Anda:", df_portfolio)
    results = []
    tickers = df_portfolio['Ticker'].tolist()

    for ticker in tickers:
        data = get_stock_data(ticker)
        data['Harga Wajar (Graham)'] = graham_number(data['EPS'], data['Book Value'])
        data['Rekomendasi'] = rekomendasi_beli(data['Harga Saat Ini'], data['Harga Wajar (Graham)'])
        data['Proyeksi 5 Tahun (Dividen Compound)'] = simulasi_dividen_compound(data['Dividen Yield'], 5)
        data['Analisis Risiko (Beta)'] = analisis_risiko(data['Beta'])
        data['VaR (95%)'] = hitung_var(ticker)
        results.append(data)

    df_results = pd.DataFrame(results)
    st.write("\nAnalisis Fundamental dan Risiko:")
    st.dataframe(df_results)

    undervalued = df_results[df_results['Rekomendasi'].str.contains("BELI")]
    if not undervalued.empty:
        st.subheader("Saham yang Direkomendasikan untuk Dibeli (Undervalued):")
        st.dataframe(undervalued)
    else:
        st.info("Tidak ada saham yang direkomendasikan beli saat ini.")

    st.subheader("📊 Korelasi Antar Saham (Diversifikasi Risiko)")
    korelasi = diversifikasi_risiko(tickers)
    st.dataframe(korelasi)

elif ticker_input:
    try:
        data = get_stock_data(ticker_input)
        data['Harga Wajar (Graham)'] = graham_number(data['EPS'], data['Book Value'])
        data['Rekomendasi'] = rekomendasi_beli(data['Harga Saat Ini'], data['Harga Wajar (Graham)'])
        data['Proyeksi 5 Tahun (Dividen Compound)'] = simulasi_dividen_compound(data['Dividen Yield'], 5)
        data['Analisis Risiko (Beta)'] = analisis_risiko(data['Beta'])
        data['VaR (95%)'] = hitung_var(ticker_input)

        df = pd.DataFrame([data])
        st.write("Hasil Analisis Saham:")
        st.dataframe(df)

        st.subheader("Grafik Harga 1 Tahun Terakhir")
        tampilkan_grafik(ticker_input)

        st.subheader("Simulasi Average Down")
        harga_awal = st.number_input("Harga Beli Awal", value=float(data['Harga Saat Ini']))
        jumlah_awal = st.number_input("Jumlah Lot Awal", value=1)
        harga_baru = st.number_input("Harga Beli Tambahan", value=float(data['Harga Saat Ini']))
        jumlah_baru = st.number_input("Jumlah Lot Tambahan", value=1)
        if st.button("Hitung Harga Rata-rata"):
            avg = simulasi_avg_down(harga_awal, jumlah_awal * 100, harga_baru, jumlah_baru * 100)
            st.success(f"Harga Rata-rata setelah Average Down: Rp{avg:,.2f}")

    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")

else:
    st.info("Silakan upload CSV atau masukkan kode saham untuk memulai analisis.")
    
