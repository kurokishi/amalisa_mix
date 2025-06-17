import warnings
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from datetime import datetime, timedelta
import tempfile
import os

# Helper function untuk format mata uang
def format_currency_idr(value):
    """Format angka menjadi string mata uang IDR"""
    return f"Rp{value:,.0f}".replace(",", ".")

# Fungsi baru: Ambil data fundamental
def get_fundamental_data(ticker):
    """Ambil data fundamental dari Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ambil data utama
        fundamental = {
            'PER': info.get('trailingPE', None),
            'PBV': info.get('priceToBook', None),
            'ROE': info.get('returnOnEquity', None),
            'EPS': info.get('trailingEps', None),
            'DER': info.get('debtToEquity', None),
            'Dividend Yield': info.get('dividendYield', None),
            'Market Cap': info.get('marketCap', None),
            'Sektor': info.get('sector', 'N/A'),
            'Industri': info.get('industry', 'N/A')
        }
        
        # Ambil data growth
        income_stmt = stock.financials
        if not income_stmt.empty:
            try:
                revenue_growth = (income_stmt.loc['Total Revenue'].iloc[0] / 
                                  income_stmt.loc['Total Revenue'].iloc[1] - 1) * 100
                fundamental['Revenue Growth'] = revenue_growth
            except:
                fundamental['Revenue Growth'] = None
    
        return fundamental
    except Exception as e:
        st.error(f"Error getting fundamental data for {ticker}: {str(e)}")
        return None

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Portfolio Analysis Tool")
st.title("🪙 AI Portfolio Management Dashboard")

# Sidebar Menu (MODIFIKASI: tambah Fundamental Analysis)
with st.sidebar:
    st.header("Menu Navigasi")
    menu_options = ["Portfolio Analysis", "Price Prediction", "What-If Simulation", 
                    "AI Recommendations", "Compound Interest", "Fundamental Analysis"]  # MODIFIKASI
    selected_menu = st.radio("Pilih Modul:", menu_options)
    
    st.divider()
    st.header("Upload Portfolio")
    uploaded_file = st.file_uploader("Upload file (CSV/Excel)", type=["csv", "xlsx"])
    
    st.divider()
    st.header("Parameter Analisis")
    prediction_days = st.slider("Jumlah Hari Prediksi", 7, 365, 30)
    risk_tolerance = st.select_slider("Toleransi Risiko", options=["Low", "Medium", "High"])

# Fungsi untuk memproses file upload
def process_uploaded_file(uploaded_file):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi untuk mendapatkan data harga
def get_stock_data(ticker, period='5y'):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi prediksi dengan Prophet
def prophet_prediction(df, days):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi prediksi dengan LSTM
def lstm_prediction(df, days):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi prediksi dengan XGBoost
def xgboost_prediction(df, days):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi visualisasi portfolio (DIPERBAIKI untuk Rp)
def visualize_portfolio(portfolio):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi simulasi what-if (DIPERBAIKI untuk Rp)
def what_if_simulation(portfolio, new_stock, new_ticker, new_lots, new_price):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi proyeksi bunga majemuk (DIPERBAIKI untuk Rp)
def compound_interest_projection(principal, monthly_add, years, rate):
    # Tetap sama seperti sebelumnya
    # ...

# Fungsi rekomendasi AI (DIPERBAIKI untuk Rp)
def generate_recommendations(portfolio):
    # Tetap sama seperti sebelumnya
    # ...

# Main App Logic
portfolio_df = process_uploaded_file(uploaded_file)

if selected_menu == "Portfolio Analysis":
    st.header("Analisis Portofolio")
    visualize_portfolio(portfolio_df)
    
elif selected_menu == "Price Prediction":
    st.header("Prediksi Harga Saham")
    # Tetap sama seperti sebelumnya
    # ...

elif selected_menu == "What-If Simulation":
    st.header("Simulasi What-If")
    # Tetap sama seperti sebelumnya
    # ...

elif selected_menu == "AI Recommendations":
    st.header("Rekomendasi AI")
    # Tetap sama seperti sebelumnya
    # ...

elif selected_menu == "Compound Interest":
    st.header("Proyeksi Bunga Majemuk")
    # Tetap sama seperti sebelumnya
    # ...

# MODIFIKASI: Tambah Fundamental Analysis
elif selected_menu == "Fundamental Analysis":
    st.header("Analisis Fundamental Saham")
    
    if portfolio_df is None:
        st.warning("Silakan upload portofolio terlebih dahulu")
    else:
        # Dapatkan data fundamental untuk semua saham
        fundamental_data = []
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            fundamental = get_fundamental_data(ticker)
            if fundamental:
                fundamental['Stock'] = row['Stock']
                fundamental['Ticker'] = ticker
                fundamental_data.append(fundamental)
        
        if not fundamental_data:
            st.error("Tidak dapat mengambil data fundamental")
        else:
            df_fundamental = pd.DataFrame(fundamental_data)
            
            # Tampilkan data dalam tabel
            st.subheader("Data Fundamental Saham")
            st.dataframe(df_fundamental.set_index('Stock'))
            
            # Visualisasi per metrik
            metrics = ['PER', 'PBV', 'ROE', 'DER', 'Dividend Yield', 'Revenue Growth']
            selected_metric = st.selectbox("Pilih Metrik untuk Visualisasi", metrics)
            
            if selected_metric in df_fundamental.columns:
                # Filter saham dengan data tersedia
                valid_data = df_fundamental.dropna(subset=[selected_metric])
                
                if not valid_data.empty:
                    # Buat grafik perbandingan
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(valid_data['Stock'], valid_data[selected_metric])
                    
                    # Tambahkan label nilai
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}', ha='center', va='bottom')
                    
                    ax.set_title(f"Perbandingan {selected_metric}")
                    ax.set_ylabel(selected_metric)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Analisis sektoral
                    st.subheader("Analisis Sektoral")
                    
                    # Hitung rata-rata sektor
                    sector_avg = valid_data.groupby('Sektor')[selected_metric].mean().reset_index()
                    
                    # Gabungkan dengan data saham
                    comparison = valid_data[['Stock', selected_metric, 'Sektor']].merge(
                        sector_avg, on='Sektor', suffixes=('', '_Sektor')
                    )
                    
                    # Hitung deviasi dari rata-rata sektor
                    comparison[f'Deviasi {selected_metric}'] = (
                        comparison[selected_metric] - comparison[selected_metric + '_Sektor']
                    )
                    
                    # Tampilkan hasil
                    st.write(f"Rata-rata {selected_metric} per Sektor:")
                    st.dataframe(sector_avg)
                    
                    st.write("Perbandingan dengan Rata-Rata Sektor:")
                    st.dataframe(comparison.sort_values(f'Deviasi {selected_metric}', ascending=False))
                    
                    # Tampilkan rekomendasi valuasi
                    st.subheader("Rekomendasi Valuasi")
                    for _, row in comparison.iterrows():
                        deviation = row[f'Deviasi {selected_metric}']
                        if deviation > 0:
                            status = "DIATAS"
                            color = "red"
                            recommendation = "Pertimbangkan untuk jual jika overvalued"
                        elif deviation < 0:
                            status = "DIBAWAH"
                            color = "green"
                            recommendation = "Potensi beli jika undervalued"
                        else:
                            status = "SAMA"
                            color = "gray"
                            recommendation = "Nilai wajar"
                        
                        st.markdown(
                            f"**{row['Stock']}**: {status} rata-rata sektor " 
                            f"({deviation:.2f}) - <span style='color:{color}'>"
                            f"{recommendation}</span>", 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning(f"Tidak ada data {selected_metric} yang tersedia")
            else:
                st.warning(f"Metrik {selected_metric} tidak tersedia dalam data")

# Simpan histori portofolio
if portfolio_df is not None and uploaded_file is not None:
    history_dir = "portfolio_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_df.to_csv(f"{history_dir}/portfolio_{timestamp}.csv", index=False)
