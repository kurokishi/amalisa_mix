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

# Sidebar Menu
with st.sidebar:
    st.header("Menu Navigasi")
    menu_options = ["Portfolio Analysis", "Price Prediction", "What-If Simulation", 
                    "AI Recommendations", "Compound Interest", "Fundamental Analysis"]
    selected_menu = st.radio("Pilih Modul:", menu_options)
    
    st.divider()
    st.header("Upload Portfolio")
    uploaded_file = st.file_uploader("Upload file (CSV/Excel)", type=["csv", "xlsx"])
    
    st.divider()
    st.header("Parameter Analisis")
    prediction_days = st.slider("Jumlah Hari Prediksi", 7, 365, 30)
    risk_tolerance = st.select_slider("Toleransi Risiko", options=["Low", "Medium", "High"])

# Fungsi untuk memproses file upload (PERBAIKAN INDENTASI)
def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung")
            return None
        
        # Validasi kolom
        required_cols = ["Stock", "Ticker", "Lot Balance", "Avg Price"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"File harus mengandung kolom: {', '.join(required_cols)}")
            return None
            
        df['Shares'] = df['Lot Balance'] * 100
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Fungsi untuk mendapatkan data harga
def get_stock_data(ticker, period='5y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Reset index dan konversi ke tz-naive
        hist = hist.reset_index()[['Date', 'Close']]
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        
        return hist.rename(columns={'Close': 'price'})
    except Exception as e:
        st.error(f"Error getting stock data for {ticker}: {str(e)}")
        return None

# Fungsi prediksi dengan Prophet
def prophet_prediction(df, days):
    try:
        # Buat salinan dataframe untuk menghindari modifikasi asli
        df_prophet = df.copy()
        
        # Ubah nama kolom
        df_prophet = df_prophet.rename(columns={'Date': 'ds', 'price': 'y'})
        
        # Hapus timezone dari kolom tanggal jika ada
        if df_prophet['ds'].dt.tz is not None:
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'price'})
    except Exception as e:
        st.error(f"Prophet Error: {str(e)}")
        return None

# Fungsi prediksi dengan LSTM
def lstm_prediction(df, days):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1,1))
        
        # Siapkan data training
        x_train, y_train = [], []
        sequence_length = 60
        
        for i in range(sequence_length, len(scaled_data)):
            x_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Bangun model LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Buat prediksi
        inputs = df['price'][-sequence_length:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.transform(inputs)
        
        future_predictions = []
        for _ in range(days):
            x_test = inputs[-sequence_length:]
            x_test = x_test.reshape(1, sequence_length, 1)
            pred = model.predict(x_test, verbose=0)
            future_predictions.append(pred[0,0])
            inputs = np.append(inputs, pred)
            
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1,1)
        
        last_date = df['Date'].iloc[-1]
        pred_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        
        return pd.DataFrame({'Date': pred_dates, 'price': future_predictions.flatten()})
    except Exception as e:
        st.error(f"LSTM Error: {str(e)}")
        return None

# Fungsi prediksi dengan XGBoost
def xgboost_prediction(df, days):
    try:
        df = df.copy()
        df.set_index('Date', inplace=True)
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Buat fitur
        for i in range(1, 61):
            df[f'lag_{i}'] = df['price'].shift(i)
            
        df.dropna(inplace=True)
        
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Latih model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        
        # Buat prediksi
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        future_df = pd.DataFrame(index=future_dates)
        future_df['day'] = future_df.index.day
        future_df['month'] = future_df.index.month
        future_df['year'] = future_df.index.year
        
        # Isi nilai lag
        current_data = df.tail(60)['price'].values
        for i in range(days):
            if i < len(current_data):
                future_df.loc[future_df.index[i], 'lag_1'] = current_data[-1-i]
            else:
                for lag in range(1, 61):
                    future_df.loc[future_df.index[i], f'lag_{lag}'] = future_df.loc[
                        future_df.index[i - lag], 'price'] if i >= lag else current_data[-lag]
            
            pred = model.predict(future_df.iloc[i:i+1].dropna(axis=1))
            future_df.loc[future_df.index[i], 'price'] = pred[0]
        
        return future_df.reset_index().rename(columns={'index': 'Date'})[['Date', 'price']]
    except Exception as e:
        st.error(f"XGBoost Error: {str(e)}")
        return None

# Fungsi visualisasi portfolio (DIPERBAIKI untuk Rp)
def visualize_portfolio(portfolio):
    if portfolio is None:
        return
    
    # Get latest prices
    portfolio['Current Price'] = portfolio['Ticker'].apply(
        lambda x: yf.Ticker(x).history(period='1d')['Close'].iloc[-1] if not pd.isna(x) else 0)
    
    portfolio['Value'] = portfolio['Shares'] * portfolio['Current Price']
    portfolio['Investment'] = portfolio['Shares'] * portfolio['Avg Price']
    portfolio['P/L'] = portfolio['Value'] - portfolio['Investment']
    portfolio['P/L %'] = (portfolio['P/L'] / portfolio['Investment']) * 100
    
    total_value = portfolio['Value'].sum()
    total_investment = portfolio['Investment'].sum()
    total_pl = total_value - total_investment
    total_pl_pct = (total_pl / total_investment) * 100 if total_investment > 0 else 0
    
    # Tampilkan metrik portfolio - UBAH KE Rp
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", format_currency_idr(total_value))
    col2.metric("Total Investment", format_currency_idr(total_investment))
    col3.metric("Total P/L", format_currency_idr(total_pl), f"{total_pl_pct:.2f}%")
    col4.metric("Number of Stocks", len(portfolio))
    
    # Grafik komposisi portfolio
    st.subheader("Portfolio Composition")
    fig, ax = plt.subplots(figsize=(10, 6))
    portfolio.groupby('Stock')['Value'].sum().plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90)
    st.pyplot(fig)
    
    # Tabel kinerja saham - UBAH KE Rp
    st.subheader("Stock Performance")
    
    # Buat salinan untuk tampilan dengan format Rp
    portfolio_display = portfolio.copy()
    money_cols = ['Current Price', 'Avg Price', 'Value', 'Investment', 'P/L']
    for col in money_cols:
        portfolio_display[col] = portfolio_display[col].apply(format_currency_idr)
    
    st.dataframe(portfolio_display.sort_values('P/L %', ascending=False).reset_index(drop=True))

# Fungsi simulasi what-if (DIPERBAIKI untuk Rp)
def what_if_simulation(portfolio, new_stock, new_ticker, new_lots, new_price):
    if portfolio is None:
        portfolio = pd.DataFrame(columns=['Stock', 'Ticker', 'Lot Balance', 'Avg Price', 'Shares'])
    
    new_row = {
        'Stock': new_stock,
        'Ticker': new_ticker,
        'Lot Balance': new_lots,
        'Avg Price': new_price,
        'Shares': new_lots * 100
    }
    
    new_portfolio = portfolio.append(new_row, ignore_index=True)
    visualize_portfolio(new_portfolio)
    return new_portfolio

# Fungsi proyeksi bunga majemuk (DIPERBAIKI untuk Rp)
def compound_interest_projection(principal, monthly_add, years, rate):
    periods = years * 12
    values = []
    current = principal
    
    for month in range(periods):
        current = current * (1 + rate/12) + monthly_add
        values.append(current)
    
    projection = pd.DataFrame({
        'Month': range(1, periods+1),
        'Value': values
    })
    
    st.subheader("Compound Interest Projection")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(projection['Month'], projection['Value'])
    ax.set_title(f"Proyeksi {years} Tahun dengan Return Tahunan {rate*100:.2f}%")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Nilai Portofolio (Rp)")
    st.pyplot(fig)
    
    return projection

# Fungsi rekomendasi AI (DIPERBAIKI untuk Rp)
def generate_recommendations(portfolio):
    if portfolio is None:
        return
    
    recommendations = []
    for _, row in portfolio.iterrows():
        ticker = row['Ticker']
        avg_price = row['Avg Price']
        current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        
        # Analisis sederhana
        pl_pct = (current_price - avg_price) / avg_price * 100
        
        if pl_pct > 25:
            recommendation = "Jual"
            reason = "Target profit tercapai"
        elif pl_pct < -15:
            recommendation = "Beli"
            reason = "Peluang average down"
        else:
            recommendation = "Tahan"
            reason = "Posisi netral"
        
        recommendations.append({
            'Saham': row['Stock'],
            'Harga Sekarang': format_currency_idr(current_price),
            'Harga Rata': format_currency_idr(avg_price),
            'P/L %': f"{pl_pct:.2f}%",
            'Rekomendasi': recommendation,
            'Alasan': reason
        })
    
    return pd.DataFrame(recommendations)

# Main App Logic
portfolio_df = process_uploaded_file(uploaded_file)

if selected_menu == "Portfolio Analysis":
    st.header("Analisis Portofolio")
    visualize_portfolio(portfolio_df)
    
elif selected_menu == "Price Prediction":
    st.header("Prediksi Harga Saham")
    
    if portfolio_df is not None:
        selected_stock = st.selectbox("Pilih Saham", portfolio_df['Stock'])
        selected_row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
        ticker = selected_row['Ticker']
        
        # Dapatkan data historis
        hist_data = get_stock_data(ticker)
        
        if hist_data is not None:
            st.subheader(f"Riwayat Harga: {selected_stock} ({ticker})")
            st.line_chart(hist_data.set_index('Date'))
            
            # Pilih model prediksi
            model_option = st.selectbox("Pilih Model Prediksi", 
                                      ["Prophet", "LSTM", "XGBoost", "Ensemble"])
            
            # Jalankan prediksi
            results = {}
            if model_option in ["Prophet", "Ensemble"]:
                results['Prophet'] = prophet_prediction(hist_data, prediction_days)
            if model_option in ["LSTM", "Ensemble"]:
                results['LSTM'] = lstm_prediction(hist_data, prediction_days)
            if model_option in ["XGBoost", "Ensemble"]:
                results['XGBoost'] = xgboost_prediction(hist_data, prediction_days)
            
            # Visualisasi hasil
            st.subheader("Prediksi Harga")
            fig, ax = plt.subplots(figsize=(12, 6))
            hist_data.plot(x='Date', y='price', ax=ax, label='Historis')
            
            for model_name, pred in results.items():
                if pred is not None:
                    pred.plot(x='Date', y='price', ax=ax, label=f'Prediksi {model_name}')
            
            ax.set_title(f"Prediksi Harga {selected_stock}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga (Rp)")
            st.pyplot(fig)
            
            # Evaluasi model
            if model_option != "Ensemble":
                if results[model_option] is not None and len(hist_data) > 30:
                    # Ambil data terakhir untuk evaluasi
                    actual = hist_data[-prediction_days:]
                    predicted = results[model_option].iloc[:len(actual)]
                    
                    if len(actual) == len(predicted):
                        rmse = np.sqrt(mean_squared_error(actual['price'], predicted['price']))
                        st.metric(f"{model_option} RMSE", f"{rmse:.2f}")

elif selected_menu == "What-If Simulation":
    st.header("Simulasi What-If")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portofolio Saat Ini")
        visualize_portfolio(portfolio_df)
    
    with col2:
        st.subheader("Tambahan Investasi Baru")
        new_stock = st.text_input("Nama Saham")
        new_ticker = st.text_input("Kode Saham")
        new_lots = st.number_input("Jumlah Lot", min_value=1, value=10)
        new_price = st.number_input("Harga per Saham (Rp)", min_value=0.01, value=100.0)
        
        if st.button("Simulasikan"):
            what_if_simulation(portfolio_df, new_stock, new_ticker, new_lots, new_price)

elif selected_menu == "AI Recommendations":
    st.header("Rekomendasi AI")
    
    if portfolio_df is not None:
        recommendations = generate_recommendations(portfolio_df)
        st.dataframe(recommendations.style.applymap(
            lambda x: 'background-color: lightgreen' if x == 'Beli' else 
                     ('background-color: salmon' if x == 'Jual' else 'background-color: lightyellow'),
            subset=['Rekomendasi']
        ))

elif selected_menu == "Compound Interest":
    st.header("Proyeksi Bunga Majemuk")
    
    principal = st.number_input("Nilai Portofolio Saat Ini (Rp)",
                               min_value=0.0, 
                               value=10000.0)
    monthly_add = st.number_input("Tambahan Investasi Bulanan (Rp)",
                                 min_value=0.0, 
                                 value=500.0)
    years = st.slider("Tahun Proyeksi", 1, 50, 10)
    rate = st.slider("Return Tahunan yang Diharapkan (%)", 0.0, 30.0, 8.0) / 100.0
    
    if st.button("Proyeksikan"):
        compound_interest_projection(principal, monthly_add, years, rate)

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
