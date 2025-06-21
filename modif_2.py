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
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from textblob import TextBlob  # Untuk analisis sentimen
import requests  # Untuk mengambil data berita
from bs4 import BeautifulSoup  # Untuk parsing HTML

# Helper function untuk format mata uang
def format_currency_idr(value):
    """Format angka menjadi string mata uang IDR"""
    return f"Rp{value:,.0f}".replace(",", ".")

# Fungsi baru: Ambil data ESG (dummy data)
def get_esg_score(ticker):
    """Ambil data ESG dari sumber eksternal (dummy data)"""
    try:
        # Dummy ESG scores berdasarkan sektor
        sector_esg_scores = {
            "Technology": 7.5,
            "Healthcare": 8.0,
            "Financial Services": 6.0,
            "Consumer Defensive": 7.0,
            "Energy": 4.5,
            "Basic Materials": 5.0,
            "Industrials": 6.5,
            "Communication Services": 7.0
        }
        
        # Dummy green scores
        green_scores = {
            "Technology": 8.0,
            "Healthcare": 7.5,
            "Financial Services": 5.0,
            "Consumer Defensive": 6.0,
            "Energy": 3.0,
            "Basic Materials": 4.0,
            "Industrials": 5.5,
            "Communication Services": 7.0
        }
        
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', 'N/A')
        
        return {
            'ESG Score': sector_esg_scores.get(sector, 5.0),
            'Green Score': green_scores.get(sector, 5.0),
            'ESG Category': "High" if sector_esg_scores.get(sector, 5.0) > 7.0 
                            else "Medium" if sector_esg_scores.get(sector, 5.0) > 5.0 
                            else "Low",
            'Sektor': sector
        }
    except Exception as e:
        st.error(f"Error getting ESG data for {ticker}: {str(e)}")
        return None

# Fungsi baru: Ambil berita saham
def get_stock_news(ticker, max_news=5):
    """Ambil berita terbaru tentang saham dari Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
            
        parsed_news = []
        for item in news[:max_news]:
            title = item.get('title', '')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '#')
            
            # Analisis sentimen dengan TextBlob
            blob = TextBlob(title)
            sentiment = blob.sentiment.polarity
            
            # Kategorikan sentimen
            if sentiment > 0.1:
                sentiment_label = "Positive 😊"
            elif sentiment < -0.1:
                sentiment_label = "Negative 😠"
            else:
                sentiment_label = "Neutral 😐"
            
            parsed_news.append({
                'title': title,
                'publisher': publisher,
                'link': link,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment
            })
            
        return parsed_news
    except Exception as e:
        st.error(f"Error getting news for {ticker}: {str(e)}")
        return []

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
        
        # Ambil data ESG
        esg_data = get_esg_score(ticker)
        if esg_data:
            fundamental.update(esg_data)
        
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

# Fungsi baru: Ambil data dividen
def get_dividend_data(ticker):
    """Ambil data dividen historis dari Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        divs = stock.dividends
        
        if divs.empty:
            return None
        
        # Konversi ke DataFrame
        div_df = divs.reset_index()
        div_df.columns = ['Date', 'Dividend']
        
        # Hitung dividend yield tahunan
        div_df['Year'] = div_df['Date'].dt.year
        annual_div = div_df.groupby('Year')['Dividend'].sum().reset_index()
        
        # Dapatkan harga penutupan tahunan
        hist = stock.history(period="max")
        hist = hist.reset_index()[['Date', 'Close']]
        hist['Year'] = hist['Date'].dt.year
        annual_price = hist.groupby('Year')['Close'].last().reset_index()
        
        # Gabungkan data
        merged = annual_div.merge(annual_price, on='Year')
        merged['Dividend Yield'] = (merged['Dividend'] / merged['Close']) * 100
        
        return merged
    except Exception as e:
        st.error(f"Error getting dividend data for {ticker}: {str(e)}")
        return None

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Portfolio Analysis Tool")
st.title("🪙 AI Portfolio Management Dashboard")

# Sidebar Menu
with st.sidebar:
    st.header("Menu Navigasi")
    # Tambahkan menu "ESG & Berita"
    menu_options = ["Portfolio Analysis", "Price Prediction", "What-If Simulation", 
                "AI Recommendations", "Compound Interest", "Fundamental Analysis", 
                "Risk Analysis", "ESG & Berita"]  # Added "ESG & Berita"

    selected_menu = st.radio("Pilih Modul:", menu_options)
    
    st.divider()
    st.header("Upload Portfolio")
    uploaded_file = st.file_uploader("Upload file (CSV/Excel)", type=["csv", "xlsx"])
    
    st.divider()
    st.header("Parameter Analisis")
    prediction_days = st.slider("Jumlah Hari Prediksi", 7, 365, 30)
    risk_tolerance = st.select_slider("Toleransi Risiko", options=["Low", "Medium", "High"])

# ... (kode lainnya tetap sama, tidak berubah) ...
# Fungsi untuk memproses file upload
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
        )
        
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

# Fungsi visualisasi portfolio
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
    
    # Tampilkan metrik portfolio
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", format_currency_idr(total_value))
    col2.metric("Total Investment", format_currency_idr(total_investment))
    col3.metric("Total P/L", format_currency_idr(total_pl), f"{total_pl_pct:.2f}%")
    col4.metric("Number of Stocks", len(portfolio))
    
    # Grafik komposisi portfolio
    st.subheader("Portfolio Composition")
    fig = px.pie(portfolio, 
                 values='Value', 
                 names='Stock', 
                 title='Portfolio Composition by Value',
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel kinerja saham
    st.subheader("Stock Performance (P/L %)")
    portfolio_sorted = portfolio.sort_values('P/L %', ascending=False)
    fig = px.bar(portfolio_sorted, 
                 x='Stock', 
                 y='P/L %', 
                 color='P/L %',
                 color_continuous_scale='RdYlGn',
                 text=portfolio_sorted['P/L %'].apply(lambda x: f"{x:.2f}%"))
    fig.update_layout(xaxis_title='Stock', yaxis_title='P/L %')
    st.plotly_chart(fig, use_container_width=True)
    
    # Buat salinan untuk tampilan dengan format Rp
    portfolio_display = portfolio.copy()
    money_cols = ['Current Price', 'Avg Price', 'Value', 'Investment', 'P/L']
    for col in money_cols:
        portfolio_display[col] = portfolio_display[col].apply(format_currency_idr)
    
    st.dataframe(portfolio_display.sort_values('P/L %', ascending=False).reset_index(drop=True))

# Fungsi simulasi what-if
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

# Fungsi proyeksi bunga majemuk
def compound_interest_projection(principal, monthly_add, years, rate, dividend_yield=0.0):
    periods = years * 12
    values = []
    current = principal
    dividend_total = 0
    
    for month in range(periods):
        # Tambahkan return dari capital gain dan dividen
        monthly_return = rate / 12 + dividend_yield / 12
        current = current * (1 + monthly_return) + monthly_add
        values.append(current)
        
        # Hitung dividen bulanan
        dividend_total += current * (dividend_yield / 12)
    
    projection = pd.DataFrame({
        'Month': range(1, periods+1),
        'Value': values
    })
    
    st.subheader("Compound Interest Projection")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(projection['Month'], projection['Value'])
    ax.set_title(f"Proyeksi {years} Tahun dengan Return Tahunan {rate*100:.2f}% + Dividen {dividend_yield*100:.2f}%")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Nilai Portofolio (Rp)")
    st.pyplot(fig)
    
    # Tampilkan total dividen
    st.metric("Total Dividen Proyeksi", format_currency_idr(dividend_total))
    
    return projection

# Fungsi rekomendasi AI
def generate_recommendations(portfolio):
    if portfolio is None:
        return None
    
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
            
            # Visualisasi hasil dengan Plotly
            st.subheader("Prediksi Harga")
            fig = go.Figure()
            # Tambahkan data historis
            fig.add_trace(go.Scatter(
                x=hist_data['Date'],
                y=hist_data['price'],
                mode='lines',
                name='Historis',
                line=dict(color='blue')
            ))
            
            for model_name, pred in results.items():
                if pred is not None:
                    fig.add_trace(go.Scatter(
                        x=pred['Date'],
                        y=pred['price'],
                        mode='lines',
                        name=f'Prediksi {model_name}'
                    ))
            
            fig.update_layout(
                title=f"Prediksi Harga {selected_stock}",
                xaxis_title="Tanggal",
                yaxis_title="Harga (Rp)",
                legend_title="Model",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Evaluasi model
            if model_option != "Ensemble":
                if results[model_option] is not None and len(hist_data) > 30:
                    # Ambil data terakhir untuk evaluasi
                    actual = hist_data[-prediction_days:]
                    predicted = results[model_option].iloc[:len(actual)]
                    
                    if len(actual) == len(predicted):
                        rmse = np.sqrt(mean_squared_error(actual['price'], predicted['price']))
                        st.metric(f"{model_option} RMSE", f"{rmse:.2f}")

# Menu What-If Simulation
elif selected_menu == "What-If Simulation":
    st.header("Simulasi What-If")
    
    # Add tabs for different simulation types
    tab1, tab2 = st.tabs(["Tambahan Saham Baru", "Simulasi Average Down"])
    
    with tab1:
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
            
            if st.button("Simulasikan Penambahan Saham"):
                what_if_simulation(portfolio_df, new_stock, new_ticker, new_lots, new_price)
    
    with tab2:
        st.subheader("Simulasi Average Down")
        
        if portfolio_df is None or portfolio_df.empty:
            st.warning("Silakan upload portofolio terlebih dahulu")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Select stock to average down
                selected_stock = st.selectbox("Pilih Saham untuk Average Down", portfolio_df['Stock'])
                selected_row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
                
                # Get current price
                current_price = yf.Ticker(selected_row['Ticker']).history(period='1d')['Close'].iloc[-1]
                
                st.metric("Saham Terpilih", selected_stock)
                st.metric("Jumlah Saham Saat Ini", f"{selected_row['Shares']:,}")
                st.metric("Harga Rata-rata Saat Ini", format_currency_idr(selected_row['Avg Price']))
                st.metric("Harga Pasar Terkini", format_currency_idr(current_price))
                
                # Strategy selection
                strategy = st.selectbox("Pilih Strategi", 
                                      ["Beli saat turun X%", 
                                       "Beli bulanan tetap (DCA)", 
                                       "Beli bertahap (Pyramid)"])
                
                # Common parameters
                total_budget = st.number_input("Total Budget untuk Average Down (Rp)", 
                                             min_value=1000000, value=10000000, step=1000000)
                max_drop = st.slider("Maksimum Penurunan Harga (%)", 10, 70, 30)
            
            with col2:
                # Strategy-specific parameters
                if strategy == "Beli saat turun X%":
                    drop_percent = st.slider("Beli setiap penurunan (%)", 5, 20, 10)
                    buy_lots = st.number_input("Jumlah Lot per Pembelian", min_value=1, value=5)
                    st.caption("Strategi: Beli setiap harga turun X% dari harga terakhir")
                
                elif strategy == "Beli bulanan tetap (DCA)":
                    monthly_lots = st.number_input("Jumlah Lot per Bulan", min_value=1, value=10)
                    duration = st.slider("Durasi (bulan)", 1, 24, 6)
                    st.caption("Strategi: Beli jumlah tetap setiap bulan")
                
                elif strategy == "Beli bertahap (Pyramid)":
                    base_lots = st.number_input("Jumlah Lot Awal", min_value=1, value=5)
                    increase_factor = st.slider("Faktor Peningkatan", 1.0, 3.0, 1.5, step=0.1)
                    drop_percent = st.slider("Beli setiap penurunan (%)", 5, 20, 10)
                    st.caption("Strategi: Tambah jumlah lot saat harga turun lebih dalam")
                
                # Run simulation button
                if st.button("Jalankan Simulasi Average Down"):
                    # Initialize variables
                    starting_price = current_price
                    budget_used = 0
                    purchases = []
                    new_avg_price = selected_row['Avg Price']
                    new_shares = selected_row['Shares']
                    current_drop = 0
                    
                    # Strategy 1: Buy at every X% drop
                    if strategy == "Beli saat turun X%":
                        while current_drop < max_drop and budget_used < total_budget:
                            # Calculate next buy price
                            buy_price = starting_price * (1 - (current_drop + drop_percent)/100)
                            
                            # Calculate cost
                            cost = buy_price * (buy_lots * 100)
                            
                            # Check if we have enough budget
                            if budget_used + cost > total_budget:
                                break
                            
                            # Add purchase
                            purchases.append({
                                'Drop %': current_drop + drop_percent,
                                'Harga Beli': buy_price,
                                'Lot': buy_lots,
                                'Saham': buy_lots * 100,
                                'Biaya': cost
                            })
                            
                            # Update portfolio
                            new_avg_price = ((new_avg_price * new_shares) + (buy_price * buy_lots * 100)) / (new_shares + (buy_lots * 100))
                            new_shares += buy_lots * 100
                            budget_used += cost
                            current_drop += drop_percent
                    
                    # Strategy 2: Monthly DCA
                    elif strategy == "Beli bulanan tetap (DCA)":
                        monthly_cost = current_price * (monthly_lots * 100)
                        months = min(duration, int(total_budget / monthly_cost))
                        
                        for month in range(1, months + 1):
                            # Simulate price drop (random between 0 and max_drop)
                            drop = min(max_drop, np.random.uniform(0, max_drop))
                            buy_price = current_price * (1 - drop/100)
                            
                            # Add purchase
                            purchases.append({
                                'Bulan': month,
                                'Penurunan %': drop,
                                'Harga Beli': buy_price,
                                'Lot': monthly_lots,
                                'Saham': monthly_lots * 100,
                                'Biaya': buy_price * monthly_lots * 100
                            })
                            
                            # Update portfolio
                            new_avg_price = ((new_avg_price * new_shares) + (buy_price * monthly_lots * 100)) / (new_shares + (monthly_lots * 100))
                            new_shares += monthly_lots * 100
                            budget_used += buy_price * monthly_lots * 100
                    
                    # Strategy 3: Pyramid buying
                    elif strategy == "Beli bertahap (Pyramid)":
                        current_level = 1
                        current_drop = 0
                        
                        while current_drop < max_drop and budget_used < total_budget:
                            # Calculate next buy price and lot size
                            buy_price = starting_price * (1 - (current_drop + drop_percent)/100)
                            level_lots = base_lots * (increase_factor ** (current_level - 1))
                            
                            # Calculate cost
                            cost = buy_price * (level_lots * 100)
                            
                            # Check if we have enough budget
                            if budget_used + cost > total_budget:
                                break
                            
                            # Add purchase
                            purchases.append({
                                'Level': current_level,
                                'Drop %': current_drop + drop_percent,
                                'Harga Beli': buy_price,
                                'Lot': level_lots,
                                'Saham': level_lots * 100,
                                'Biaya': cost
                            })
                            
                            # Update portfolio
                            new_avg_price = ((new_avg_price * new_shares) + (buy_price * level_lots * 100)) / (new_shares + (level_lots * 100))
                            new_shares += level_lots * 100
                            budget_used += cost
                            current_drop += drop_percent
                            current_level += 1
                    
                    # Display results
                    if purchases:
                        # Create purchases dataframe
                        df_purchases = pd.DataFrame(purchases)
                        
                        # Calculate savings
                        original_cost = selected_row['Shares'] * selected_row['Avg Price']
                        new_cost = original_cost + budget_used
                        savings_per_share = selected_row['Avg Price'] - new_avg_price
                        total_savings = savings_per_share * selected_row['Shares']
                        
                        # Show summary metrics
                        st.success(f"Simulasi Selesai! Total pembelian: {format_currency_idr(budget_used)}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Harga Rata-rata Baru", 
                                    format_currency_idr(new_avg_price), 
                                    f"↓ {selected_row['Avg Price'] - new_avg_price:.2f}")
                        col2.metric("Total Saham Baru", f"{new_shares:,}", f"↑ {new_shares - selected_row['Shares']:,}")
                        col3.metric("Penghematan per Saham", 
                                   format_currency_idr(savings_per_share), 
                                   f"Total: {format_currency_idr(total_savings)}")
                        
                        # Show purchases table
                        st.subheader("Detail Pembelian")
                        st.dataframe(df_purchases)
                        
                        # Create chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot price drop
                        if strategy == "Beli bulanan tetap (DCA)":
                            x = df_purchases['Bulan']
                            x_label = "Bulan"
                        else:
                            x = df_purchases['Drop %']
                            x_label = "Penurunan Harga (%)"
                        
                        # Plot purchase points
                        ax.scatter(x, df_purchases['Harga Beli'], 
                                  s=df_purchases['Lot']*50, 
                                  c='red', alpha=0.7, label='Pembelian')
                        
                        # Plot original and new average price
                        ax.axhline(y=selected_row['Avg Price'], color='blue', 
                                  linestyle='--', label='Harga Rata Lama')
                        ax.axhline(y=new_avg_price, color='green', 
                                  linestyle='--', label='Harga Rata Baru')
                        
                        ax.set_title(f"Simulasi Average Down untuk {selected_stock}")
                        ax.set_xlabel(x_label)
                        ax.set_ylabel("Harga (Rp)")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        st.pyplot(fig)
                        
                        # Show updated portfolio
                        st.subheader("Portofolio Setelah Average Down")
                        updated_portfolio = portfolio_df.copy()
                        mask = updated_portfolio['Stock'] == selected_stock
                        updated_portfolio.loc[mask, 'Avg Price'] = new_avg_price
                        updated_portfolio.loc[mask, 'Lot Balance'] = new_shares / 100
                        updated_portfolio.loc[mask, 'Shares'] = new_shares
                        visualize_portfolio(updated_portfolio)
                    else:
                        st.warning("Tidak ada pembelian yang dilakukan. Budget tidak cukup atau parameter tidak valid.")

# Menu AI Recommendations
elif selected_menu == "AI Recommendations":
    st.header("Rekomendasi AI")
    
    if portfolio_df is not None:
        # Buat tab untuk jenis rekomendasi berbeda
        tab1, tab2 = st.tabs(["Rekomendasi Portofolio", "Rekomendasi Penambahan Sektor"])
        
        with tab1:
            recommendations = generate_recommendations(portfolio_df)
            st.dataframe(recommendations.style.applymap(
                lambda x: 'background-color: lightgreen' if x == 'Beli' else 
                         ('background-color: salmon' if x == 'Jual' else 'background-color: lightyellow'),
                subset=['Rekomendasi']
            ))
        
        with tab2:
            st.subheader("Rekomendasi Penambahan Saham Berdasarkan Sektor")
            
            # 1. Identifikasi sektor yang underweight/belum dimiliki
            # Hitung alokasi sektor saat ini
            sector_allocation = {}
            total_value = 0
            
            # Dapatkan data fundamental untuk menghitung alokasi sektor
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                fundamental = get_fundamental_data(ticker)
                if fundamental and 'Sektor' in fundamental:
                    sector = fundamental['Sektor']
                    current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                    value = row['Shares'] * current_price
                    
                    if sector not in sector_allocation:
                        sector_allocation[sector] = 0
                    sector_allocation[sector] += value
                    total_value += value
            
            # Konversi ke persentase
            for sector in sector_allocation:
                sector_allocation[sector] = (sector_allocation[sector] / total_value) * 100
            
            # Tampilkan alokasi sektor saat ini
            st.subheader("Alokasi Sektor Saat Ini")
            st.bar_chart(pd.DataFrame.from_dict(sector_allocation, orient='index', columns=['Alokasi (%)']))
            
            # Identifikasi sektor yang belum dimiliki atau underweight
            # Asumsikan target alokasi minimal 10% per sektor
            underweight_sectors = []
            for sector, allocation in sector_allocation.items():
                if allocation < 10:
                    underweight_sectors.append(sector)
            
            # Sektor yang belum dimiliki
            all_sectors = ["Financial Services", "Energy", "Consumer Defensive", 
                          "Healthcare", "Technology", "Basic Materials", 
                          "Communication Services", "Industrials"]
            missing_sectors = [s for s in all_sectors if s not in sector_allocation]
            
            # Gabungkan sektor yang direkomendasikan
            recommended_sectors = underweight_sectors + missing_sectors
            if not recommended_sectors:
                st.success("Portofolio Anda sudah terdiversifikasi dengan baik di semua sektor utama!")
                st.stop()
            
            st.subheader(f"Sektor yang Direkomendasikan: {', '.join(recommended_sectors)}")
            
            # 2. Berikan rekomendasi saham untuk sektor tersebut
            # Daftar saham unggulan per sektor (bisa diperluas)
            sector_top_stocks = {
                "Financial Services": ["BBRI.JK", "BMRI.JK", "BBNI.JK", "BJBR.JK", "BTPN.JK"],
                "Energy": ["TLKM.JK", "EXCL.JK", "PGAS.JK", "ADRO.JK", "PTBA.JK"],
                "Consumer Defensive": ["ICBP.JK", "UNVR.JK", "MYOR.JK", "ULTJ.JK", "WIKA.JK"],
                "Healthcare": ["SILO.JK", "KLBF.JK", "DVLA.JK", "KAEF.JK"],
                "Technology": ["GOTO.JK", "BBHI.JK", "DMMX.JK", "MTEL.JK"],
                "Basic Materials": ["ANTM.JK", "INCO.JK", "SMBR.JK", "ADMR.JK"],
                "Communication Services": ["EXCL.JK", "ISAT.JK", "FREN.JK"],
                "Industrials": ["ASII.JK", "GGRM.JK", "INCO.JK", "TKIM.JK"]
            }
            
            # Ambil data fundamental untuk saham yang direkomendasikan
            recommendations = []
            for sector in recommended_sectors:
                if sector in sector_top_stocks:
                    for ticker in sector_top_stocks[sector]:
                        # Skip jika sudah ada di portofolio
                        if ticker in portfolio_df['Ticker'].values:
                            continue
                            
                        fundamental = get_fundamental_data(ticker)
                        if fundamental:
                            # Pastikan memiliki data yang diperlukan
                            if fundamental.get('PER') and fundamental.get('Dividend Yield') and fundamental.get('ROE'):
                                recommendations.append({
                                    'Sektor': sector,
                                    'Kode Saham': ticker,
                                    'PER': fundamental['PER'],
                                    'Dividend Yield (%)': fundamental.get('Dividend Yield', 0) * 100,
                                    'ROE (%)': fundamental.get('ROE', 0) * 100,
                                    'Market Cap': format_currency_idr(fundamental.get('Market Cap', 0))
                                })
            
            if recommendations:
                df_rec = pd.DataFrame(recommendations)
                
                # Urutkan berdasarkan valuasi menarik (PER rendah) dan dividen tinggi
                df_rec = df_rec.sort_values(by=['PER', 'Dividend Yield (%)'], ascending=[True, False])
                
                st.subheader("Rekomendasi Saham Berdasarkan Valuasi & Dividen")
                st.dataframe(df_rec)
                
                # Berikan penjelasan analisis
                st.subheader("Analisis Rekomendasi")
                st.write("""
                **Kriteria seleksi:**
                - Saham dari sektor yang underweight/belum dimiliki
                - PER rendah (valuasi menarik)
                - Dividend Yield tinggi (potensi penghasilan pasif)
                - ROE tinggi (efisiensi penggunaan modal)
                
                **Cara menggunakan rekomendasi:**
                1. Pilih saham dengan PER < 15 dan Dividend Yield > 3% sebagai prioritas
                2. Pertimbangkan diversifikasi ke beberapa sektor
                3. Gunakan fitur What-If Simulation untuk menguji dampaknya pada portofolio
                """)
            else:
                st.warning("Tidak ditemukan rekomendasi saham untuk sektor yang dipilih")
    else:
        st.warning("Silakan upload portofolio terlebih dahulu")

# Menu Compound Interest
elif selected_menu == "Compound Interest":
    st.header("Proyeksi Bunga Majemuk")
    
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Nilai Portofolio Saat Ini (Rp)",
                                   min_value=0.0, 
                                   value=10000.0)
        monthly_add = st.number_input("Tambahan Investasi Bulanan (Rp)",
                                     min_value=0.0, 
                                     value=500.0)
        years = st.slider("Tahun Proyeksi", 1, 50, 10)
        rate = st.slider("Return Tahunan yang Diharapkan (%)", 0.0, 30.0, 8.0) / 100.0
        
    with col2:
        # Jika ada portofolio, hitung rata-rata dividend yield
        avg_dividend_yield = 0.0
        if portfolio_df is not None:
            dividend_yields = []
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                try:
                    stock = yf.Ticker(ticker)
                    dividend_yield = stock.info.get('dividendYield', 0.0) or 0.0
                    if dividend_yield > 0:
                        dividend_yields.append(dividend_yield)
                except:
                    pass
            
            if dividend_yields:
                avg_dividend_yield = np.mean(dividend_yields)
        
        dividend_yield = st.slider("Dividend Yield Tahunan (%)", 
                                 0.0, 15.0, 
                                 float(avg_dividend_yield * 100)) / 100.0
    
    if st.button("Proyeksikan"):
        compound_interest_projection(principal, monthly_add, years, rate, dividend_yield)

# Menu Fundamental Analysis
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
            metrics = ['PER', 'PBV', 'ROE', 'DER', 'Dividend Yield', 'Revenue Growth', 'ESG Score', 'Green Score']
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
            
            # ANALISIS DIVIDEN BARU
            st.divider()
            st.subheader("Analisis Dividen Saham")
            
            # Pilih saham untuk analisis dividen
            selected_stock = st.selectbox("Pilih Saham untuk Analisis Dividen", portfolio_df['Stock'])
            selected_row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
            ticker = selected_row['Ticker']
            shares = selected_row['Shares']
            
            # Dapatkan data dividen
            dividend_data = get_dividend_data(ticker)
            
            if dividend_data is None:
                st.warning(f"Tidak ada data dividen untuk {selected_stock}")
            else:
                # Tampilkan metrik utama
                st.subheader(f"Dividen {selected_stock} ({ticker})")
                
                # Hitung metrik penting
                current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                last_year = dividend_data.iloc[-1]
                current_yield = last_year['Dividend Yield']
                total_dividend = last_year['Dividend'] * shares
                
                # Hitung rata-rata 5 tahun
                avg_5y_yield = dividend_data['Dividend Yield'].tail(5).mean()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dividend Yield Terakhir", f"{current_yield:.2f}%")
                col2.metric("Total Dividen Terakhir", format_currency_idr(total_dividend))
                col3.metric("Rata-rata Dividend Yield 5 Tahun", f"{avg_5y_yield:.2f}%")
                col4.metric("Jumlah Saham", f"{shares:,}")
                
                # Tampilkan grafik historis
                st.subheader("Riwayat Dividen")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Grafik 1: Dividen per tahun
                ax1.bar(dividend_data['Year'], dividend_data['Dividend'])
                ax1.set_title("Dividen per Saham per Tahun")
                ax1.set_ylabel("Dividen (Rp)")
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Grafik 2: Dividend Yield
                ax2.plot(dividend_data['Year'], dividend_data['Dividend Yield'], 'o-')
                ax2.set_title("Dividend Yield per Tahun")
                ax2.set_ylabel("Dividend Yield (%)")
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Proyeksi dividen
                st.subheader("Proyeksi Dividen")
                projection = calculate_dividend_projection(dividend_data)
                
                if projection is not None:
                    # Hitung total dividen untuk proyeksi
                    projection['Total Dividen'] = projection['Projected Dividend'] * shares
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Proyeksi Dividen per Saham:")
                        st.dataframe(projection[['Year', 'Projected Dividend']].set_index('Year'))
                    
                    with col2:
                        st.write("Total Dividen untuk Portofolio:")
                        st.dataframe(projection[['Year', 'Total Dividen']].set_index('Year'))
                    
                    # Grafik proyeksi
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(projection['Year'], projection['Projected Dividend'])
                    ax.set_title("Proyeksi Dividen per Saham")
                    ax.set_xlabel("Tahun")
                    ax.set_ylabel("Dividen (Rp)")
                    st.pyplot(fig)

# Menu ESG & Berita
elif selected_menu == "ESG & Berita":
    st.header("Analisis ESG & Berita Saham")
    
    if portfolio_df is None:
        st.warning("Silakan upload portofolio terlebih dahulu")
    else:
        tab1, tab2 = st.tabs(["Analisis ESG", "Berita & Sentimen"])
        
        with tab1:
            st.subheader("Analisis ESG Portofolio")
            
            # Kumpulkan data ESG
            esg_data = []
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                esg = get_esg_score(ticker)
                if esg:
                    esg['Stock'] = row['Stock']
                    esg['Ticker'] = ticker
                    esg_data.append(esg)
            
            if not esg_data:
                st.warning("Tidak ada data ESG yang tersedia")
            else:
                df_esg = pd.DataFrame(esg_data)
                
                # Tampilkan tabel ESG
                st.dataframe(df_esg.set_index('Stock'))
                
                # Visualisasi ESG Score
                st.subheader("Perbandingan ESG Score")
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(df_esg['Stock'], df_esg['ESG Score'], color='green')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
                
                ax.axhline(y=7.0, color='gold', linestyle='--', label='Good ESG')
                ax.axhline(y=5.0, color='red', linestyle='--', label='Average ESG')
                ax.set_ylabel("ESG Score")
                ax.set_title("ESG Score per Saham")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Visualisasi Green Score
                st.subheader("Perbandingan Green Score")
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(df_esg['Stock'], df_esg['Green Score'], color='lightgreen')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
                
                ax.axhline(y=7.0, color='gold', linestyle='--', label='Good Green Score')
                ax.axhline(y=5.0, color='red', linestyle='--', label='Average Green Score')
                ax.set_ylabel("Green Score")
                ax.set_title("Green Score per Saham")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Analisis sektoral ESG
                st.subheader("Analisis Sektoral ESG")
                sector_esg = df_esg.groupby('Sektor')[['ESG Score', 'Green Score']].mean().reset_index()
                
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot ESG Score per sektor
                ax[0].bar(sector_esg['Sektor'], sector_esg['ESG Score'], color='seagreen')
                ax[0].set_title("Rata-rata ESG Score per Sektor")
                ax[0].set_ylabel("ESG Score")
                ax[0].tick_params(axis='x', rotation=45)
                
                # Plot Green Score per sektor
                ax[1].bar(sector_esg['Sektor'], sector_esg['Green Score'], color='lightgreen')
                ax[1].set_title("Rata-rata Green Score per Sektor")
                ax[1].set_ylabel("Green Score")
                ax[1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretasi ESG
                st.subheader("Interpretasi ESG Score")
                st.markdown("""
                - **ESG Score (8-10)**: Praktik ESG sangat baik, risiko rendah
                - **ESG Score (6-7.9)**: Praktik ESG cukup baik, risiko sedang
                - **ESG Score (4-5.9)**: Praktik ESG perlu perbaikan, risiko tinggi
                - **ESG Score (<4)**: Praktik ESG buruk, risiko sangat tinggi
                
                **Green Score** mengukur komitmen perusahaan terhadap lingkungan:
                - **Green Score (7-10)**: Praktik ramah lingkungan sangat baik
                - **Green Score (5-6.9)**: Praktik ramah lingkungan cukup
                - **Green Score (<5)**: Perlu perbaikan praktik lingkungan
                """)
                
        with tab2:
            st.subheader("Berita & Analisis Sentimen Saham")
            
            # Pilih saham
            selected_stock = st.selectbox("Pilih Saham", portfolio_df['Stock'])
            selected_row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
            ticker = selected_row['Ticker']
            
            # Dapatkan berita
            news = get_stock_news(ticker, max_news=10)
            
            if not news:
                st.warning(f"Tidak ada berita terkini untuk {selected_stock}")
            else:
                # Hitung distribusi sentimen
                sentiment_counts = {'Positive 😊': 0, 'Negative 😠': 0, 'Neutral 😐': 0}
                for item in news:
                    sentiment_counts[item['sentiment']] += 1
                
                # Tampilkan ringkasan sentimen
                col1, col2, col3 = st.columns(3)
                col1.metric("Berita Positif", sentiment_counts['Positive 😊'])
                col2.metric("Berita Netral", sentiment_counts['Neutral 😐'])
                col3.metric("Berita Negatif", sentiment_counts['Negative 😠'])
                
                # Grafik pie sentimen
                st.subheader("Distribusi Sentimen Berita")
                sentiment_df = pd.DataFrame({
                    'Sentimen': list(sentiment_counts.keys()),
                    'Jumlah': list(sentiment_counts.values())
                })
                
                fig = px.pie(sentiment_df, values='Jumlah', names='Sentimen',
                             color='Sentimen',
                             color_discrete_map={
                                 'Positive 😊': 'green',
                                 'Neutral 😐': 'gray',
                                 'Negative 😠': 'red'
                             })
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan berita dengan grouping berdasarkan sentimen
                st.subheader(f"Berita Terkini untuk {selected_stock}")
                
                # Tab untuk setiap jenis sentimen
                tab_positive, tab_neutral, tab_negative = st.tabs([
                    f"Positif 😊 ({sentiment_counts['Positive 😊']})",
                    f"Netral 😐 ({sentiment_counts['Neutral 😐']})",
                    f"Negatif 😠 ({sentiment_counts['Negative 😠']})"
                ])
                
                with tab_positive:
                    for item in news:
                        if item['sentiment'] == "Positive 😊":
                            st.markdown(f"### [{item['title']}]({item['link']})")
                            st.caption(f"Publisher: {item['publisher']}")
                            st.write(f"**Sentimen**: {item['sentiment']} (Score: {item['sentiment_score']:.2f})")
                            st.write("---")
                
                with tab_neutral:
                    for item in news:
                        if item['sentiment'] == "Neutral 😐":
                            st.markdown(f"### [{item['title']}]({item['link']})")
                            st.caption(f"Publisher: {item['publisher']}")
                            st.write(f"**Sentimen**: {item['sentiment']} (Score: {item['sentiment_score']:.2f})")
                            st.write("---")
                
                with tab_negative:
                    for item in news:
                        if item['sentiment'] == "Negative 😠":
                            st.markdown(f"### [{item['title']}]({item['link']})")
                            st.caption(f"Publisher: {item['publisher']}")
                            st.write(f"**Sentimen**: {item['sentiment']} (Score: {item['sentiment_score']:.2f})")
                            st.write("---")
                
                # Analisis tren sentimen
                st.subheader("Analisis Tren Sentimen")
                st.markdown("""
                - **Sentimen positif** biasanya terkait kinerja perusahaan yang baik, proyek baru, atau peningkatan laba
                - **Sentimen negatif** biasanya terkait masalah hukum, penurunan kinerja, atau faktor eksternal negatif
                - **Sentimen netral** biasanya berita faktual tanpa opini kuat
                
                **Rekomendasi**:
                - Pantau berita negatif untuk potensi risiko investasi
                - Manfaatkan berita positif sebagai konfirmasi keputusan investasi
                """)

# Add Risk Analysis menu option
elif selected_menu == "Risk Analysis":
    st.header("Analisis Risiko Portofolio")
    
    if portfolio_df is None:
        st.warning("Silakan upload portofolio terlebih dahulu")
    else:
        tickers = portfolio_df['Ticker'].tolist()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        st.subheader("Mengumpulkan Data Saham...")
        price_data = {}
        returns_data = {}
        
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
                if not stock_data.empty:
                    returns = stock_data['Adj Close'].pct_change().dropna()
                    price_data[ticker] = stock_data['Adj Close']
                    returns_data[ticker] = returns
            except Exception as e:
                st.error(f"Error retrieving data for {ticker}: {str(e)}")
        
        if not price_data:
            st.error("Tidak ada data yang berhasil diambil. Coba lagi nanti.")
            st.stop()
            
        # Bangun DataFrame dengan benar
        prices_df = pd.concat(price_data.values(), axis=1, keys=price_data.keys())
        returns_df = pd.concat(returns_data.values(), axis=1, keys=returns_data.keys())
        
        #######################################################
        # TAMBAHKAN DI SINI: Validasi struktur data
        #st.write("Struktur kolom prices_df:", prices_df.columns)
        #st.write("Level kolom prices_df:", prices_df.columns.nlevels)
        #st.write("Contoh nilai kolom:", prices_df.columns[:5])  # 5 kolom pertama
        #######################################################
        
        # PERBAIKAN DI SINI: Akses kolom yang benar untuk MultiIndex
        portfolio_df['Current Price'] = portfolio_df['Ticker'].apply(
            lambda x: prices_df[(x, 'Close')].iloc[-1]  # Akses kolom 'Close' secara eksplisit
            if (x, 'Close') in prices_df.columns 
            else np.nan
        )
        
        portfolio_df['Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
        total_value = portfolio_df['Value'].sum()
        portfolio_df['Weight'] = portfolio_df['Value'] / total_value
        
        # Create weights vector
        weights = portfolio_df.set_index('Ticker')['Weight']
        
        # Calculate individual volatilities (annualized)
        st.subheader("Volatilitas Saham")
        volatilities = returns_df.std() * np.sqrt(252)  # Annualized
        
        # PERBAIKAN DI SINI: Akses yang benar untuk volatilitas
        volatilities = volatilities.to_frame('Volatilitas Tahunan').reset_index()
        volatilities = volatilities.rename(columns={'index': 'Ticker'})
        volatilities = volatilities.sort_values('Volatilitas Tahunan', ascending=False)
        
        # Add portfolio weights
        volatilities['Weight'] = volatilities['Ticker'].map(weights)
        
        # Format for display
        display_vol = volatilities.copy()
        display_vol['Volatilitas Tahunan'] = display_vol['Volatilitas Tahunan'].apply(lambda x: f"{x:.2%}")
        display_vol['Weight'] = display_vol['Weight'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(display_vol)
        # Portfolio volatility
        cov_matrix = returns_df.cov() * 252  # Annualized covariance matrix
        
        # PERBAIKAN DI SINI: Konversi weights ke array untuk perhitungan
        weights_array = np.array(weights)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        col1, col2 = st.columns(2)
        col1.metric("Volatilitas Portofolio Tahunan", f"{portfolio_volatility:.2%}")
        col2.metric("Volatilitas Portofolio Harian", f"{(portfolio_volatility / np.sqrt(252)):.2%}")
        
        # Heatmap of correlations
        st.subheader("Heatmap Korelasi Saham")
        corr_matrix = returns_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns)
        
        # Annotate with correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                        ha="center", va="center", color="w")
        
        st.pyplot(fig)

        st.subheader("Risiko vs Return Saham")
        
        # Hitung return tahunan rata-rata
        annual_returns = returns_df.mean() * 252
        
        # Buat DataFrame untuk plot
        risk_return_df = pd.DataFrame({
            'Ticker': volatilities['Ticker'],
            'Volatility': volatilities['Volatilitas Tahunan'],
            'Return': annual_returns.values,
            'Weight': volatilities['Weight']
        })
        
        # PERBAIKAN: Bersihkan nilai NaN di kolom Weight
        risk_return_df['Weight'] = risk_return_df['Weight'].fillna(0)
        
        # Buat scatter plot interaktif
        fig = px.scatter(
            risk_return_df, 
            x='Volatility', 
            y='Return',
            text='Ticker',
            size='Weight',
            color='Ticker',
            title='Risiko vs Return Saham',
            hover_data=['Weight']
        )
        
        # Tambahkan garis referensi
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title='Volatilitas Tahunan',
            yaxis_title='Return Tahunan Rata-Rata'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Value at Risk (VaR) calculation
        st.subheader("Value at Risk (VaR)")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights_array).sum(axis=1)
        
        # Historical VaR
        confidence_level = st.slider("Tingkat Kepercayaan", 90, 99, 95)
        var_hist = -np.percentile(portfolio_returns, 100 - confidence_level)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = portfolio_returns.mean()
        var_param = -(mean_return - 1.645 * portfolio_returns.std())
        
        col1, col2 = st.columns(2)
        col1.metric(f"VaR Historis ({confidence_level}%)", f"{var_hist:.2%}")
        col2.metric("VaR Parametrik (95%)", f"{var_param:.2%}")
        
        st.caption("Catatan: VaR menunjukkan potensi kerugian maksimum pada tingkat kepercayaan tertentu")
        
        # Beta calculation
        st.subheader("Beta Saham (Risiko Sistematis)")
        
        # Download market index data (Jakarta Composite Index - ^JKSE)
        try:
            market_data = yf.download('^JKSE', start=start_date, end=end_date, auto_adjust=False)['Adj Close']
            market_returns = market_data.pct_change().dropna()
            
            # Align dates
            common_dates = returns_df.index.intersection(market_returns.index)
            returns_df_aligned = returns_df.loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]
            
            # Calculate beta for each stock
            betas = {}
            for ticker in returns_df_aligned.columns.levels[0]:
                stock_returns = returns_df_aligned[ticker].values
                cov = np.cov(stock_returns, market_returns_aligned)
                beta = cov[0, 1] / cov[1, 1]
                betas[ticker] = beta
                
            betas_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
            betas_df['Weight'] = weights
            
            # Format for display
            display_betas = betas_df.copy()
            display_betas['Beta'] = display_betas['Beta'].apply(lambda x: f"{x:.2f}")
            display_betas['Weight'] = display_betas['Weight'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_betas.sort_values('Beta', ascending=False))
            
            # Calculate portfolio beta
            portfolio_beta = (betas_df['Beta'] * betas_df['Weight']).sum()
            st.metric("Beta Portofolio", f"{portfolio_beta:.2f}")
            
            # Interpretation
            st.subheader("Interpretasi Beta")
            if portfolio_beta < 0.8:
                st.success("Portofolio Anda relatif defensif (beta < 0.8)")
                st.write("Portofolio ini cenderung kurang volatil dibanding pasar. "
                         "Saham-saham defensif biasanya lebih stabil selama penurunan pasar.")
            elif portfolio_beta > 1.2:
                st.warning("Portofolio Anda agresif (beta > 1.2)")
                st.write("Portofolio ini lebih volatil dibanding pasar. "
                         "Potensi return lebih tinggi, tetapi risiko juga lebih besar.")
            else:
                st.info("Portofolio Anda seimbang (beta antara 0.8-1.2)")
                st.write("Portofolio ini memiliki volatilitas yang sebanding dengan pasar.")
                
        except Exception as e:
            st.error(f"Gagal mengambil data indeks pasar: {str(e)}")
            st.warning("Beta tidak dapat dihitung tanpa data indeks pasar")

# Simpan histori portofolio
if portfolio_df is not None and uploaded_file is not None:
    history_dir = "portfolio_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_df.to_csv(f"{history_dir}/portfolio_{timestamp}.csv", index=False)