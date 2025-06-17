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

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Portfolio Analysis Tool")
st.title("🪙 AI Portfolio Management Dashboard")

# Sidebar Menu
with st.sidebar:
    st.header("Menu Navigasi")
    menu_options = ["Portfolio Analysis", "Price Prediction", "What-If Simulation", "AI Recommendations", "Compound Interest"]
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
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist.reset_index()[['Date', 'Close']].rename(columns={'Close': 'price'})

# Fungsi prediksi dengan Prophet
def prophet_prediction(df, days):
    try:
        df_prophet = df.rename(columns={'Date': 'ds', 'price': 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'price'})
    except Exception as e:
        st.error(f"Prophet Error: {str(e)}")
        return None

# Fungsi prediksi dengan LSTM (DIPERBAIKI)
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
            
        # PERBAIKAN DI SINI (tambahkan penutup kurung)
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
    col1.metric("Total Value", f"${total_value:,.2f}")
    col2.metric("Total Investment", f"${total_investment:,.2f}")
    col3.metric("Total P/L", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
    col4.metric("Number of Stocks", len(portfolio))
    
    # Grafik komposisi portfolio
    st.subheader("Portfolio Composition")
    fig, ax = plt.subplots(figsize=(10, 6))
    portfolio.groupby('Stock')['Value'].sum().plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90)
    st.pyplot(fig)
    
    # Tabel kinerja saham
    st.subheader("Stock Performance")
    st.dataframe(portfolio.sort_values('P/L %', ascending=False).reset_index(drop=True))

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
    ax.set_title(f"Projection for {years} Years at {rate*100:.2f}% Annual Return")
    ax.set_xlabel("Months")
    ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)
    
    return projection

# Fungsi rekomendasi AI
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
            recommendation = "Sell"
            reason = "Profit target achieved"
        elif pl_pct < -15:
            recommendation = "Buy"
            reason = "Opportunity to average down"
        else:
            recommendation = "Hold"
            reason = "Neutral position"
        
        recommendations.append({
            'Stock': row['Stock'],
            'Current Price': current_price,
            'Avg Price': avg_price,
            'P/L %': pl_pct,
            'Recommendation': recommendation,
            'Reason': reason
        })
    
    return pd.DataFrame(recommendations)

# Main App Logic
portfolio_df = process_uploaded_file(uploaded_file)

if selected_menu == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    visualize_portfolio(portfolio_df)
    
elif selected_menu == "Price Prediction":
    st.header("Stock Price Prediction")
    
    if portfolio_df is not None:
        selected_stock = st.selectbox("Select Stock", portfolio_df['Stock'])
        selected_row = portfolio_df[portfolio_df['Stock'] == selected_stock].iloc[0]
        ticker = selected_row['Ticker']
        
        # Dapatkan data historis
        hist_data = get_stock_data(ticker)
        
        if hist_data is not None:
            st.subheader(f"Historical Price: {selected_stock} ({ticker})")
            st.line_chart(hist_data.set_index('Date'))
            
            # Pilih model prediksi
            model_option = st.selectbox("Select Prediction Model", 
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
            st.subheader("Price Predictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            hist_data.plot(x='Date', y='price', ax=ax, label='Historical')
            
            for model_name, pred in results.items():
                if pred is not None:
                    pred.plot(x='Date', y='price', ax=ax, label=f'{model_name} Prediction')
            
            ax.set_title(f"{selected_stock} Price Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            st.pyplot(fig)
            
            # Evaluasi model (jika ada data aktual untuk evaluasi)
            if model_option != "Ensemble":
                if results[model_option] is not None and len(hist_data) > 30:
                    # Ambil data terakhir untuk evaluasi
                    actual = hist_data[-prediction_days:]
                    predicted = results[model_option].iloc[:len(actual)]
                    
                    if len(actual) == len(predicted):
                        rmse = np.sqrt(mean_squared_error(actual['price'], predicted['price']))
                        st.metric(f"{model_option} RMSE", f"{rmse:.2f}")

elif selected_menu == "What-If Simulation":
    st.header("What-If Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Portfolio")
        visualize_portfolio(portfolio_df)
    
    with col2:
        st.subheader("Add New Investment")
        new_stock = st.text_input("Stock Name")
        new_ticker = st.text_input("Ticker Symbol")
        new_lots = st.number_input("Number of Lots", min_value=1, value=10)
        new_price = st.number_input("Price per Share ($)", min_value=0.01, value=100.0)
        
        if st.button("Simulate"):
            what_if_simulation(portfolio_df, new_stock, new_ticker, new_lots, new_price)

elif selected_menu == "AI Recommendations":
    st.header("AI Investment Recommendations")
    
    if portfolio_df is not None:
        recommendations = generate_recommendations(portfolio_df)
        st.dataframe(recommendations.style.applymap(
            lambda x: 'background-color: lightgreen' if x == 'Buy' else 
                     ('background-color: salmon' if x == 'Sell' else 'background-color: lightyellow'),
            subset=['Recommendation']
        ))

elif selected_menu == "Compound Interest":
    st.header("Compound Interest Projection")
    
    principal = st.number_input("Current Portfolio Value ($)", 
                               min_value=0.0, 
                               value=10000.0)
    monthly_add = st.number_input("Monthly Additional Investment ($)", 
                                 min_value=0.0, 
                                 value=500.0)
    years = st.slider("Years of Projection", 1, 50, 10)
    rate = st.slider("Expected Annual Return (%)", 0.0, 30.0, 8.0) / 100.0
    
    if st.button("Generate Projection"):
        compound_interest_projection(principal, monthly_add, years, rate)

# Simpan histori portofolio
if portfolio_df is not None and uploaded_file is not None:
    history_dir = "portfolio_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_df.to_csv(f"{history_dir}/portfolio_{timestamp}.csv", index=False)
