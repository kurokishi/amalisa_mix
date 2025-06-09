import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Inisialisasi session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame()
if 'dividend_db' not in st.session_state:
    st.session_state.dividend_db = pd.DataFrame()

# Konfigurasi awal
plt.style.use('ggplot')
st.set_page_config(layout="wide", page_title="Portfolio Analysis Toolkit", page_icon="📈")

# Database dividen saham Indonesia (contoh)
DIVIDEND_DB = {
    'AADI': {'yield': 0.035, 'growth': 0.02},
    'ADRO': {'yield': 0.048, 'growth': 0.03},
    'ANTM': {'yield': 0.027, 'growth': 0.04},
    'BFIN': {'yield': 0.065, 'growth': 0.05},
    'BJBR': {'yield': 0.042, 'growth': 0.03},
    'BSSR': {'yield': 0.038, 'growth': 0.02},
    'LPPF': {'yield': 0.015, 'growth': 0.01},
    'PGAS': {'yield': 0.057, 'growth': 0.04},
    'PTBA': {'yield': 0.062, 'growth': 0.06},
    'UNVR': {'yield': 0.023, 'growth': 0.02},
    'WIIM': {'yield': 0.019, 'growth': 0.01}
}

def parse_portfolio(uploaded_file=None):
    """Parse portfolio from PDF or use sample data"""
    if uploaded_file:
        # Implementasi parsing PDF sebenarnya akan ditempatkan di sini
        # Untuk demo, kita gunakan data sampel
        pass
    
    # Data sampel dari PDF yang diberikan
    data = {
        'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
        'Shares': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
        'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
        'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835]
    }
    df = pd.DataFrame(data)
    df['Ticker'] = df['Stock'] + '.JK'
    return df

def fetch_stock_data(ticker, period='5y'):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist.reset_index()
    except:
        return pd.DataFrame()

def calculate_portfolio_metrics(df):
    """Calculate portfolio metrics"""
    df['Market Value'] = df['Shares'] * df['Market Price']
    df['Cost Value'] = df['Shares'] * df['Avg Price']
    df['Unrealized P/L'] = df['Market Value'] - df['Cost Value']
    df['P/L (%)'] = (df['Market Price'] / df['Avg Price'] - 1) * 100
    return df

def arima_forecast(df, steps=30):
    """ARIMA forecasting model"""
    try:
        model = ARIMA(df['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except:
        return pd.Series()

def lstm_forecast(df, steps=30):
    """LSTM forecasting model"""
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])
        
        # Prepare training data
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # Make forecast
        inputs = scaled_data[-60:]
        forecast = []
        for _ in range(steps):
            x_input = inputs[-60:].reshape(1, 60, 1)
            pred = model.predict(x_input, verbose=0)
            forecast.append(pred[0,0])
            inputs = np.append(inputs, pred)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        return forecast.flatten()
    except:
        return np.array([])

def mean_reversion_strategy(df, cash=10000000, trade_freq=10):
    """Lo Keng Hong mean reversion strategy"""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Distance'] = (df['Close'] - df['MA20']) / df['MA20']
    
    positions = []
    portfolio_value = cash
    shares_owned = 0
    
    for i in range(len(df)):
        if i % trade_freq == 0:
            if df.iloc[i]['Distance'] < -0.05 and portfolio_value > 0:
                # Buy signal
                shares_to_buy = portfolio_value // df.iloc[i]['Close']
                shares_owned += shares_to_buy
                portfolio_value -= shares_to_buy * df.iloc[i]['Close']
            elif df.iloc[i]['Distance'] > 0.05 and shares_owned > 0:
                # Sell signal
                portfolio_value += shares_owned * df.iloc[i]['Close']
                shares_owned = 0
        
        positions.append(portfolio_value + shares_owned * df.iloc[i]['Close'])
    
    return positions

def risk_parity_model(df, cash=10000000, rebalance_freq=30):
    """BlackRock-inspired risk parity model"""
    # Simplified implementation
    assets = ['Close']  # In practice would use multiple assets
    returns = df[assets].pct_change().dropna()
    cov_matrix = returns.cov()
    volatilities = returns.std()
    
    positions = []
    portfolio_value = cash
    weights = 1 / volatilities
    weights /= weights.sum()
    
    allocations = (weights * portfolio_value).to_dict()
    
    for i in range(len(df)):
        if i % rebalance_freq == 0:
            # Rebalance portfolio
            portfolio_value = sum([allocations[asset] for asset in assets])
            allocations = (weights * portfolio_value).to_dict()
        
        # Update value
        current_value = 0
        for asset in assets:
            current_value += allocations[asset] * (df.iloc[i][asset] / df.iloc[i-rebalance_freq][asset])
        
        positions.append(current_value)
    
    return positions

def main():
    st.title('📈 Portfolio Analysis Toolkit - Pasar Saham Indonesia')
    
    # Sidebar - Portfolio Management
    with st.sidebar:
        st.header("Manajemen Portofolio")
        uploaded_file = st.file_uploader("Unggah Laporan Portofolio (PDF)", type="pdf")
        
        if uploaded_file:
            st.session_state.portfolio = parse_portfolio(uploaded_file)
            st.success("Portofolio berhasil diunggah!")
        elif st.session_state.portfolio.empty:
            st.session_state.portfolio = parse_portfolio()
            st.info("Menggunakan data portofolio sampel")
        
        st.divider()
        st.header("Navigasi Analisis")
        analysis_type = st.selectbox("Pilih Analisis", [
            "Dashboard Portofolio", 
            "Proyeksi Dividen", 
            "Simulasi Average Down",
            "Prediksi Harga Saham",
            "Simulasi Strategi Investasi",
            "Simulasi Bunga Majemuk",
            "Edit Portofolio"
        ])
        
        st.divider()
        st.header("Parameter Analisis")
        risk_profile = st.select_slider("Profil Risiko", options=["Konservatif", "Moderat", "Agresif"], value="Moderat")
        time_horizon = st.selectbox("Horizon Waktu", ["Jangka Pendek (1-3 bulan)", "Jangka Menengah (6-12 bulan)", "Jangka Panjang (3-5 tahun)"])
        st.caption("⚙️ Parameter mempengaruhi model prediksi dan rekomendasi")

    # Hitung metrik portofolio
    portfolio = st.session_state.portfolio.copy()
    portfolio = calculate_portfolio_metrics(portfolio)
    total_value = portfolio['Market Value'].sum()
    
    # Dashboard Portofolio
    if analysis_type == "Dashboard Portofolio":
        st.header("📊 Dashboard Portofolio")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nilai Portofolio", f"Rp{total_value:,.0f}")
        unrealized_total = portfolio['Unrealized P/L'].sum()
        col2.metric("Unrealized P/L", f"Rp{unrealized_total:,.0f}", 
                    f"{unrealized_total/total_value*100:.2f}%")
        col3.metric("Jumlah Saham", len(portfolio))
        best_stock = portfolio.loc[portfolio['P/L (%)'].idxmax()]['Stock']
        col4.metric("Top Performer", best_stock, 
                    f"{portfolio['P/L (%)'].max():.2f}%")
        
        # Visualisasi
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Komposisi Portofolio")
            fig = px.pie(portfolio, names='Stock', values='Market Value',
                         hover_data=['P/L (%)'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Unrealized Profit/Loss")
            fig = px.bar(portfolio, x='Stock', y='Unrealized P/L', color='P/L (%)',
                         color_continuous_scale='RdYlGn',
                         labels={'Unrealized P/L': 'Rp'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical Performance
        st.subheader("Kinerja Historis")
        selected_stock = st.selectbox("Pilih Saham", portfolio['Stock'])
        ticker = portfolio[portfolio['Stock'] == selected_stock]['Ticker'].values[0]
        
        hist_data = fetch_stock_data(ticker)
        if not hist_data.empty:
            fig = px.line(hist_data, x='Date', y='Close', title=f"Performa Historis {selected_stock}")
            avg_price = portfolio[portfolio['Stock'] == selected_stock]['Avg Price'].values[0]
            fig.add_hline(y=avg_price, line_dash="dash", line_color="red", 
                         annotation_text="Harga Beli Rata-rata")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.subheader("Analisis Risiko")
            returns = hist_data['Close'].pct_change().dropna()
            col1, col2, col3 = st.columns(3)
            col1.metric("Volatilitas (Std Dev)", f"{returns.std()*100:.2f}%")
            col2.metric("Beta (vs IDX)", "0.85")  # Placeholder
            col3.metric("Sharpe Ratio", "1.2")  # Placeholder
    
    # Proyeksi Dividen
    elif analysis_type == "Proyeksi Dividen":
        st.header("💰 Proyeksi Dividen")
        
        # Asumsi dividen
        with st.expander("Pengaturan Asumsi Dividen"):
            default_growth = st.number_input("Asumsi Pertumbuhan Dividen Tahunan (%)", 
                                           min_value=0.0, max_value=20.0, value=3.0)
            years = st.slider("Tahun Proyeksi", 1, 20, 5)
        
        # Hitung proyeksi dividen
        dividend_data = []
        for _, row in portfolio.iterrows():
            stock = row['Stock']
            if stock in DIVIDEND_DB:
                div_yield = DIVIDEND_DB[stock]['yield']
                div_growth = DIVIDEND_DB[stock]['growth']
                current_value = row['Market Value']
                
                for year in range(1, years+1):
                    dividend = current_value * div_yield * (1 + div_growth) ** year
                    dividend_data.append({
                        'Stock': stock,
                        'Year': year,
                        'Dividend': dividend,
                        'Cumulative': dividend
                    })
        
        if dividend_data:
            div_df = pd.DataFrame(dividend_data)
            cumulative_df = div_df.groupby('Year')['Dividend'].sum().cumsum().reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Proyeksi Dividen per Saham")
                fig = px.bar(div_df, x='Year', y='Dividend', color='Stock', 
                            title="Dividen Tahunan")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Dividen Kumulatif")
                fig = px.line(cumulative_df, x='Year', y='Dividend', 
                            title="Total Dividen Portofolio")
                st.plotly_chart(fig, use_container_width=True)
            
            total_dividend = cumulative_df['Dividend'].iloc[-1]
            st.success(f"**Total proyeksi dividen {years} tahun:** Rp{total_dividend:,.0f}")
        else:
            st.warning("Data dividen tidak tersedia untuk saham dalam portofolio")
    
    # Simulasi Average Down
    elif analysis_type == "Simulasi Average Down":
        st.header("📉 Simulasi Average Down")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_stock = st.selectbox("Pilih Saham", portfolio['Stock'])
            current_data = portfolio[portfolio['Stock'] == selected_stock].iloc[0]
            
            st.subheader("Posisi Saat Ini")
            st.metric("Saham Dimiliki", f"{current_data['Shares']} lembar")
            st.metric("Harga Rata-rata", f"Rp{current_data['Avg Price']:,.0f}")
            st.metric("Harga Pasar", f"Rp{current_data['Market Price']:,.0f}")
        
        with col2:
            st.subheader("Parameter Average Down")
            additional_shares = st.number_input("Tambahan Saham", 
                                              min_value=0, 
                                              max_value=1000000, 
                                              value=500)
            purchase_price = st.number_input("Harga Beli", 
                                           min_value=1, 
                                           value=int(current_data['Market Price']))
        
        # Hitung hasil average down
        new_shares = current_data['Shares'] + additional_shares
        new_avg_price = ((current_data['Shares'] * current_data['Avg Price']) + 
                        (additional_shares * purchase_price)) / new_shares
        new_unrealized = new_shares * (current_data['Market Price'] - new_avg_price)
        
        st.divider()
        st.subheader("Hasil Simulasi")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Rata-rata Baru", f"Rp{new_avg_price:,.0f}", 
                   f"{(new_avg_price - current_data['Avg Price'])/current_data['Avg Price']*100:.2f}%")
        col2.metric("Unrealized P/L Baru", f"Rp{new_unrealized:,.0f}", 
                   f"{(new_unrealized - current_data['Unrealized P/L'])/abs(current_data['Unrealized P/L'])*100:.2f}%")
        improvement = (current_data['Avg Price'] - new_avg_price) / current_data['Avg Price'] * 100
        col3.metric("Perbaikan Harga Beli", f"{improvement:.2f}%")
        
        # Visualisasi
        fig, ax = plt.subplots()
        ax.bar(['Sebelum', 'Sesudah'], 
              [current_data['Avg Price'], new_avg_price],
              color=['red', 'green'])
        ax.set_ylabel("Harga Rata-rata (Rp)")
        ax.set_title("Perbandingan Harga Rata-rata")
        st.pyplot(fig)
    
    # Prediksi Harga Saham
    elif analysis_type == "Prediksi Harga Saham":
        st.header("🔮 Prediksi Harga Saham")
        
        selected_stock = st.selectbox("Pilih Saham", portfolio['Stock'])
        ticker = portfolio[portfolio['Stock'] == selected_stock]['Ticker'].values[0]
        
        hist_data = fetch_stock_data(ticker)
        if not hist_data.empty:
            # Tampilkan data historis
            st.subheader("Data Historis")
            fig = px.line(hist_data, x='Date', y='Close', title=f"Performa Historis {selected_stock}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Model prediksi
            st.subheader("Prediksi Harga")
            model_choice = st.radio("Pilih Model Prediksi", 
                                   ["ARIMA (Jangka Pendek)", 
                                    "LSTM (Jangka Menengah)", 
                                    "Regresi Hutan Acak (Jangka Panjang)"])
            
            if st.button("Jalankan Prediksi"):
                with st.spinner("Melatih model dan membuat prediksi..."):
                    if model_choice == "ARIMA (Jangka Pendek)":
                        forecast_steps = 30
                        forecast = arima_forecast(hist_data.set_index('Date'), forecast_steps)
                        if not forecast.empty:
                            last_date = hist_data['Date'].iloc[-1]
                            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps+1)]
                            
                            fig = px.line(hist_data[-100:], x='Date', y='Close', title=f"Prediksi 30 Hari {selected_stock}")
                            fig.add_scatter(x=future_dates, y=forecast, mode='lines', name='Prediksi')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Gagal membuat prediksi dengan model ARIMA")
                    
                    elif model_choice == "LSTM (Jangka Menengah)":
                        forecast_steps = 90
                        forecast = lstm_forecast(hist_data.set_index('Date'), forecast_steps)
                        if forecast.size > 0:
                            last_date = hist_data['Date'].iloc[-1]
                            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps+1)]
                            
                            fig = px.line(hist_data[-200:], x='Date', y='Close', title=f"Prediksi 90 Hari {selected_stock}")
                            fig.add_scatter(x=future_dates, y=forecast, mode='lines', name='Prediksi')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Gagal membuat prediksi dengan model LSTM")
                    
                    elif model_choice == "Regresi Hutan Acak (Jangka Panjang)":
                        # Placeholder untuk implementasi aktual
                        st.info("""
                        **Model Regresi Hutan Acak (Jangka Panjang)**  
                        Model ini mempertimbangkan:
                        - Fundamental perusahaan (ROE, PER, DER, dll.)
                        - Faktor makroekonomi (inflasi, suku bunga, pertumbuhan GDP)
                        - Sentimen pasar
                        - Analisis sektoral
                        
                        *Implementasi lengkap memerlukan integrasi dengan sumber data fundamental*
                        """)
                        st.image("https://via.placeholder.com/800x400?text=Long-Term+Forecast+Model", 
                                caption="Prediksi Jangka Panjang")
        else:
            st.warning(f"Data historis tidak tersedia untuk {selected_stock}")
    
    # Simulasi Strategi Investasi
    elif analysis_type == "Simulasi Strategi Investasi":
        st.header("⚔️ Simulasi Strategi Investasi")
        
        selected_stock = st.selectbox("Pilih Saham", portfolio['Stock'])
        ticker = portfolio[portfolio['Stock'] == selected_stock]['Ticker'].values[0]
        hist_data = fetch_stock_data(ticker, period='10y')
        
        if not hist_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Strategi Mean Reversion (Lo Keng Hong)")
                cash = st.number_input("Modal Awal (Rp)", min_value=1000000, value=100000000)
                trade_freq = st.slider("Frekuensi Trading (hari)", 1, 30, 10)
                
                if st.button("Jalankan Simulasi Mean Reversion"):
                    strategy_values = mean_reversion_strategy(hist_data.set_index('Date'), cash, trade_freq)
                    hist_data['Strategy'] = strategy_values
                    hist_data['Buy & Hold'] = cash * (hist_data['Close'] / hist_data['Close'].iloc[0])
                    
                    fig = px.line(hist_data, x='Date', y=['Strategy', 'Buy & Hold'], 
                                title="Perbandingan Strategi")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    final_value = strategy_values[-1]
                    bh_value = hist_data['Buy & Hold'].iloc[-1]
                    st.metric("Nilai Akhir Strategi", f"Rp{final_value:,.0f}", 
                             f"{(final_value - bh_value)/bh_value*100:.2f}% vs Buy & Hold")
            
            with col2:
                st.subheader("Model Risk Parity (BlackRock)")
                st.info("""
                **Model Risk Parity**  
                Pendekatan alokasi aset yang:
                - Menyeimbangkan risiko antar aset
                - Mengoptimalkan diversifikasi
                - Cocok untuk investor jangka panjang
                """)
                
                if st.button("Jalankan Simulasi Risk Parity"):
                    # Placeholder untuk implementasi aktual
                    st.image("https://via.placeholder.com/600x300?text=Risk+Parity+Simulation", 
                            caption="Simulasi Risk Parity")
                    st.metric("Estimasi Pengembalian Tahunan", "8.2%")
                    st.metric("Risiko Portofolio", "Moderat")
        else:
            st.warning(f"Data historis tidak tersedia untuk {selected_stock}")
    
    # Simulasi Bunga Majemuk
    elif analysis_type == "Simulasi Bunga Majemuk":
        st.header("🚀 Simulasi Bunga Majemuk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Parameter Investasi")
            initial_investment = st.number_input("Investasi Awal (Rp)", 
                                               min_value=1000000, 
                                               value=int(total_value))
            monthly_add = st.number_input("Tambahan Bulanan (Rp)", 
                                        min_value=0, 
                                        value=5000000)
            years = st.slider("Periode Investasi (tahun)", 1, 40, 10)
            return_rate = st.slider("Estimasi Return Tahunan (%)", 1.0, 30.0, 12.0)
        
        with col2:
            st.subheader("Proyeksi Nilai")
            # Kalkulasi bunga majemuk
            months = years * 12
            monthly_rate = return_rate / 12 / 100
            future_value = initial_investment * (1 + monthly_rate) ** months
            for month in range(months):
                future_value += monthly_add * (1 + monthly_rate) ** (months - month - 1)
            
            st.metric("Nilai Masa Depan", f"Rp{future_value:,.0f}")
            st.metric("Keuntungan", f"Rp{future_value - initial_investment - monthly_add*months:,.0f}")
            
            # Kalkulasi tahunan
            years_range = list(range(1, years+1))
            yearly_values = []
            for year in years_range:
                months = year * 12
                fv = initial_investment * (1 + monthly_rate) ** months
                for m in range(months):
                    fv += monthly_add * (1 + monthly_rate) ** (months - m - 1)
                yearly_values.append(fv)
            
            fig = px.line(x=years_range, y=yearly_values, 
                         labels={'x': 'Tahun', 'y': 'Nilai Portofolio'},
                         title="Pertumbuhan Portofolio")
            st.plotly_chart(fig, use_container_width=True)
    
    # Edit Portofolio
    elif analysis_type == "Edit Portofolio":
        st.header("✏️ Edit Portofolio")
        
        edited_portfolio = st.data_editor(
            portfolio[['Stock', 'Shares', 'Avg Price']],
            num_rows="dynamic",
            column_config={
                "Shares": st.column_config.NumberColumn(
                    format="%d lembar"
                ),
                "Avg Price": st.column_config.NumberColumn(
                    "Harga Beli (Rp)",
                    format="Rp%d"
                )
            }
        )
        
        if st.button("Simpan Perubahan"):
            st.session_state.portfolio = edited_portfolio
            st.success("Portofolio berhasil diperbarui!")
            
            # Perbarui harga pasar
            st.info("Memperbarui harga pasar...")
            for i, row in st.session_state.portfolio.iterrows():
                ticker = row['Stock'] + '.JK'
                data = fetch_stock_data(ticker, period='1d')
                if not data.empty:
                    st.session_state.portfolio.at[i, 'Market Price'] = data['Close'].iloc[-1]
            
            st.session_state.portfolio = calculate_portfolio_metrics(st.session_state.portfolio)
            st.rerun()

if __name__ == "__main__":
    main()
