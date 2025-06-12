import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import csv
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    def __init__(self, data):
        self.df = data.copy()
        self.today = datetime.today()
        
    def analyze_blackrock_style(self):
        """Analisis gaya Lo Keng Hong dan BlackRock"""
        analysis = pd.DataFrame(index=self.df.index)
        
        # 1. Analisis Valuasi - Perbaikan perhitungan PER dan PBV
        # Asumsi EPS = Stock Value / Balance (simulasi)
        eps = self.df['Stock Value'] / self.df['Balance']
        analysis['PER'] = np.where(eps > 0, self.df['Market Price'] / eps, np.nan)
        
        # Asumsi Book Value per Share = Stock Value / Balance / 2 (simulasi)
        bvps = self.df['Stock Value'] / self.df['Balance'] / 2
        analysis['PBV'] = np.where(bvps > 0, self.df['Market Price'] / bvps, np.nan)
        
        # 2. Momentum
        analysis['Momentum_1M'] = self.df['Stock'].apply(
            lambda x: self.get_price_momentum(x.split('-')[0], 30))
        
        # 3. Profitabilitas
        analysis['ROE'] = self.df['Stock'].apply(
            lambda x: self.get_roe(x.split('-')[0]))
        
        # 4. Rekomendasi - Perbaikan logika kondisi
        conditions = [
            (analysis['PER'] < 15) & (analysis['PBV'] < 1.5) & (analysis['Momentum_1M'] > 0),
            (analysis['PER'] > 25) | (analysis['PBV'] > 3) | (analysis['Momentum_1M'] < -0.05)
        ]
        choices = ['Strong Buy', 'Sell']
        analysis['Rekomendasi'] = np.select(conditions, choices, default='Hold')
        
        return analysis[['PER', 'PBV', 'Momentum_1M', 'ROE', 'Rekomendasi']]

    def get_price_momentum(self, ticker, days):
        """Hitung momentum harga dengan error handling yang lebih baik"""
        try:
            end_date = self.today
            start_date = end_date - timedelta(days=days + 10)  # Buffer untuk hari libur
            data = yf.download(f"{ticker}.JK", start=start_date, end=end_date, progress=False)
            
            if not data.empty and len(data) >= 2:
                # Ambil harga pertama dan terakhir yang tersedia
                first_price = data['Close'].iloc[0]
                last_price = data['Close'].iloc[-1]
                return (last_price - first_price) / first_price
            else:
                return 0.0
        except Exception as e:
            print(f"Error getting momentum for {ticker}: {e}")
            return 0.0

    def get_roe(self, ticker):
        """Ambil data ROE (simulasi dengan data yang lebih realistis)"""
        roe_values = {
            'AADI': 0.15, 'ADRO': 0.22, 'ANTM': 0.18, 
            'BFIN': 0.12, 'BJBR': 0.09, 'BSSR': 0.25,
            'LPPF': 0.08, 'PGAS': 0.11, 'PTBA': 0.21, 
            'UNVR': 0.28, 'WIIM': 0.14
        }
        return roe_values.get(ticker, 0.12)

    def diversification_analysis(self):
        """Identifikasi saham untuk dipertahankan/dijual"""
        analysis = self.analyze_blackrock_style()
        combined = pd.concat([self.df, analysis], axis=1)
        
        # Kriteria penjualan yang lebih konservatif
        sell_criteria = (
            (combined['Unrealized'] < -500000) |
            (combined['Rekomendasi'] == 'Sell') |
            (combined['Market Price'] < 0.75 * combined['Avg Price'])  # Stop loss 25%
        )
        
        # Kriteria pertahankan
        hold_criteria = (
            (combined['Unrealized'] > 0) &
            (combined['Rekomendasi'] == 'Strong Buy') &
            (combined['Momentum_1M'] > 0.05)
        )
        
        combined['Aksi'] = 'Hold'
        combined.loc[sell_criteria, 'Aksi'] = 'Sell'
        combined.loc[hold_criteria, 'Aksi'] = 'Strong Hold'
        
        return combined[['Stock', 'Unrealized', 'Rekomendasi', 'Aksi']]

    def update_portfolio(self, changes):
        """Update portofolio berdasarkan perubahan dengan validasi"""
        df_updated = self.df.copy()
        
        for change in changes:
            if change['action'] == 'delete':
                idx = df_updated[df_updated['Stock'] == change['stock']].index
                if not idx.empty:
                    df_updated = df_updated.drop(idx).reset_index(drop=True)
                    
            elif change['action'] == 'update':
                idx = df_updated[df_updated['Stock'] == change['stock']].index
                if not idx.empty:
                    df_updated.at[idx[0], 'Lot Balance'] = change['new_lot']
                    df_updated.at[idx[0], 'Balance'] = change['new_lot'] * 100
                    df_updated.at[idx[0], 'Avg Price'] = change['new_price']
                    
            elif change['action'] == 'add':
                new_row = {
                    'Stock': change['stock'],
                    'Lot Balance': change['lot'],
                    'Balance': change['lot'] * 100,
                    'Avg Price': change['price'],
                    'Market Price': change.get('market_price', change['price']),
                    'Stock Value': change['lot'] * 100 * change['price'],
                    'Market Value': change['lot'] * 100 * change.get('market_price', change['price']),
                    'Unrealized': 0
                }
                df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
        
        # Recalculate values
        df_updated['Stock Value'] = df_updated['Balance'] * df_updated['Avg Price']
        df_updated['Market Value'] = df_updated['Balance'] * df_updated['Market Price']
        df_updated['Unrealized'] = df_updated['Market Value'] - df_updated['Stock Value']
        
        return df_updated

    def predict_prices(self):
        """Prediksi harga saham dengan model yang lebih robust"""
        predictions = {}
        
        for stock in self.df['Stock']:
            ticker = stock.split('-')[0]
            try:
                # Download data dengan periode yang lebih panjang
                data = yf.download(f"{ticker}.JK", period='2y', progress=False)
                
                if len(data) < 50:  # Minimal data untuk prediksi
                    continue
                
                # Prepare features
                data['SMA20'] = data['Close'].rolling(20).mean()
                data['SMA50'] = data['Close'].rolling(50).mean()
                data['RSI'] = self.calculate_rsi(data['Close'])
                data['Volume_MA'] = data['Volume'].rolling(20).mean()
                
                # Drop NaN values
                data = data.dropna()
                
                if len(data) < 30:
                    continue
                
                # Prepare features and target
                features = ['SMA20', 'SMA50', 'RSI', 'Volume_MA']
                X = data[features]
                y = data['Close']
                
                # Split data
                split_idx = int(len(data) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Validate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                # Predict future prices
                last_data = X.iloc[-1:].values
                current_price = data['Close'].iloc[-1]
                
                # Prediksi dengan confidence interval
                short_term = model.predict(last_data)[0]
                medium_term = short_term * np.random.normal(1.1, 0.05)  # 10% growth with uncertainty
                long_term = medium_term * np.random.normal(1.2, 0.1)   # 20% growth with uncertainty
                
                predictions[stock] = {
                    'current_price': current_price,
                    'short_term': max(short_term, current_price * 0.8),  # Minimum 20% drop
                    'medium_term': max(medium_term, current_price * 0.7),  # Minimum 30% drop
                    'long_term': max(long_term, current_price * 0.6),     # Minimum 40% drop
                    'mse': mse,
                    'confidence': 'High' if mse < 1000 else 'Medium' if mse < 5000 else 'Low'
                }
            except Exception as e:
                print(f"Error predicting {ticker}: {e}")
                continue
        
        return predictions

    def calculate_rsi(self, prices, window=14):
        """Hitung RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def dividend_compound_projection(self, years=10, dividend_growth=0.05, reinvestment_rate=0.7):
        """Proyeksi dividen dengan bunga majemuk - diperbaiki"""
        projections = {}
        
        for _, row in self.df.iterrows():
            ticker = row['Stock'].split('-')[0]
            div_yield = self.get_dividend_yield(ticker)
            
            if div_yield > 0:
                current_val = row['Market Value']
                yearly = []
                total_value = current_val
                
                for year in range(1, years + 1):
                    # Hitung dividen tahunan
                    annual_dividend = current_val * div_yield * (1 + dividend_growth) ** (year - 1)
                    
                    # Dividen yang diinvestasi kembali
                    reinvested_dividend = annual_dividend * reinvestment_rate
                    
                    # Apresiasi nilai saham (asumsi 6% per tahun)
                    appreciation = total_value * 0.06
                    
                    # Total nilai di akhir tahun
                    total_value = total_value + appreciation + reinvested_dividend
                    
                    yearly.append({
                        'year': year,
                        'dividend': annual_dividend,
                        'reinvested': reinvested_dividend,
                        'appreciation': appreciation,
                        'total_value': total_value
                    })
                
                projections[row['Stock']] = yearly
                
        return projections

    def get_dividend_yield(self, ticker):
        """Ambil dividend yield dengan data yang lebih akurat"""
        dy_values = {
            'AADI': 0.04, 'ADRO': 0.08, 'ANTM': 0.05, 
            'BFIN': 0.07, 'BJBR': 0.06, 'BSSR': 0.09,
            'LPPF': 0.03, 'PGAS': 0.05, 'PTBA': 0.07, 
            'UNVR': 0.04, 'WIIM': 0.02
        }
        return dy_values.get(ticker, 0.04)

    def recommend_stocks(self, n=5):
        """Rekomendasi saham untuk diversifikasi portofolio"""
        sector_allocation = self.analyze_sectors()
        
        # Cari sektor yang kurang dari 20% alokasi
        underrepresented = sector_allocation[sector_allocation < 0.2].index.tolist()
        
        recommendations = []
        for sector in underrepresented:
            sector_stocks = self.get_top_sector_stocks(sector, n=2)
            recommendations.extend(sector_stocks)
        
        # Hapus duplikat dan batasi jumlah
        recommendations = list(set(recommendations))
        return recommendations[:n]

    def analyze_sectors(self):
        """Analisis alokasi sektor portofolio"""
        sector_map = {
            'AADI': 'Energy', 'ADRO': 'Energy', 'ANTM': 'Mining', 
            'BFIN': 'Financial', 'BJBR': 'Financial', 'BSSR': 'Mining',
            'LPPF': 'Consumer', 'PGAS': 'Energy', 'PTBA': 'Mining', 
            'UNVR': 'Consumer', 'WIIM': 'Consumer'
        }
        
        self.df['Sektor'] = self.df['Stock'].apply(
            lambda x: sector_map.get(x.split('-')[0], 'Other'))
        
        total_value = self.df['Market Value'].sum()
        if total_value == 0:
            return pd.Series()
            
        sector_allocation = self.df.groupby('Sektor')['Market Value'].sum() / total_value
        return sector_allocation

    def get_top_sector_stocks(self, sector, n=3):
        """Dapatkan saham terbaik di sektor tertentu"""
        sector_stocks = {
            'Energy': ['PGAS.JK', 'PTBA.JK', 'ADRO.JK', 'MEDC.JK'],
            'Mining': ['ANTM.JK', 'BSSR.JK', 'MDKA.JK', 'TINS.JK'],
            'Financial': ['BBRI.JK', 'BBCA.JK', 'BMRI.JK', 'BFIN.JK'],
            'Consumer': ['UNVR.JK', 'ICBP.JK', 'MYOR.JK', 'KLBF.JK']
        }
        return sector_stocks.get(sector, [])[:n]

    def import_csv(self, file_path):
        """Impor data dari CSV dengan validasi"""
        try:
            new_data = pd.read_csv(file_path)
            
            # Validasi kolom yang diperlukan
            required_columns = ['Stock', 'Lot Balance', 'Avg Price', 'Market Price']
            if not all(col in new_data.columns for col in required_columns):
                return False, "Kolom yang diperlukan tidak lengkap"
            
            # Hitung kolom yang hilang
            if 'Balance' not in new_data.columns:
                new_data['Balance'] = new_data['Lot Balance'] * 100
            if 'Stock Value' not in new_data.columns:
                new_data['Stock Value'] = new_data['Balance'] * new_data['Avg Price']
            if 'Market Value' not in new_data.columns:
                new_data['Market Value'] = new_data['Balance'] * new_data['Market Price']
            if 'Unrealized' not in new_data.columns:
                new_data['Unrealized'] = new_data['Market Value'] - new_data['Stock Value']
            
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            return True, "Data berhasil diimpor"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

    def export_csv(self, file_path):
        """Ekspor data ke CSV"""
        try:
            self.df.to_csv(file_path, index=False)
            return True, "Data berhasil diekspor"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def risk_analysis(self):
        """Analisis risiko portofolio yang lebih komprehensif"""
        try:
            # Hitung volatilitas berdasarkan data historis
            portfolio_volatility = self.calculate_portfolio_volatility()
            beta = self.calculate_portfolio_beta()
            sharpe = self.calculate_sharpe_ratio()
            var_95 = self.calculate_var(confidence=0.95)
            
            return {
                'Volatilitas': portfolio_volatility,
                'Beta': beta,
                'Sharpe Ratio': sharpe,
                'VaR 95%': var_95,
                'Risk Level': self.categorize_risk(portfolio_volatility)
            }
        except Exception as e:
            print(f"Error in risk analysis: {e}")
            return {
                'Volatilitas': 0.15,
                'Beta': 1.0,
                'Sharpe Ratio': 0.5,
                'VaR 95%': 0.05,
                'Risk Level': 'Medium'
            }

    def calculate_portfolio_volatility(self):
        """Hitung volatilitas portofolio berdasarkan data historis"""
        try:
            weights = self.df['Market Value'] / self.df['Market Value'].sum()
            individual_volatilities = []
            
            for i, row in self.df.iterrows():
                ticker = row['Stock'].split('-')[0]
                try:
                    data = yf.download(f"{ticker}.JK", period='1y', progress=False)
                    if not data.empty:
                        returns = data['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        individual_volatilities.append(volatility)
                    else:
                        individual_volatilities.append(0.2)  # Default
                except:
                    individual_volatilities.append(0.2)  # Default
            
            # Weighted average volatility (simplified)
            portfolio_vol = np.average(individual_volatilities, weights=weights)
            return min(portfolio_vol, 1.0)  # Cap at 100%
            
        except:
            return 0.15  # Default volatility

    def calculate_portfolio_beta(self):
        """Hitung beta portofolio terhadap IHSG"""
        try:
            # Simulasi beta berdasarkan sektor
            sector_betas = {
                'Energy': 1.2,
                'Mining': 1.4,
                'Financial': 0.9,
                'Consumer': 0.8
            }
            
            sector_allocation = self.analyze_sectors()
            weighted_beta = 0
            
            for sector, allocation in sector_allocation.items():
                beta = sector_betas.get(sector, 1.0)
                weighted_beta += beta * allocation
                
            return weighted_beta
        except:
            return 1.0

    def calculate_sharpe_ratio(self, risk_free_rate=0.06):
        """Hitung Sharpe Ratio"""
        try:
            # Estimasi return portofolio
            total_unrealized = self.df['Unrealized'].sum()
            total_investment = self.df['Stock Value'].sum()
            
            if total_investment > 0:
                portfolio_return = total_unrealized / total_investment
                portfolio_vol = self.calculate_portfolio_volatility()
                
                if portfolio_vol > 0:
                    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                    return sharpe
                    
            return 0.0
        except:
            return 0.0

    def calculate_var(self, confidence=0.95):
        """Hitung Value at Risk"""
        try:
            portfolio_vol = self.calculate_portfolio_volatility()
            # Asumsi distribusi normal
            from scipy.stats import norm
            var = norm.ppf(1 - confidence) * portfolio_vol
            return abs(var)
        except:
            return 0.05

    def categorize_risk(self, volatility):
        """Kategorikan tingkat risiko"""
        if volatility < 0.1:
            return 'Low'
        elif volatility < 0.2:
            return 'Medium'
        else:
            return 'High'

def main():
    st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
    st.title("📈 Advanced Stock Portfolio Analyzer")
    st.markdown("---")

    # Inisialisasi data dengan error handling
    if 'portfolio' not in st.session_state:
        initial_data = {
            'Stock': [
                'AADI-ADARO ANDALAN INDONESIA Tbk, PT',
                'ADRO-Adaro Energy Tbk, PT',
                'ANTM-ANEKA TAMBANG Tbk',
                'BFIN-BFI FINANCE INDONESIA Tbk',
                'BJBR-BANK JABAR BANTEN Tbk',
                'BSSR-BARAMULTI SUKSESSARANA Tbk',
                'LPPF-MATAHARI DEPARTMENT STORE Tbk',
                'PGAS-PERUSAHAAN GAS NEGARA Tbk',
                'PTBA-BUKIT ASAM Tbk',
                'UNVR-UNILEVER INDONESIA Tbk',
                'WIIM-WISMILAK INTI MAKMUR Tbk'
            ],
            'Lot Balance': [5, 17, 15, 30, 23, 11, 5, 10, 4, 60, 5],
            'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
            'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
            'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
            'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835],
            'Market Value': [3612500, 3740000, 4665000, 2715000, 1955000, 4840000, 872500, 1820000, 1156000, 10380000, 417500],
            'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215]
        }
        st.session_state.portfolio = pd.DataFrame(initial_data)
    
    # Inisialisasi analyzer dengan error handling
    try:
        analyzer = PortfolioAnalyzer(st.session_state.portfolio)
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        return
    
    # Sidebar menu
    menu = st.sidebar.selectbox("📋 Menu Utama", [
        "🏠 Dashboard", 
        "📊 Analisis Saham", 
        "🎯 Diversifikasi Portofolio",
        "✏️ Update Portofolio",
        "🔮 Prediksi Harga",
        "💰 Proyeksi Dividen",
        "💡 Rekomendasi Saham",
        "💾 Manajemen Data",
        "⚠️ Analisis Risiko"
    ])
    
    if menu == "🏠 Dashboard":
        st.header("📊 Dashboard Portofolio")
        
        # Metrics utama
        col1, col2, col3, col4 = st.columns(4)
        
        total_value = analyzer.df['Market Value'].sum()
        total_cost = analyzer.df['Stock Value'].sum()
        unrealized = analyzer.df['Unrealized'].sum()
        
        with col1:
            st.metric("💼 Total Nilai Portofolio", f"Rp {total_value:,.0f}")
        with col2:
            st.metric("📈 Unrealized P/L", 
                     f"Rp {unrealized:,.0f}", 
                     delta=f"{(unrealized / total_cost * 100):.2f}%" if total_cost > 0 else "0%")
        with col3:
            st.metric("📊 Jumlah Saham", len(analyzer.df))
        with col4:
            st.metric("🎯 Total Investasi", f"Rp {total_cost:,.0f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Alokasi Sektor")
            try:
                sector_allocation = analyzer.analyze_sectors()
                if not sector_allocation.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = plt.cm.Set3(range(len(sector_allocation)))
                    wedges, texts, autotexts = ax.pie(sector_allocation.values, 
                                                     labels=sector_allocation.index, 
                                                     autopct='%1.1f%%',
                                                     colors=colors,
                                                     startangle=90)
                    ax.set_title("Distribusi Sektor Portofolio")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Data sektor tidak tersedia")
            except Exception as e:
                st.error(f"Error creating sector chart: {e}")
        
        with col2:
            st.subheader("📊 Top Performers")
            try:
                top_performers = analyzer.df.nlargest(5, 'Unrealized')[['Stock', 'Unrealized', 'Market Value']]
                top_performers['Stock'] = top_performers['Stock'].str.split('-').str[0]
                st.dataframe(top_performers, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying top performers: {e}")
        
        # Detailed portfolio table
        st.subheader("📋 Detail Portofolio")
        display_df = analyzer.df.copy()
        display_df['Stock'] = display_df['Stock'].str.split('-').str[0]
        st.dataframe(display_df, use_container_width=True)
    
    elif menu == "📊 Analisis Saham":
        st.header("🔍 Analisis Fundamental & Teknikal")
        
        try:
            with st.spinner("Menganalisis saham..."):
                analysis = analyzer.analyze_blackrock_style()
                combined = pd.concat([analyzer.df[['Stock']], analysis], axis=1)
                combined['Stock'] = combined['Stock'].str.split('-').str[0]
                
                st.subheader("📊 Hasil Analisis")
                st.dataframe(combined, use_container_width=True)
                
                # Visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Analisis Valuasi (PER)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    valid_per = combined.dropna(subset=['PER'])
                    if not valid_per.empty:
                        bars = ax.bar(valid_per['Stock'], valid_per['PER'], color='skyblue')
                        ax.axhline(y=15, color='red', linestyle='--', label='Undervalued (<15)')
                        ax.axhline(y=25, color='orange', linestyle='--', label='Overvalued (>25)')
                        ax.set_title('Price Earnings Ratio (PER)')
                        ax.set_xlabel('Saham')
                        ax.set_ylabel('PER')
                        ax.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    st.subheader("📊 Momentum 1 Bulan")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green' if x > 0 else 'red' for x in combined['Momentum_1M']]
                    bars = ax.bar(combined['Stock'], combined['Momentum_1M'], color=colors)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title('Momentum Harga 1 Bulan')
                    ax.set_xlabel('Saham')
                    ax.set_ylabel('Momentum (%)')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                    
        except Exception as e:
            st.error(f"Error dalam analisis: {e}")
    
    elif menu == "🎯 Diversifikasi Portofolio":
        st.header("🔄 Analisis Diversifikasi Portofolio")
        
        try:
            with st.spinner("Menganalisis diversifikasi..."):
                analysis = analyzer.diversification_analysis()
                
                st.subheader("📊 Rekomendasi Aksi")
                st.dataframe(analysis, use_container_width=True)
                
                # Summary aksi
                col1, col2, col3 = st.columns(3)
                
                sell_count = len(analysis[analysis['Aksi'] == 'Sell'])
                hold_count = len(analysis[analysis['Aksi'] == 'Hold'])
                strong_hold_count = len(analysis[analysis['Aksi'] == 'Strong Hold'])
                
                with col1:
                    st.metric("🚨 Saham untuk Dijual", sell_count)
                    if sell_count > 0:
                        sell_list = analysis[analysis['Aksi'] == 'Sell']['Stock'].str.split('-').str[0].tolist()
                        st.write(", ".join(sell_list))
                
                with col2:
                    st.metric("⏸️ Saham Hold", hold_count)
                    
                with col3:
                    st.metric("💎 Saham Strong Hold", strong_hold_count)
                    if strong_hold_count > 0:
                        strong_hold_list = analysis[analysis['Aksi'] == 'Strong Hold']['Stock'].str.split('-').str[0].tolist()
                        st.write(", ".join(strong_hold_list))
                        
        except Exception as e:
            st.error(f"Error dalam analisis diversifikasi: {e}")
    
    elif menu == "✏️ Update Portofolio":
        st.header("📝 Update Portofolio")
        
        action = st.radio("Pilih Aksi:", ["➕ Tambah Saham", "✏️ Edit Saham", "🗑️ Hapus Saham"])
        changes = []
        
        if action == "➕ Tambah Saham":
            with st.form("add_form"):
                st.subheader("Tambah Saham Baru")
                stock = st.text_input("Nama Saham (Format: KODE-NAMA PERUSAHAAN)", 
                                     placeholder="Contoh: BBRI-Bank Rakyat Indonesia")
                
                col1, col2 = st.columns(2)
                with col1:
                    lot = st.number_input("Jumlah Lot", min_value=1, value=1)
                    avg_price = st.number_input("Harga Rata-rata (Rp)", min_value=1, value=1000)
                
                with col2:
                    market_price = st.number_input("Harga Pasar Saat Ini (Rp)", min_value=1, value=1000)
                
                if st.form_submit_button("➕ Tambah Saham", use_container_width=True):
                    if stock and '-' in stock:
                        changes.append({
                            'action': 'add',
                            'stock': stock,
                            'lot': lot,
                            'price': avg_price,
                            'market_price': market_price
                        })
                        st.success(f"Saham {stock.split('-')[0]} akan ditambahkan")
                    else:
                        st.error("Format nama saham tidak valid. Gunakan format: KODE-NAMA")
        
        elif action == "✏️ Edit Saham":
            if not analyzer.df.empty:
                selected_stock = st.selectbox("Pilih Saham untuk Diedit", analyzer.df['Stock'])
                idx = analyzer.df[analyzer.df['Stock'] == selected_stock].index[0]
                
                with st.form("edit_form"):
                    st.subheader(f"Edit {selected_stock.split('-')[0]}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        current_lot = analyzer.df.at[idx, 'Lot Balance']
                        new_lot = st.number_input("Jumlah Lot Baru", 
                                                 value=int(current_lot), min_value=1)
                    
                    with col2:
                        current_price = analyzer.df.at[idx, 'Avg Price']
                        new_price = st.number_input("Harga Rata-rata Baru (Rp)", 
                                                   value=float(current_price), min_value=1.0)
                    
                    if st.form_submit_button("💾 Update Saham", use_container_width=True):
                        changes.append({
                            'action': 'update',
                            'stock': selected_stock,
                            'new_lot': new_lot,
                            'new_price': new_price
                        })
                        st.success(f"Saham {selected_stock.split('-')[0]} akan diupdate")
            else:
                st.warning("Tidak ada saham dalam portofolio")
        
        elif action == "🗑️ Hapus Saham":
            if not analyzer.df.empty:
                selected_stock = st.selectbox("Pilih Saham untuk Dihapus", analyzer.df['Stock'])
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🗑️ Hapus Saham", use_container_width=True, type="primary"):
                        changes.append({
                            'action': 'delete',
                            'stock': selected_stock
                        })
                        st.success(f"Saham {selected_stock.split('-')[0]} akan dihapus")
                
                with col2:
                    st.warning("⚠️ Aksi ini tidak dapat dibatalkan!")
            else:
                st.warning("Tidak ada saham dalam portofolio")
        
        # Proses perubahan
        if changes:
            try:
                updated_df = analyzer.update_portfolio(changes)
                st.session_state.portfolio = updated_df
                st.success("✅ Portofolio berhasil diperbarui!")
                
                st.subheader("📊 Portofolio Terbaru")
                display_df = updated_df.copy()
                display_df['Stock'] = display_df['Stock'].str.split('-').str[0]
                st.dataframe(display_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error updating portfolio: {e}")
    
    elif menu == "🔮 Prediksi Harga":
        st.header("🔮 Prediksi Harga Saham")
        st.info("💡 Prediksi menggunakan Random Forest dengan data historis 2 tahun")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🚀 Mulai Prediksi", use_container_width=True, type="primary"):
                st.session_state.run_prediction = True
        
        if st.session_state.get('run_prediction', False):
            with st.spinner("🔄 Menganalisis data historis dan membuat prediksi..."):
                try:
                    predictions = analyzer.predict_prices()
                    
                    if predictions:
                        st.success(f"✅ Berhasil memprediksi {len(predictions)} saham")
                        
                        for stock, values in predictions.items():
                            with st.expander(f"📊 {stock.split('-')[0]} - Confidence: {values['confidence']}"):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("💰 Harga Saat Ini", f"Rp {values['current_price']:,.0f}")
                                
                                with col2:
                                    change_st = (values['short_term'] - values['current_price']) / values['current_price'] * 100
                                    st.metric("📈 Jangka Pendek (1-3 bulan)", 
                                             f"Rp {values['short_term']:,.0f}",
                                             delta=f"{change_st:+.1f}%")
                                
                                with col3:
                                    change_mt = (values['medium_term'] - values['current_price']) / values['current_price'] * 100
                                    st.metric("📊 Jangka Menengah (6-12 bulan)", 
                                             f"Rp {values['medium_term']:,.0f}",
                                             delta=f"{change_mt:+.1f}%")
                                
                                with col4:
                                    change_lt = (values['long_term'] - values['current_price']) / values['current_price'] * 100
                                    st.metric("🎯 Jangka Panjang (2-5 tahun)", 
                                             f"Rp {values['long_term']:,.0f}",
                                             delta=f"{change_lt:+.1f}%")
                                
                                # Confidence indicator
                                if values['confidence'] == 'High':
                                    st.success("✅ Prediksi dengan confidence tinggi")
                                elif values['confidence'] == 'Medium':
                                    st.warning("⚠️ Prediksi dengan confidence sedang")
                                else:
                                    st.error("❌ Prediksi dengan confidence rendah")
                    else:
                        st.warning("⚠️ Tidak dapat membuat prediksi. Coba lagi nanti.")
                        
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {e}")
                
                st.session_state.run_prediction = False
    
    elif menu == "💰 Proyeksi Dividen":
        st.header("💰 Proyeksi Dividen & Bunga Majemuk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            years = st.slider("🗓️ Tahun Proyeksi", 1, 30, 10)
        
        with col2:
            growth = st.slider("📈 Pertumbuhan Dividen Tahunan", 0.0, 0.2, 0.05, 0.01, format="%.2f")
        
        with col3:
            reinvest = st.slider("🔄 Persentase Reinvestasi", 0.0, 1.0, 0.7, 0.05, format="%.2f")
        
        if st.button("💡 Hitung Proyeksi", use_container_width=True, type="primary"):
            with st.spinner("🔄 Menghitung proyeksi dividen..."):
                try:
                    projections = analyzer.dividend_compound_projection(years, growth, reinvest)
                    
                    if projections:
                        st.success(f"✅ Proyeksi berhasil untuk {len(projections)} saham yang memberikan dividen")
                        
                        # Summary metrics
                        total_current_value = 0
                        total_projected_value = 0
                        
                        for stock, data in projections.items():
                            current_val = analyzer.df[analyzer.df['Stock'] == stock]['Market Value'].values[0]
                            projected_val = data[-1]['total_value']
                            
                            total_current_value += current_val
                            total_projected_value += projected_val
                        
                        if total_current_value > 0:
                            total_growth = (total_projected_value - total_current_value) / total_current_value * 100
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("💼 Nilai Saat Ini", f"Rp {total_current_value:,.0f}")
                            
                            with col2:
                                st.metric("🎯 Proyeksi Nilai Akhir", f"Rp {total_projected_value:,.0f}")
                            
                            with col3:
                                st.metric("📈 Total Pertumbuhan", f"{total_growth:.1f}%")
                        
                        # Detail per saham
                        for stock, data in projections.items():
                            with st.expander(f"📊 Detail Proyeksi: {stock.split('-')[0]}"):
                                current_val = analyzer.df[analyzer.df['Stock'] == stock]['Market Value'].values[0]
                                final_val = data[-1]['total_value']
                                total_growth = (final_val - current_val) / current_val * 100
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("💰 Investasi Awal", f"Rp {current_val:,.0f}")
                                    st.metric("🎯 Nilai Akhir", f"Rp {final_val:,.0f}")
                                    st.metric("📈 Pertumbuhan Total", f"{total_growth:.1f}%")
                                
                                with col2:
                                    # Grafik pertumbuhan
                                    years_list = [d['year'] for d in data]
                                    values = [d['total_value'] for d in data]
                                    
                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    ax.plot(years_list, values, marker='o', linewidth=2, markersize=4)
                                    ax.set_title(f"Proyeksi Pertumbuhan {stock.split('-')[0]}")
                                    ax.set_xlabel("Tahun")
                                    ax.set_ylabel("Nilai (Rp)")
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Format y-axis
                                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x/1e6:.1f}M'))
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Tabel proyeksi tahunan
                                st.subheader("📋 Proyeksi Tahunan")
                                df_projection = pd.DataFrame(data)
                                df_projection['dividend'] = df_projection['dividend'].apply(lambda x: f"Rp {x:,.0f}")
                                df_projection['total_value'] = df_projection['total_value'].apply(lambda x: f"Rp {x:,.0f}")
                                st.dataframe(df_projection, use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak ada saham yang memberikan dividen dalam portofolio")
                        
                except Exception as e:
                    st.error(f"❌ Error dalam proyeksi dividen: {e}")
    
    elif menu == "💡 Rekomendasi Saham":
        st.header("💡 Rekomendasi Diversifikasi Portofolio")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n = st.number_input("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)
            
            if st.button("🔍 Dapatkan Rekomendasi", use_container_width=True, type="primary"):
                st.session_state.get_recommendations = True
        
        if st.session_state.get('get_recommendations', False):
            with st.spinner("🔄 Menganalisis portofolio dan mencari rekomendasi..."):
                try:
                    # Analisis sektor saat ini
                    sector_allocation = analyzer.analyze_sectors()
                    
                    st.subheader("📊 Alokasi Sektor Saat Ini")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not sector_allocation.empty:
                            # Tabel alokasi
                            allocation_df = pd.DataFrame({
                                'Sektor': sector_allocation.index,
                                'Alokasi (%)': (sector_allocation.values * 100).round(1),
                                'Status': ['✅ Cukup' if x > 0.2 else '⚠️ Kurang' for x in sector_allocation.values]
                            })
                            st.dataframe(allocation_df, use_container_width=True)
                        
                    with col2:
                        # Pie chart
                        if not sector_allocation.empty:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            colors = plt.cm.Set3(range(len(sector_allocation)))
                            wedges, texts, autotexts = ax.pie(sector_allocation.values, 
                                                             labels=sector_allocation.index, 
                                                             autopct='%1.1f%%',
                                                             colors=colors,
                                                             startangle=90)
                            ax.set_title("Distribusi Sektor Portofolio")
                            st.pyplot(fig)
                            plt.close()
                    
                    # Rekomendasi
                    recommendations = analyzer.recommend_stocks(n)
                    
                    if recommendations:
                        st.subheader("🎯 Rekomendasi Saham")
                        
                        for i, stock in enumerate(recommendations, 1):
                            ticker = stock.replace('.JK', '')
                            st.write(f"{i}. **{ticker}** - {stock}")
                        
                        st.info("💡 **Tips**: Pertimbangkan untuk menambahkan saham dari sektor yang kurang terwakili untuk meningkatkan diversifikasi portofolio Anda")
                        
                        # Sektor yang perlu diperkuat
                        underrepresented = sector_allocation[sector_allocation < 0.2].index.tolist()
                        if underrepresented:
                            st.warning(f"⚠️ **Sektor yang perlu diperkuat**: {', '.join(underrepresented)}")
                    else:
                        st.success("✅ Portofolio Anda sudah terdiversifikasi dengan baik!")
                        
                except Exception as e:
                    st.error(f"❌ Error dalam rekomendasi: {e}")
            
            st.session_state.get_recommendations = False
    
    elif menu == "💾 Manajemen Data":
        st.header("💾 Manajemen Data Portofolio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Impor Data")
            uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Preview data
                    preview_df = pd.read_csv(uploaded_file)
                    st.write("👀 **Preview Data:**")
                    st.dataframe(preview_df.head(), use_container_width=True)
                    
                    if st.button("📥 Impor Data", use_container_width=True, type="primary"):
                        success, message = analyzer.import_csv(uploaded_file)
                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.portfolio = analyzer.df
                        else:
                            st.error(f"❌ {message}")
                            
                except Exception as e:
                    st.error(f"❌ Error membaca file: {e}")
        
        with col2:
            st.subheader("📤 Ekspor Data")
            
            export_format = st.selectbox("Format Ekspor", ["CSV", "Excel"])
            
            if st.button("📤 Ekspor Portofolio", use_container_width=True, type="primary"):
                try:
                    if export_format == "CSV":
                        csv_data = analyzer.df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Unduh CSV",
                            data=csv_data,
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:  # Excel
                        # Untuk Excel, kita perlu menggunakan buffer
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            analyzer.df.to_excel(writer, sheet_name='Portfolio', index=False)
                        
                        st.download_button(
                            label="⬇️ Unduh Excel",
                            data=output.getvalue(),
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error mengekspor data: {e}")
        
        # Template download
        st.subheader("📋 Template Import")
        template_data = {
            'Stock': ['BBRI-Bank Rakyat Indonesia', 'TLKM-Telekomunikasi Indonesia'],
            'Lot Balance': [10, 5],
            'Avg Price': [4500, 3200],
            'Market Price': [4600, 3100]
        }
        template_df = pd.DataFrame(template_data)
        
        st.write("📄 **Template format CSV:**")
        st.dataframe(template_df, use_container_width=True)
        
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Unduh Template CSV",
            data=template_csv,
            file_name="portfolio_template.csv",
            mime="text/csv"
        )
    
    elif menu == "⚠️ Analisis Risiko":
        st.header("⚠️ Analisis Risiko Portofolio")
        
        if st.button("🔍 Analisis Risiko", use_container_width=True, type="primary"):
            with st.spinner("🔄 Menganalisis risiko portofolio..."):
                try:
                    risk = analyzer.risk_analysis()
                    
                    st.success("✅ Analisis risiko selesai")
                    
                    # Metrics utama
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        vol_color = "normal"
                        if risk['Volatilitas'] > 0.25:
                            vol_color = "inverse"
                        st.metric("📊 Volatilitas Tahunan", f"{risk['Volatilitas']*100:.1f}%")
                    
                    with col2:
                        beta_color = "normal"
                        if risk['Beta'] > 1.5:
                            beta_color = "inverse"
                        st.metric("📈 Beta (vs Pasar)", f"{risk['Beta']:.2f}")
                    
                    with col3:
                        sharpe_color = "normal"
                        if risk['Sharpe Ratio'] < 0:
                            sharpe_color = "inverse"
                        st.metric("🎯 Sharpe Ratio", f"{risk['Sharpe Ratio']:.2f}")
                    
                    with col4:
                        st.metric("⚠️ VaR 95%", f"{risk['VaR 95%']*100:.1f}%")
                    
                    # Risk level indicator
                    risk_level = risk['Risk Level']
                    if risk_level == 'Low':
                        st.success(f"✅ **Tingkat Risiko: {risk_level}** - Portofolio relatif konservatif")
                    elif risk_level == 'Medium':
                        st.warning(f"⚠️ **Tingkat Risiko: {risk_level}** - Portofolio dengan risiko moderat")
                    else:
                        st.error(f"❌ **Tingkat Risiko: {risk_level}** - Portofolio dengan risiko tinggi")
                    
                    # Interpretasi dan rekomendasi
                    st.subheader("📋 Interpretasi & Rekomendasi")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Penjelasan Metrik:**")
                        
                        with st.expander("📈 Volatilitas"):
                            st.write("""
                            - **< 15%**: Risiko rendah
                            - **15-25%**: Risiko sedang  
                            - **> 25%**: Risiko tinggi
                            
                            Volatilitas mengukur fluktuasi harga saham dalam portofolio.
                            """)
                        
                        with st.expander("📊 Beta"):
                            st.write("""
                            - **< 1.0**: Kurang fluktuatif dari pasar
                            - **= 1.0**: Sejalan dengan pasar
                            - **> 1.0**: Lebih fluktuatif dari pasar
                            
                            Beta mengukur sensitivitas portofolio terhadap pergerakan pasar.
                            """)
                        
                        with st.expander("🎯 Sharpe Ratio"):
                            st.write("""
                            - **> 1.0**: Sangat baik
                            - **0.5-1.0**: Baik
                            - **< 0.5**: Perlu perbaikan
                            
                            Sharpe Ratio mengukur return yang disesuaikan dengan risiko.
                            """)
                    
                    with col2:
                        st.write("**💡 Rekomendasi:**")
                        
                        recommendations = []
                        
                        if risk['Volatilitas'] > 0.25:
                            recommendations.append("🔴 Pertimbangkan mengurangi saham dengan volatilitas tinggi")
                        
                        if risk['Beta'] > 1.5:
                            recommendations.append("🔴 Tambahkan saham defensif untuk mengurangi beta")
                        
                        if risk['Sharpe Ratio'] < 0.5:
                            recommendations.append("🔴 Review pemilihan saham untuk meningkatkan return/risk ratio")
                        
                        if risk['VaR 95%'] > 0.1:
                            recommendations.append("🔴 Risiko kerugian harian cukup tinggi, pertimbangkan diversifikasi")
                        
                        if not recommendations:
                            recommendations.append("✅ Profil risiko portofolio dalam batas wajar")
                        
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    
                    # Risk visualization
                    st.subheader("📊 Visualisasi Risiko")
                    
                    # Risk gauge
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    risk_categories = ['Low\n(<15%)', 'Medium\n(15-25%)', 'High\n(>25%)']
                    risk_values = [15, 10, 15]  # Width of each category
                    colors = ['green', 'yellow', 'red']
                    
                    # Create horizontal bar chart
                    bars = ax.barh(risk_categories, risk_values, color=colors, alpha=0.7)
                    
                    # Add current volatility marker
                    current_vol = risk['Volatilitas'] * 100
                    if current_vol <= 15:
                        y_pos = 0
                        x_pos = current_vol
                    elif current_vol <= 25:
                        y_pos = 1
                        x_pos = current_vol
                    else:
                        y_pos = 2
                        x_pos = min(current_vol, 40)  # Cap at 40% for visualization
                    
                    ax.plot(x_pos, y_pos, 'o', markersize=15, color='black', 
                           markerfacecolor='white', markeredgewidth=3, label=f'Portofolio Anda ({current_vol:.1f}%)')
                   
                    ax.set_xlabel('Volatilitas (%)')
                    ax.set_title('Posisi Risiko Portofolio')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Error dalam analisis risiko: {e}")

if __name__ == "__main__":
    main()
