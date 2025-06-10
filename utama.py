import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import csv
import os
from datetime import datetime, timedelta

class PortfolioAnalyzer:
    def __init__(self, data):
        self.df = data
        self.today = datetime.today()
        
    def analyze_blackrock_style(self):
        """Analisis gaya Lo Keng Hong dan BlackRock"""
        analysis = pd.DataFrame()
        
        # 1. Analisis Valuasi
        analysis['PER'] = self.df['Market Price'] / (self.df['Stock Value'] / self.df['Balance'])
        analysis['PBV'] = self.df['Market Price'] / (self.df['Stock Value'] / self.df['Balance'] / 2)  # Asumsi
        
        # 2. Momentum
        analysis['Momentum_1M'] = self.df['Stock'].apply(
            lambda x: self.get_price_momentum(x.split('-')[0], 30)
        
        # 3. Profitabilitas
        analysis['ROE'] = self.df['Stock'].apply(
            lambda x: self.get_roe(x.split('-')[0]))
        
        # 4. Rekomendasi
        conditions = [
            (analysis['PER'] < 15) & (analysis['PBV'] < 1.5) & (analysis['Momentum_1M'] > 0),
            (analysis['PER'] > 25) | (analysis['PBV'] > 3) | (analysis['Momentum_1M'] < -0.05)
        ]
        choices = ['Strong Buy', 'Sell']
        analysis['Rekomendasi'] = np.select(conditions, choices, default='Hold')
        
        return analysis[['PER', 'PBV', 'Momentum_1M', 'ROE', 'Rekomendasi']]

    def get_price_momentum(self, ticker, days):
        """Hitung momentum harga"""
        try:
            end_date = self.today
            start_date = end_date - timedelta(days=days)
            data = yf.download(f"{ticker}.JK", start=start_date, end=end_date)
            if not data.empty:
                return (data['Close'][-1] - data['Close'][0]) / data['Close'][0]
        except:
            return 0
        return 0

    def get_roe(self, ticker):
        """Ambil data ROE (simulasi)"""
        # Implementasi nyata perlu akses data fundamental
        roe_values = {'AADI': 0.15, 'ADRO': 0.22, 'ANTM': 0.18, 
                     'BFIN': 0.12, 'BJBR': 0.09, 'BSSR': 0.25,
                     'LPPF': 0.08, 'PGAS': 0.11, 'PTBA': 0.21, 'UNVR': 0.28, 'WIIM': 0.14}
        return roe_values.get(ticker, 0.12)

    def diversification_analysis(self):
        """Identifikasi saham untuk dipertahankan/dijual"""
        analysis = self.analyze_blackrock_style()
        combined = pd.concat([self.df, analysis], axis=1)
        
        # Kriteria penjualan
        sell_criteria = (
            (combined['Unrealized'] < -500000) |
            (combined['Rekomendasi'] == 'Sell') |
            (combined['Market Price'] < 0.8 * combined['Avg Price'])
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
        """Update portofolio berdasarkan perubahan"""
        for change in changes:
            idx = self.df[self.df['Stock'] == change['stock']].index
            if not idx.empty:
                if change['action'] == 'delete':
                    self.df = self.df.drop(idx)
                elif change['action'] == 'update':
                    self.df.at[idx[0], 'Lot Balance'] = change['new_lot']
                    self.df.at[idx[0], 'Avg Price'] = change['new_price']
            elif change['action'] == 'add':
                new_row = {
                    'Stock': change['stock'],
                    'Lot Balance': change['lot'],
                    'Balance': change['lot'] * 100,
                    'Avg Price': change['price'],
                    'Market Price': change['price'],
                    'Unrealized': 0
                }
                self.df = self.df.append(new_row, ignore_index=True)
        
        # Recalculate values
        self.df['Stock Value'] = self.df['Balance'] * self.df['Avg Price']
        self.df['Market Value'] = self.df['Balance'] * self.df['Market Price']
        self.df['Unrealized'] = self.df['Market Value'] - self.df['Stock Value']
        return self.df

    def predict_prices(self):
        """Prediksi harga saham jangka pendek, menengah, panjang"""
        predictions = {}
        for stock in self.df['Stock']:
            ticker = stock.split('-')[0]
            try:
                data = yf.download(f"{ticker}.JK", period='5y')
                if len(data) < 100:
                    continue
                
                # Prepare data
                data['SMA50'] = data['Close'].rolling(50).mean()
                data['SMA200'] = data['Close'].rolling(200).mean()
                data = data.dropna()
                
                # Fit model
                X = data[['SMA50', 'SMA200']]
                y = data['Close']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                
                # Predict future prices
                last_data = X.iloc[-1].values.reshape(1, -1)
                short_term = model.predict(last_data)[0]
                medium_term = short_term * 1.15  # Simulasi
                long_term = medium_term * 1.25   # Simulasi
                
                predictions[stock] = {
                    'short_term': short_term,
                    'medium_term': medium_term,
                    'long_term': long_term
                }
            except:
                continue
        
        return predictions

    def dividend_compound_projection(self, years=10, dividend_growth=0.05, reinvestment_rate=0.7):
        """Proyeksi dividen dengan bunga majemuk"""
        projections = {}
        for _, row in self.df.iterrows():
            ticker = row['Stock'].split('-')[0]
            div_yield = self.get_dividend_yield(ticker)
            
            if div_yield > 0:
                current_val = row['Market Value']
                div_per_year = current_val * div_yield
                total = current_val
                yearly = []
                
                for year in range(1, years + 1):
                    dividend = div_per_year * (1 + dividend_growth) ** year
                    reinvested = dividend * reinvestment_rate
                    total = total * 1.05 + reinvested  # Asumsi apresiasi harga 5% + reinvestasi
                    yearly.append({
                        'year': year,
                        'dividend': dividend,
                        'reinvested': reinvested,
                        'total_value': total
                    })
                
                projections[row['Stock']] = yearly
        return projections

    def get_dividend_yield(self, ticker):
        """Ambil dividend yield (simulasi)"""
        dy_values = {'AADI': 0.04, 'ADRO': 0.08, 'ANTM': 0.05, 
                    'BFIN': 0.07, 'BJBR': 0.06, 'BSSR': 0.09,
                    'LPPF': 0.03, 'PGAS': 0.05, 'PTBA': 0.07, 'UNVR': 0.04, 'WIIM': 0.02}
        return dy_values.get(ticker, 0.05)

    def recommend_stocks(self, n=5):
        """Rekomendasi saham untuk diversifikasi portofolio"""
        # Analisis sektor yang kurang terwakili
        sector_allocation = self.analyze_sectors()
        underrepresented = sector_allocation[sector_allocation['Allocation'] < 0.1].index.tolist()
        
        # Rekomendasi berdasarkan kinerja sektor
        recommendations = []
        for sector in underrepresented:
            sector_stocks = self.get_top_sector_stocks(sector, n=2)
            recommendations.extend(sector_stocks)
        
        return recommendations[:n]

    def analyze_sectors(self):
        """Analisis alokasi sektor portofolio"""
        # Pemetaan sektor (simulasi)
        sector_map = {
            'AADI': 'Energy', 'ADRO': 'Energy', 'ANTM': 'Mining', 
            'BFIN': 'Financial', 'BJBR': 'Financial', 'BSSR': 'Mining',
            'LPPF': 'Retail', 'PGAS': 'Energy', 'PTBA': 'Mining', 
            'UNVR': 'Consumer', 'WIIM': 'Consumer'
        }
        
        self.df['Sektor'] = self.df['Stock'].apply(lambda x: sector_map.get(x.split('-')[0], 'Other'))
        sector_allocation = self.df.groupby('Sektor')['Market Value'].sum() / self.df['Market Value'].sum()
        return sector_allocation

    def get_top_sector_stocks(self, sector, n=3):
        """Dapatkan saham terbaik di sektor tertentu"""
        # Database saham per sektor (simulasi)
        sector_stocks = {
            'Energy': ['PGAS.JK', 'PTBA.JK', 'ADRO.JK'],
            'Mining': ['ANTM.JK', 'BSSR.JK', 'MDKA.JK'],
            'Financial': ['BBRI.JK', 'BBCA.JK', 'BFIN.JK'],
            'Consumer': ['UNVR.JK', 'ICBP.JK', 'MYOR.JK'],
            'Retail': ['LPPF.JK', 'AMRT.JK', 'MAPA.JK']
        }
        return sector_stocks.get(sector, [])[:n]

    def import_csv(self, file_path):
        """Impor data dari CSV"""
        try:
            new_data = pd.read_csv(file_path)
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            return True
        except:
            return False

    def export_csv(self, file_path):
        """Ekspor data ke CSV"""
        try:
            self.df.to_csv(file_path, index=False)
            return True
        except:
            return False

    # FITUR TAMBAHAN
    def risk_analysis(self):
        """Analisis risiko portofolio"""
        volatility = self.calculate_volatility()
        beta = self.calculate_beta()
        sharpe = self.calculate_sharpe()
        
        return {
            'Volatilitas': volatility,
            'Beta': beta,
            'Sharpe Ratio': sharpe
        }

    def calculate_volatility(self, window=30):
        """Hitung volatilitas portofolio"""
        # Implementasi nyata membutuhkan data historis
        return 0.15  # Simulasi

    def calculate_beta(self):
        """Hitung beta terhadap pasar"""
        # Beta terhadap IHSG
        return 1.02  # Simulasi

    def calculate_sharpe(self, risk_free_rate=0.05):
        """Hitung Sharpe Ratio"""
        # Return portofolio tahunan
        portfolio_return = 0.12  # Simulasi
        return (portfolio_return - risk_free_rate) / self.calculate_volatility()

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("📈 Advanced Stock Portfolio Analyzer")
    st.write("Tools analisis saham profesional dengan fitur lengkap")

    # Inisialisasi data
    if 'portfolio' not in st.session_state:
        initial_data = {
            'Stock': [
                'AADI-ADARO ANDALAN INDONESIA Tbk, PT',
                'ADRO-Alamtri Resources Indonesia Tbk',
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
    
    analyzer = PortfolioAnalyzer(st.session_state.portfolio.copy())
    
    # Menu utama
    menu = st.sidebar.selectbox("Menu Utama", [
        "Dashboard", 
        "Analisis Saham", 
        "Diversifikasi Portofolio",
        "Update Portofolio",
        "Prediksi Harga",
        "Proyeksi Dividen",
        "Rekomendasi Saham",
        "Manajemen Data",
        "Analisis Risiko"
    ])
    
    if menu == "Dashboard":
        st.header("Dashboard Portofolio")
        col1, col2, col3 = st.columns(3)
        total_value = analyzer.df['Market Value'].sum()
        unrealized = analyzer.df['Unrealized'].sum()
        
        col1.metric("Total Nilai Portofolio", f"Rp {total_value:,.0f}")
        col2.metric("Unrealized P/L", f"Rp {unrealized:,.0f}", 
                    delta=f"{(unrealized / total_value * 100):.2f}%" if total_value > 0 else "0%")
        col3.metric("Jumlah Saham", len(analyzer.df))
        
        # Grafik alokasi
        st.subheader("Alokasi Sektor")
        sector_allocation = analyzer.analyze_sectors()
        fig, ax = plt.subplots()
        ax.pie(sector_allocation, labels=sector_allocation.index, autopct='%1.1f%%')
        st.pyplot(fig)
        
        # Top performers
        st.subheader("Performansi Saham")
        top_performers = analyzer.df.nlargest(5, 'Unrealized')
        st.dataframe(top_performers[['Stock', 'Unrealized']])
    
    elif menu == "Analisis Saham":
        st.header("Analisis Fundamental & Teknikal")
        analysis = analyzer.analyze_blackrock_style()
        combined = pd.concat([analyzer.df['Stock'], analysis], axis=1)
        st.dataframe(combined)
        
        # Visualisasi
        st.subheader("Visualisasi Valuasi")
        fig, ax = plt.subplots()
        ax.bar(combined['Stock'], combined['PER'], label='PER')
        ax.bar(combined['Stock'], combined['PBV'], label='PBV', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
    
    elif menu == "Diversifikasi Portofolio":
        st.header("Analisis Diversifikasi Portofolio")
        analysis = analyzer.diversification_analysis()
        st.dataframe(analysis)
        
        st.subheader("Rekomendasi Aksi")
        sell_list = analysis[analysis['Aksi'] == 'Sell']['Stock'].tolist()
        hold_list = analysis[analysis['Aksi'] == 'Strong Hold']['Stock'].tolist()
        
        col1, col2 = st.columns(2)
        col1.write("🚨 Saham untuk dipertimbangkan dijual:")
        col1.write(sell_list)
        
        col2.write("💎 Saham untuk dipertahankan:")
        col2.write(hold_list)
    
    elif menu == "Update Portofolio":
        st.header("Update Portofolio")
        action = st.radio("Pilih Aksi:", ["Tambah Saham", "Edit Saham", "Hapus Saham"])
        changes = []
        
        if action == "Tambah Saham":
            with st.form("add_form"):
                stock = st.text_input("Nama Saham (Format: KODE-NAMA)")
                lot = st.number_input("Jumlah Lot", min_value=1)
                avg_price = st.number_input("Harga Rata-rata", min_value=1)
                market_price = st.number_input("Harga Pasar", min_value=1)
                
                if st.form_submit_button("Tambah Saham"):
                    changes.append({
                        'action': 'add',
                        'stock': stock,
                        'lot': lot,
                        'price': avg_price,
                        'market_price': market_price
                    })
        
        elif action == "Edit Saham":
            selected_stock = st.selectbox("Pilih Saham", analyzer.df['Stock'])
            idx = analyzer.df[analyzer.df['Stock'] == selected_stock].index[0]
            
            with st.form("edit_form"):
                new_lot = st.number_input("Jumlah Lot Baru", 
                                         value=analyzer.df.at[idx, 'Lot Balance'])
                new_price = st.number_input("Harga Rata-rata Baru", 
                                          value=analyzer.df.at[idx, 'Avg Price'])
                
                if st.form_submit_button("Update Saham"):
                    changes.append({
                        'action': 'update',
                        'stock': selected_stock,
                        'new_lot': new_lot,
                        'new_price': new_price
                    })
        
        elif action == "Hapus Saham":
            selected_stock = st.selectbox("Pilih Saham", analyzer.df['Stock'])
            if st.button("Hapus Saham"):
                changes.append({
                    'action': 'delete',
                    'stock': selected_stock
                })
        
        if changes:
            updated_df = analyzer.update_portfolio(changes)
            st.session_state.portfolio = updated_df
            st.success("Portofolio berhasil diperbarui!")
            st.dataframe(updated_df)
    
    elif menu == "Prediksi Harga":
        st.header("Prediksi Harga Saham")
        st.info("Prediksi menggunakan machine learning (Random Forest) dengan data historis")
        
        if st.button("Mulai Prediksi"):
            predictions = analyzer.predict_prices()
            for stock, values in predictions.items():
                st.subheader(stock)
                col1, col2, col3 = st.columns(3)
                col1.metric("Jangka Pendek (1-3 bulan)", f"Rp {values['short_term']:,.0f}")
                col2.metric("Jangka Menengah (6-12 bulan)", f"Rp {values['medium_term']:,.0f}")
                col3.metric("Jangka Panjang (2-5 tahun)", f"Rp {values['long_term']:,.0f}")
    
    elif menu == "Proyeksi Dividen":
        st.header("Proyeksi Dividen & Bunga Majemuk")
        years = st.slider("Tahun Proyeksi", 1, 30, 10)
        growth = st.slider("Pertumbuhan Dividen Tahunan", 0.0, 0.2, 0.05)
        reinvest = st.slider("Persentase Reinvestasi", 0.0, 1.0, 0.7)
        
        projections = analyzer.dividend_compound_projection(years, growth, reinvest)
        
        for stock, data in projections.items():
            st.subheader(stock)
            last_year = data[-1]
            st.metric("Nilai Portofolio Akhir Tahun", 
                     f"Rp {last_year['total_value']:,.0f}",
                     delta=f"{(last_year['total_value'] / analyzer.df[analyzer.df['Stock'] == stock]['Market Value'].values[0] - 1) * 100:.1f}%")
            
            # Grafik pertumbuhan
            years_list = [d['year'] for d in data]
            values = [d['total_value'] for d in data]
            fig, ax = plt.subplots()
            ax.plot(years_list, values)
            ax.set_title("Proyeksi Pertumbuhan Nilai Investasi")
            st.pyplot(fig)
    
    elif menu == "Rekomendasi Saham":
        st.header("Rekomendasi Diversifikasi Portofolio")
        n = st.number_input("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)
        
        if st.button("Dapatkan Rekomendasi"):
            recommendations = analyzer.recommend_stocks(n)
            st.subheader("Saham Rekomendasi:")
            for stock in recommendations:
                st.write(f"- {stock}")
            
            st.info("Pertimbangkan untuk menambahkan saham di sektor yang kurang terwakili di portofolio Anda")
    
    elif menu == "Manajemen Data":
        st.header("Manajemen Data Portofolio")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Impor Data")
            uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
            if uploaded_file:
                if analyzer.import_csv(uploaded_file):
                    st.success("Data berhasil diimpor!")
                else:
                    st.error("Gagal mengimpor data")
        
        with col2:
            st.subheader("Ekspor Data")
            if st.button("Ekspor ke CSV"):
                if analyzer.export_csv("current_portfolio.csv"):
                    with open("current_portfolio.csv", "rb") as f:
                        st.download_button("Unduh Portofolio", f, file_name="portfolio.csv")
                else:
                    st.error("Gagal mengekspor data")
    
    elif menu == "Analisis Risiko":
        st.header("Analisis Risiko Portofolio")
        risk = analyzer.risk_analysis()
        
        st.metric("Volatilitas (Standar Deviasi Tahunan)", f"{risk['Volatilitas']*100:.1f}%")
        st.metric("Beta (Risiko Sistematis)", f"{risk['Beta']:.2f}")
        st.metric("Sharpe Ratio (Return Disesuaikan Risiko)", f"{risk['Sharpe Ratio']:.2f}")
        
        st.progress(int(risk['Volatilitas']*100))
        st.caption("Tingkat Risiko Portofolio")

if __name__ == "__main__":
    main()
