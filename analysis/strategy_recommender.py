import streamlit as st
import pandas as pd
import random

def generate_fake_sentiment():
    return random.choice(['Positif', 'Netral', 'Negatif'])

def generate_fake_dividend_yield():
    return random.uniform(0, 8)  # Yield lebih realistis

def generate_fake_risk_level():
    return random.choice(['Rendah', 'Menengah', 'Tinggi'])

def generate_fake_growth():
    return random.uniform(-5, 25)  # Range lebih masuk akal

def generate_fake_roe():
    return random.uniform(5, 30)  # Return on Equity (%)

def generate_fake_volatility():
    return random.uniform(10, 60)  # Volatilitas historis (%)

def show_strategy_recommendation(portfolio_df):
    st.header("🧠 Auto Strategy Recommendation")
    st.caption("Sistem rekomendasi berbasis analisis fundamental dan teknikal simulatif")

    # Validasi dataframe
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Silakan upload portofolio terlebih dahulu di halaman Upload Data")
        return None
        
    # Cek kolom wajib
    required_columns = {'Stock', 'Ticker'}
    if not required_columns.issubset(portfolio_df.columns):
        missing = required_columns - set(portfolio_df.columns)
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
        return None

    # Tab untuk multiple views
    tab1, tab2 = st.tabs(["📋 Rekomendasi Saham", "📊 Analisis Portofolio"])

    with tab1:
        st.subheader("Rekomendasi Per Saham")
        
        rekomendasi = []
        for _, row in portfolio_df.iterrows():
            saham = row['Stock']
            ticker = row['Ticker']

            # Generate data simulasi
            sentiment = generate_fake_sentiment()
            div_yield = generate_fake_dividend_yield()
            risk = generate_fake_risk_level()
            pred_growth = generate_fake_growth()
            roe = generate_fake_roe()
            volatility = generate_fake_volatility()

            # Logika rekomendasi diperluas
            strategi = []
            
            # Fundamental kuat
            if div_yield >= 4 and roe > 15:
                strategi.append("🔁 Reinvest Dividen")
                
            # Prospek pertumbuhan
            if pred_growth > 15 and risk != 'Tinggi':
                strategi.append("📈 Hold Jangka Panjang")
                
            # Untuk saham volatil
            if volatility > 40 and sentiment == 'Positif':
                strategi.append("📆 Dollar-Cost Averaging")
                
            # Potensi undervalued
            if roe > 20 and pred_growth > 10:
                strategi.append("💰 Value Investing")
                
            # Sinyal risiko
            if sentiment == 'Negatif' or pred_growth < 0:
                strategi.append("🚫 Tinjau Ulang Kepemilikan")
                
            # Saham defensif
            if div_yield > 5 and volatility < 30:
                strategi.append("🛡️ Dividen + Proteksi Risiko")
                
            if not strategi:
                strategi.append("🔄 Hold / Monitor Berkala")

            rekomendasi.append({
                'Saham': saham,
                'Ticker': ticker,
                'Sentimen': sentiment,
                'DY (%)': f"{div_yield:.2f}",
                'Growth (%)': f"{pred_growth:.2f}",
                'ROE (%)': f"{roe:.2f}",
                'Volatilitas (%)': f"{volatility:.2f}",
                'Risiko': risk,
                'Strategi': " • ".join(strategi)
            })

        df_rek = pd.DataFrame(rekomendasi)
        st.dataframe(
            df_rek,
            use_container_width=True,
            hide_index=True,
            height=40 * min(len(df_rek), 10)
        )

    with tab2:
        st.subheader("Analisis Portofolio")
        
        # Hitung statistik portofolio
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rata-rata Dividend Yield", f"{random.uniform(2.5, 5.5):.2f}%")
        with col2:
            st.metric("Growth Rata-rata", f"{random.uniform(5, 15):.2f}%")
        with col3:
            st.metric("Ekspektasi Return Tahunan", f"{random.uniform(8, 22):.2f}%")
        
        # Rekomendasi alokasi
        with st.expander("⚖️ Rekomendasi Alokasi Aset"):
            st.info("""
            Berdasarkan profil risiko dan kondisi pasar simulatif:
            - 45% Saham Growth
            - 30% Saham Dividen
            - 15% Reksadana Pasar Uang
            - 10% Obligasi
            """)
        
        # Peringatan risiko
        high_risk = [r for r in rekomendasi if r['Risiko'] == 'Tinggi']
        if high_risk:
            with st.expander("⚠️ Saham Berisiko Tinggi", expanded=True):
                st.warning("Portofolio mengandung saham berisiko tinggi:")
                risky_df = pd.DataFrame(high_risk)[['Saham', 'Ticker', 'Strategi']]
                st.dataframe(risky_df, hide_index=True)

    # Catatan footer
    st.info("""
    **Catatan:**  
    Data dan rekomendasi pada halaman ini merupakan **simulasi acak** untuk keperluan demonstrasi. 
    Untuk analisis sesungguhnya, diperlukan integrasi dengan data pasar real-time dan fundamental perusahaan.
    """)
