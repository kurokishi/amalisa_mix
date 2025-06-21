import streamlit as st
from datetime import datetime
import os

# Import fungsi-fungsi utama dari modul-modul
from data.processor import process_uploaded_file
from analysis.portfolio import visualize_portfolio
from analysis.recommendations import show_ai_recommendations
from analysis.what_if import show_what_if_simulation
from analysis.compound import show_compound_projection
from analysis.fundamental import show_fundamental_analysis
from analysis.risk import show_risk_analysis
from analysis.esg import show_esg_analysis
from analysis.sentiment import show_sentiment_analysis
from models.predictor import show_price_prediction

# Konfigurasi halaman
st.set_page_config(layout="wide", page_title="AI Portfolio Management Dashboard")
st.title("🪙 AI Portfolio Management Dashboard")

# Sidebar: Navigasi & Upload
with st.sidebar:
    st.header("Menu Navigasi")
    menu_options = [
        "Portfolio Analysis", "Price Prediction", "What-If Simulation", 
        "AI Recommendations", "Compound Interest", "Fundamental Analysis", 
        "Risk Analysis", "ESG & Berita"]
    selected_menu = st.radio("Pilih Modul:", menu_options)

    st.divider()
    st.header("Upload Portfolio")
    uploaded_file = st.file_uploader("Upload file (CSV/Excel)", type=["csv", "xlsx"])

    st.divider()
    st.header("Parameter Analisis")
    prediction_days = st.slider("Jumlah Hari Prediksi", 7, 365, 30)
    risk_tolerance = st.select_slider("Toleransi Risiko", options=["Low", "Medium", "High"])

# Proses file upload
portfolio_df = process_uploaded_file(uploaded_file)

# Routing berdasarkan menu
if selected_menu == "Portfolio Analysis":
    st.header("Analisis Portofolio")
    visualize_portfolio(portfolio_df)

elif selected_menu == "Price Prediction":
    show_price_prediction(portfolio_df, prediction_days)

elif selected_menu == "What-If Simulation":
    show_what_if_simulation(portfolio_df)

elif selected_menu == "AI Recommendations":
    show_ai_recommendations(portfolio_df)

elif selected_menu == "Compound Interest":
    show_compound_projection(portfolio_df)

elif selected_menu == "Fundamental Analysis":
    show_fundamental_analysis(portfolio_df)

elif selected_menu == "Risk Analysis":
    show_risk_analysis(portfolio_df)

elif selected_menu == "ESG & Berita":
    show_esg_analysis(portfolio_df)
    show_sentiment_analysis(portfolio_df)

# Simpan histori portofolio
if portfolio_df is not None and uploaded_file is not None:
    history_dir = "portfolio_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_df.to_csv(f"{history_dir}/portfolio_{timestamp}.csv", index=False)
