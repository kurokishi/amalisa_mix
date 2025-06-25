
import streamlit as st
from data.processor import process_uploaded_file
from analysis.portfolio import visualize_portfolio
from models.predictor import show_price_prediction
from analysis.what_if import show_what_if_simulation
from analysis.recommendations import show_ai_recommendations
from analysis.compound import show_compound_projection
from analysis.dca import show_dca_simulation
from analysis.long_term_simulation import show_long_term_growth_simulation
from analysis.rebalancing import show_rebalancing_recommendation
from analysis.fundamental import show_fundamental_analysis
from analysis.risk import show_risk_analysis
from analysis.esg import show_esg_analysis
from analysis.sentiment import show_sentiment_analysis
from analysis.reinvest_dividen import show_reinvest_dividen
from analysis.strategy_recommender import show_strategy_recommendation
from analysis.strategy_simulation import show_strategy_simulation
from analysis.strategy_portfolio import show_portfolio_strategy_simulation
from analysis.auto_reallocate import show_auto_reallocation_simulation

# Konfigurasi halaman
st.set_page_config(page_title="📊 Analisis Saham LKH AI", layout="wide")

st.title("📊 Dashboard Analisis Saham LKH + AI Strategy")

# Upload file portofolio
uploaded_file = st.sidebar.file_uploader("📤 Upload File Portofolio (Excel/CSV)", type=["xlsx", "csv"])
portfolio_df = process_uploaded_file(uploaded_file) if uploaded_file else None

# Sidebar Menu
menu_options = {
    "📊 Portfolio Analysis": visualize_portfolio,
    "📈 Price Prediction": show_price_prediction,
    "🧮 What-If Simulation": show_what_if_simulation,
    "🤖 AI Recommendations": show_ai_recommendations,
    "⚖️ Rebalancing": show_rebalancing_recommendation,
    "📚 Fundamental Analysis": show_fundamental_analysis,
    "📉 Risk Analysis": show_risk_analysis,
    "🌱 ESG & Sentiment": lambda df: (show_esg_analysis(df), show_sentiment_analysis(df)),
    "📆 DCA Simulation": show_dca_simulation,
    "🔁 Reinvest Dividen": show_reinvest_dividen,
    "🚀 Long-Term AI Simulation": show_long_term_growth_simulation,
    "🧠 Strategy Recommendation": show_strategy_recommendation,
    "🧪 Strategy Simulation": show_strategy_simulation,
    "📦 Strategy Portfolio Simulation": show_portfolio_strategy_simulation,
    "🔄 Auto Reallocation Simulation": show_auto_reallocation_simulation
}

selected = st.sidebar.selectbox("📌 Pilih Modul Analisis:", list(menu_options.keys()))

# Jalankan modul terpilih
if selected:
    with st.container():
        menu_options[selected](portfolio_df)
