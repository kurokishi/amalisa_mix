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
from analysis.dca import show_dca_simulation
from analysis.long_term_simulation import show_long_term_growth_simulation
from analysis.rebalancing import show_rebalancing_recommendation
from analysis.reinvest_dividen import show_reinvest_dividen
from analysis.strategy_recommender import show_strategy_recommendation
from analysis.strategy_simulation import show_strategy_simulation
from analysis.strategy_portfolio import show_portfolio_strategy_simulation
from models.predictor import show_price_prediction

# Konfigurasi halaman
st.set_page_config(layout="wide", page_title="📊 AI Portfolio Management Dashboard", page_icon="📈")
st.title("🪙 AI Portfolio Management Dashboard")

# Sidebar: Navigasi & Upload
with st.sidebar:
    st.header("🧭 Navigasi Modul")
    menu_options = {
        "📊 Portfolio Analysis": visualize_portfolio,
        "📈 Price Prediction": show_price_prediction,
        "🧮 What-If Simulation": show_what_if_simulation,
        "🤖 AI Recommendations": show_ai_recommendations,
        "📉 Compound Interest": show_compound_projection,
        "📆 DCA Simulation": show_dca_simulation,
        "🔁 Reinvest Dividen": show_reinvest_dividen,
        "🧠 Strategy Recommendation": show_strategy_recommendation,
        "🧪 Strategy Simulation": show_strategy_simulation,
        "📦 Strategy Portfolio Simulation": show_portfolio_strategy_simulation,
        "🚀 Long-Term AI Simulation": show_long_term_growth_simulation,
        "⚖️ Rebalancing": show_rebalancing_recommendation,
        "📚 Fundamental Analysis": show_fundamental_analysis,
        "📉 Risk Analysis": show_risk_analysis,
        "🌱 ESG & Berita": lambda df: (show_esg_analysis(df), show_sentiment_analysis(df))
    }

    selected_menu = st.selectbox("📌 Pilih Modul Analisis:", list(menu_options.keys()))

    st.divider()
    st.header("📤 Upload Portfolio")
    uploaded_file = st.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx"])

    st.divider()
    st.header("⚙️ Parameter Prediksi")
    prediction_days = st.slider("🔮 Hari Prediksi Harga", 7, 365, 30)
    risk_tolerance = st.select_slider("💡 Toleransi Risiko", options=["Low", "Medium", "High"])

# Proses file upload
portfolio_df = process_uploaded_file(uploaded_file)

if uploaded_file and portfolio_df is not None:
    st.success("✅ Portofolio berhasil diproses!")
    with st.expander("📄 Lihat Portofolio Saat Ini"):
        st.dataframe(portfolio_df)

    # Jalankan fungsi dari menu yang dipilih
    st.markdown("---")
    st.subheader(selected_menu)
    selected_function = menu_options[selected_menu]
    if selected_menu == "📈 Price Prediction":
        selected_function(portfolio_df, prediction_days)
    else:
        selected_function(portfolio_df)

else:
    st.info("📥 Silakan upload file portofolio terlebih dahulu untuk memulai analisis.")

# Simpan histori portofolio
if portfolio_df is not None and uploaded_file is not None:
    history_dir = "portfolio_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_df.to_csv(f"{history_dir}/portfolio_{timestamp}.csv", index=False)
