import streamlit as st
import pandas as pd
import yfinance as yf
from utils.formatter import format_currency_idr
from data.fetcher import get_fundamental_data


def generate_simple_recommendation(ticker, avg_price):
    try:
        current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        pl_pct = (current_price - avg_price) / avg_price * 100

        if pl_pct > 25:
            return "Jual", "Target profit tercapai"
        elif pl_pct < -15:
            return "Beli", "Peluang average down"
        else:
            return "Tahan", "Posisi netral"
    except:
        return "Tahan", "Data tidak tersedia"


def show_ai_recommendations(portfolio_df):
    if portfolio_df is None:
        st.warning("Silakan upload portofolio terlebih dahulu")
        return

    st.header("Rekomendasi AI")
    tab1, tab2 = st.tabs(["Portofolio Saat Ini", "Penambahan Sektor"])

    with tab1:
        recommendations = []
        for _, row in portfolio_df.iterrows():
            rec, reason = generate_simple_recommendation(row['Ticker'], row['Avg Price'])
            current_price = yf.Ticker(row['Ticker']).history(period='1d')['Close'].iloc[-1]
            pl_pct = (current_price - row['Avg Price']) / row['Avg Price'] * 100
            recommendations.append({
                'Saham': row['Stock'],
                'Harga Sekarang': format_currency_idr(current_price),
                'Harga Rata': format_currency_idr(row['Avg Price']),
                'P/L %': f"{pl_pct:.2f}%",
                'Rekomendasi': rec,
                'Alasan': reason
            })
        df = pd.DataFrame(recommendations)
        st.dataframe(df.style.applymap(
            lambda x: 'background-color: lightgreen' if x == 'Beli' else
                     ('background-color: salmon' if x == 'Jual' else 'background-color: lightyellow'),
            subset=['Rekomendasi']
        ))

    with tab2:
        st.subheader("Rekomendasi Saham dari Sektor Kurang Terwakili")

        # Hitung alokasi sektor saat ini
        sektor_value = {}
        total_val = 0
        for _, row in portfolio_df.iterrows():
            f = get_fundamental_data(row['Ticker'])
            if not f: continue
            sektor = f.get('Sektor', 'Lainnya')
            val = row['Shares'] * yf.Ticker(row['Ticker']).history(period='1d')['Close'].iloc[-1]
            sektor_value[sektor] = sektor_value.get(sektor, 0) + val
            total_val += val

        for k in sektor_value:
            sektor_value[k] = sektor_value[k] / total_val * 100

        st.bar_chart(pd.DataFrame.from_dict(sektor_value, orient='index', columns=['Alokasi (%)']))

        underrepresented = [s for s, v in sektor_value.items() if v < 10]
        all_sectors = ["Financial Services", "Energy", "Consumer Defensive", "Healthcare",
                       "Technology", "Basic Materials", "Communication Services", "Industrials"]
        missing = [s for s in all_sectors if s not in sektor_value]
        target_sectors = list(set(underrepresented + missing))

        st.markdown(f"**Sektor direkomendasikan:** {', '.join(target_sectors)}")

        sector_stocks = {
            "Financial Services": ["BBRI.JK", "BMRI.JK"],
            "Energy": ["PGAS.JK", "PTBA.JK"],
            "Consumer Defensive": ["UNVR.JK", "ICBP.JK"],
            "Healthcare": ["KLBF.JK", "KAEF.JK"],
            "Technology": ["MTEL.JK", "DCII.JK"],
            "Basic Materials": ["ANTM.JK", "INCO.JK"],
            "Communication Services": ["ISAT.JK", "EXCL.JK"],
            "Industrials": ["ASII.JK", "WEGE.JK"]
        }

        result = []
        for s in target_sectors:
            for ticker in sector_stocks.get(s, []):
                if ticker in portfolio_df['Ticker'].values:
                    continue
                f = get_fundamental_data(ticker)
                if f and f.get('PER') and f.get('Dividend Yield'):
                    result.append({
                        'Sektor': s,
                        'Kode': ticker,
                        'PER': f['PER'],
                        'Div. Yield (%)': round(f['Dividend Yield'] * 100, 2) if f['Dividend Yield'] else 0,
                        'ROE (%)': round(f['ROE'] * 100, 2) if f['ROE'] else 0,
                        'Market Cap': format_currency_idr(f['Market Cap'])
                    })

        if result:
            df_res = pd.DataFrame(result).sort_values(by=['PER', 'Div. Yield (%)'], ascending=[True, False])
            st.dataframe(df_res)
        else:
            st.info("Tidak ada rekomendasi tersedia.")
