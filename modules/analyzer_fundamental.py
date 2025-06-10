# modules/analyzer_fundamental.py
import yfinance as yf
import pandas as pd

def fetch_fundamental_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker + ".JK")
    info = stock.info

    try:
        return {
            "PER": info.get("trailingPE", None),
            "PBV": info.get("priceToBook", None),
            "EPS": info.get("trailingEps", None),
            "Dividen Yield (%)": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "ROE (%)": (info.get("returnOnEquity", 0) or 0) * 100,
            "Book Value": info.get("bookValue", None),
            "Target Price": info.get("targetMeanPrice", None),
            "Harga Saat Ini": info.get("currentPrice", None)
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return {}

def valuasi_diskon(current_price, book_value, eps, expected_return=0.15):
    """
    Metode valuasi konservatif ala Lo Kheng Hong
    """
    try:
        intrinsic_value = book_value + (eps * 8)  # konservatif
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value * 100
        expected_annual_return = ((intrinsic_value / current_price) ** (1 / 5)) - 1  # CAGR 5 tahun
        return round(intrinsic_value, 2), round(margin_of_safety, 2), round(expected_annual_return * 100, 2)
    except:
        return None, None, None

def analisa_saham(ticker: str) -> dict:
    data = fetch_fundamental_data(ticker)
    if not data:
        return {}

    intrinsic_value, margin_of_safety, cagr = valuasi_diskon(
        current_price=data["Harga Saat Ini"],
        book_value=data["Book Value"],
        eps=data["EPS"]
    )

    data["Nilai Wajar (LKH)"] = intrinsic_value
    data["Margin of Safety (%)"] = margin_of_safety
    data["CAGR Estimasi (%)"] = cagr

    # Kriteria sederhana untuk rekomendasi
    data["Rekomendasi"] = "BUY" if (margin_of_safety and margin_of_safety >= 30 and data["PBV"] <= 1.2) else "HOLD"

    return data
