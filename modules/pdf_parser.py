# modules/pdf_parser.py
import pdfplumber
import pandas as pd
import re

def parse_portfolio_pdf(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()

    # Ambil bagian tabel portofolio saham
    pattern = r"(\d+)\s([A-Z]+)-.*?(\d+,\d{2}|\d+)\s+(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,-]+)"
    matches = re.findall(pattern, text)

    data = []
    for match in matches:
        _, kode, lot, _, avg_price, stock_val, market_price, market_val, unrealized = match
        data.append({
            "Kode Saham": kode,
            "Lot": int(lot),
            "Harga Beli": int(avg_price.replace(',', '')),
            "Nilai Beli": int(stock_val.replace(',', '')),
            "Harga Pasar": int(market_price.replace(',', '')),
            "Nilai Pasar": int(market_val.replace(',', '')),
            "Laba/Rugi": int(unrealized.replace(',', '').replace('-', '-'))
        })

    return pd.DataFrame(data)
