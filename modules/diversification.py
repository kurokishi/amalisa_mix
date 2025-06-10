# modules/diversification.py
import pandas as pd

def analisa_diversifikasi(df: pd.DataFrame) -> pd.DataFrame:
    df['Bobot Portofolio (%)'] = df['Nilai Pasar'] / df['Nilai Pasar'].sum() * 100

    def rekomendasi(bobot):
        if bobot > 20:
            return "Jual Sebagian (Terlalu Dominan)"
        elif bobot < 5:
            return "Tambah (Kurang Terwakili)"
        else:
            return "Tahan"

    df['Rekomendasi Diversifikasi'] = df['Bobot Portofolio (%)'].apply(rekomendasi)
    return df[['Kode Saham', 'Lot', 'Nilai Pasar', 'Bobot Portofolio (%)', 'Rekomendasi Diversifikasi']]
