import pandas as pd
import streamlit as st

# Fungsi untuk memproses file yang diunggah

def process_uploaded_file(uploaded_file):
    """
    Membaca file portofolio dan memastikan format sesuai.
    Menambahkan kolom jumlah saham (Shares) berdasarkan lot.
    """
    if uploaded_file is None:
        return None

    try:
        # Deteksi format file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap upload file .csv atau .xlsx")
            return None

        # Validasi kolom yang dibutuhkan
        required_cols = ["Stock", "Ticker", "Lot Balance", "Avg Price"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"File harus memiliki kolom: {', '.join(required_cols)}")
            return None

        # Hitung jumlah saham berdasarkan lot (1 lot = 100 lembar)
        df['Shares'] = df['Lot Balance'] * 100

        return df

    except Exception as e:
        st.error(f"Gagal memproses file: {str(e)}")
        return None
