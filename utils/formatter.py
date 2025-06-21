def format_currency_idr(value):
    """
    Mengubah angka menjadi format mata uang Rupiah.
    Contoh: 1000000 -> Rp1.000.000
    """
    try:
        if value is None:
            return "-"
        return f"Rp{value:,.0f}".replace(",", ".")
    except:
        return str(value)
