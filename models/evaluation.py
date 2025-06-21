import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def evaluate_prediction(y_true, y_pred):
    """
    Menghitung metrik evaluasi prediksi: MAE, RMSE, MAPE.
    Input berupa dua array harga aktual dan prediksi.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def compare_models(y_true, predictions_dict):
    """
    Menerima dict: {nama_model: array_prediksi}, dan membandingkan akurasinya.
    Output berupa DataFrame skor evaluasi tiap model.
    """
    results = []
    for name, y_pred in predictions_dict.items():
        mae, rmse, mape = evaluate_prediction(y_true, y_pred)
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape
        })
    return pd.DataFrame(results).sort_values(by='RMSE')
