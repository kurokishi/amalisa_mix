import pandas as pd
import numpy as np


def simple_average_ensemble(predictions: dict):
    """
    Menggabungkan prediksi beberapa model dengan rata-rata sederhana.
    Input: dict nama_model -> DataFrame dengan kolom 'Date' dan 'price'.
    Output: DataFrame prediksi ensemble.
    """
    if not predictions:
        return None

    all_df = []
    for model_name, df in predictions.items():
        df_copy = df.copy()
        df_copy.rename(columns={'price': f'price_{model_name}'}, inplace=True)
        all_df.append(df_copy.set_index('Date'))

    merged = pd.concat(all_df, axis=1, join='inner')
    merged['price'] = merged.mean(axis=1)
    return merged.reset_index()[['Date', 'price']]


def weighted_average_ensemble(predictions: dict, weights: dict):
    """
    Menggabungkan prediksi dengan bobot berdasarkan performa model.
    Input:
        - predictions: dict model -> DataFrame(Date, price)
        - weights: dict model -> bobot (misal dari 1 / RMSE)
    Output: DataFrame hasil ensemble berbobot.
    """
    if not predictions or not weights:
        return None

    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items() if k in predictions}

    all_df = []
    for model_name, df in predictions.items():
        df_copy = df.copy()
        weight = norm_weights.get(model_name, 0)
        df_copy[f'weighted_{model_name}'] = df_copy['price'] * weight
        all_df.append(df_copy.set_index('Date')[[f'weighted_{model_name}']])

    merged = pd.concat(all_df, axis=1, join='inner')
    merged['price'] = merged.sum(axis=1)
    return merged.reset_index()[['Date', 'price']]
