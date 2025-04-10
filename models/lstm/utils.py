import pandas as pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple

def preprocess_lstm_sequences(
    dataset: pd.DataFrame,
    sequence_length: int,
    prediction_window: int = 1
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Preprocesa un DataFrame para usarlo como entrada de un modelo LSTM.

    Args:
        dataset (pd.DataFrame): Dataset original.
        sequence_length (int): Número de pasos de entrada (timesteps).
        prediction_window (int): Cuántos pasos adelante se desea predecir.

    Returns:
        X (np.ndarray): Secuencias de entrada para LSTM.
        y (np.ndarray): Valores objetivos.
        feature_names (list): Nombres de columnas de entrada.
        id_series (np.ndarray): ID de cada secuencia para posible partición.
    """
    dataset = dataset.copy()

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(dataset[['disease', 'name', 'classification']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(['disease', 'name', 'classification'])
    )
    dataset = pd.concat([dataset.reset_index(drop=True), encoded_df], axis=1)

    # Features a usar
    feature_cols = [ 'Population', 'incidence', 'csum', 't'] + list(encoded_df.columns)

    # Eliminar filas con NaN en features
    dataset = dataset.dropna(subset=feature_cols)

    # Escalado
    scaler = MinMaxScaler()
    dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

    # Inicializar listas de salida
    X, y, id_proy_list = [], [], []

    # Generar secuencias por grupo
    for _, group in dataset.groupby('id_proy'):
        group = group.sort_values(by='t')

        # Calcular target con shift (puede generar NaN)
        group['target'] = group['i_cases'].shift(-prediction_window)
        group = group.dropna(subset=['target'])

        data = group[feature_cols].values
        targets = group['target'].values

        if len(data) < sequence_length + prediction_window:
            continue  # ignorar grupos muy pequeños

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(targets[i + sequence_length - 1])  # valor a predecir
            id_proy_list.append(group['id_proy'].iloc[i + sequence_length - 1])

    return np.array(X), np.array(y), feature_cols, np.array(id_proy_list)


def split_train_test_by_id_proy(dataset: pd.DataFrame, test_id_proy: str):
    """
    Separa el dataset en conjunto de entrenamiento y testeo en base al valor de `id_proy`.

    Args:
        dataset (pd.DataFrame): El dataset preprocesado.
        test_id_proy (str): El valor de `id_proy` que se usará como test.

    Returns:
        train_df (pd.DataFrame): Subconjunto para entrenamiento.
        test_df (pd.DataFrame): Subconjunto para testeo.
    """
    train_df = dataset[dataset['id_proy'] != test_id_proy].copy()
    test_df = dataset[dataset['id_proy'] == test_id_proy].copy()
    return train_df, test_df
