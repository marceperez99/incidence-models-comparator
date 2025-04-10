from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

from metrics.loss_function import loss_function
from models.lstm.utils import preprocess_lstm_sequences


def run_lstm(
    dataset,
    test_id_proy: str = "CENTRAL-DENGUE-CONFIRMADO",
    sequence_length: int = 8,
    prediction_window: int = 1,
    epochs: int = 150,
    batch_size: int = 32
):
    """
    Entrena y evalúa un modelo LSTM sobre series temporales con separación por id_proy.

    Args:
        dataset (DataFrame): Datos originales.
        test_id_proy (str): ID del proyecto a usar como test.
        sequence_length (int): Longitud de cada secuencia (timesteps).
        prediction_window (int): A cuántos pasos predecir.
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño de lote.
    """
    # Preprocesamiento
    X, y, feature_names, id_series = preprocess_lstm_sequences(dataset, sequence_length, prediction_window)

    # Separar train y test
    X_train = X[id_series != test_id_proy]
    y_train = y[id_series != test_id_proy]
    X_test = X[id_series == test_id_proy]
    y_test = y[id_series == test_id_proy]

    # Modelo LSTM

    # model = Sequential([
    #     Input(shape=(sequence_length, X.shape[2])),
    #     LSTM(32, return_sequences=True),  # captura secuencia completa
    #     Dropout(0.3),  # regularización
    #     LSTM(16),  # reducción para generalizar
    #     Dense(8, activation='relu'),  # capa intermedia
    #     Dense(1)  # salida (regresión)
    # ])
    model = Sequential([
        Input(shape=(5, X.shape[2])),  # 5 timesteps, n features
        LSTM(32, return_sequences=False),  # más unidades que el original
        Dropout(0.2),  # regularización
        Dense(16, activation='relu'),
        Dense(1)  # salida para regresión
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])


    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print(f"🔹 Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    # Predicción y evaluación
    y_pred = model.predict(X_test).flatten()
    y_pred = y_pred.tolist()
    y_test = y_test.tolist()

    loss = loss_function(y_pred, y_test)

    print(f"📉 LSTM Loss: {loss:.4f}")

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Observado')
    plt.plot(y_pred, label='Predicho')
    plt.title('LSTM - Observado vs Predicho')
    plt.xlabel('Muestras')
    plt.ylabel('Número de casos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return loss
