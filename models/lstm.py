from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import utils

def lstm_model(dataset, training_window, prediction_window, plot=False):
    dataset = dataset.copy()
    dataset = dataset.sort_values(by='t')
    scaler = MinMaxScaler()

    # Normalizar casos
    dataset['target'] = scaler.fit_transform(dataset[['i_cases']])

    X, y = [], []

    # Crear secuencias para LSTM
    for i in range(len(dataset) - training_window - prediction_window + 1):
        seq_x = dataset['target'].iloc[i:i + training_window].values
        seq_y = dataset['target'].iloc[i + training_window + prediction_window - 1]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

    # Dividir en train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Definir el modelo LSTM
    model = keras.Sequential([
        keras.layers.Input(shape=(training_window, 1)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(16),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entrenamiento
    model.fit(X_train, y_train, epochs=200, batch_size=16,
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

    # Predicción
    predictions = model.predict(X_test).flatten()
    predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    loss = utils.loss_function(predictions_inverse.tolist(), y_test_inverse.tolist())

    # Gráfico
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inverse, label='Observed', marker='o')
        plt.plot(predictions_inverse, label='Predicted', marker='x')
        plt.title(f'LSTM - Ventana: {training_window}, Predicción: {prediction_window}')
        plt.xlabel('Muestras')
        plt.ylabel('Casos')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return loss, model, predictions_inverse, y_test_inverse
