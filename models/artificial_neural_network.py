import keras
import functools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow import keras
import pandas as pd
import utils


def preprocess_ann_data(dataset, lagged_number_of_weeks, prediction_window):
    dataset = dataset.copy()

    # One-hot encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_features = one_hot_encoder.fit_transform(dataset[['disease', 'name', 'classification']])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=one_hot_encoder.get_feature_names_out(['disease', 'name', 'classification'])
    )
    dataset = pd.concat([dataset.reset_index(drop=True), encoded_df], axis=1)

    # Variables temporales
    dataset['timestamp'] = dataset['date'].astype('datetime64[ns]').view('int64') // 10 ** 9
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day

    # Variable objetivo futura
    dataset['prediction'] = dataset['i_cases'].shift(-prediction_window)

    # Crear lags
    for i in range(lagged_number_of_weeks):
        dataset[f'i_cases_{i+1}'] = dataset.groupby(
            ['level', 'name', 'disease', 'classification']
        )['i_cases'].shift(i + 1)

    # Atributos
    lag_cols = [f'i_cases_{i+1}' for i in range(lagged_number_of_weeks)]
    feature_cols = ['timestamp', 'i_cases', 'year', 'month', 'day'] + list(encoded_df.columns) + lag_cols

    # Eliminar filas con NaN
    dataset = dataset.dropna(subset=feature_cols + ['prediction'])

    # Escalar timestamp
    scaler = MinMaxScaler()
    dataset['timestamp'] = scaler.fit_transform(dataset[['timestamp']]).flatten()

    X = dataset[feature_cols].values
    y = dataset['prediction'].values

    return X, y, feature_cols


def build_ann(input_dim, layers_config, activation='relu'):
    model = keras.Sequential()

    # Capa de entrada
    model.add(keras.layers.Dense(layers_config[0], activation=activation, input_shape=(input_dim,)))

    # Capas ocultas
    for units in layers_config[1:]:
        model.add(keras.layers.Dense(units, activation=activation))

    # Capa de salida
    model.add(keras.layers.Dense(1))  # Regresión

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def search_best_ann_architecture(dataset, lagged_number_of_weeks, prediction_window, architectures, epochs=150, batch_size=16):
    X, y, feature_names = preprocess_ann_data(dataset, lagged_number_of_weeks, prediction_window)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_loss = float('inf')
    best_model = None
    best_config = None
    best_predictions = None

    for config in architectures:
        print(f"🔍 Probando arquitectura: {config}")

        model = build_ann(len(feature_names), layers_config=config)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

        predictions = model.predict(X_test).flatten()
        loss = utils.loss_function(predictions.tolist(), y_test.tolist())

        print(f"📉 Loss actual: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_predictions = predictions.copy()
            best_config = config

    # 🔍 Mostrar gráfico de la mejor arquitectura
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Observed', marker='o')
    plt.plot(best_predictions, label='Predicted', marker='x')
    plt.title(f'Mejor Red ANN - Arquitectura: {best_config} | Loss: {best_loss:.4f}')
    plt.xlabel('Muestras')
    plt.ylabel('Número de casos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"\n✅ Mejor arquitectura: {best_config} con loss: {best_loss}")
    return best_model, best_config, best_loss

def ann_evaluator(dataset):
    @functools.cache
    def ann_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        if individual[1] == 0 or individual[1] > 5: return float('inf')
        return ann_model(dataset, individual[0], individual[1])

    return ann_evaluation
