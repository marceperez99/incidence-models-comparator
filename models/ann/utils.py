import keras
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow import keras
import pandas as pd


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

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

    # Agrupación base para lags y prediction
    group_keys = ['id_proy', 'level', 'name', 'disease', 'classification']

    # Variable objetivo futura (shift agrupado)
    dataset['prediction'] = dataset.groupby(group_keys)['i_cases'].shift(-prediction_window)

    # Crear lags también agrupados
    for i in range(lagged_number_of_weeks):
        dataset[f'i_cases_{i + 1}'] = dataset.groupby(group_keys)['i_cases'].shift(i + 1)

    # Atributos a usar
    lag_cols = [f'i_cases_{i + 1}' for i in range(lagged_number_of_weeks)]
    feature_cols = ['timestamp', 'i_cases', 'year', 'month', 'day', 'Population', 'incidence',
                    'temp_promedio_7d', 'precip_promedio_7d',
                    'temp_promedio_14d', 'precip_promedio_14d'] + list(encoded_df.columns) + lag_cols

    # Eliminar filas con NaN
    dataset = dataset.dropna(subset=feature_cols + ['prediction'])

    # Escalar variables numéricas
    numeric_cols = ['timestamp', 'i_cases', 'year', 'month', 'day', 'Population', 'incidence',
                    'temp_promedio_7d', 'precip_promedio_7d', 'temp_promedio_14d', 'precip_promedio_14d'] + lag_cols

    scaler = MinMaxScaler()
    dataset[numeric_cols] = scaler.fit_transform(dataset[numeric_cols])

    X = dataset[feature_cols].values
    y = dataset['prediction'].values

    return dataset, feature_cols

def build_ann(input_dim, layers_config, activation='relu'):
    model = keras.Sequential()

    # Capa de entrada
    model.add(keras.layers.Input(input_dim))


    # Capas ocultas
    for units in layers_config[1:]:
        model.add(keras.layers.Dense(units, activation=activation))

    # Capa de salida
    model.add(keras.layers.Dense(1))  # Regresión

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
