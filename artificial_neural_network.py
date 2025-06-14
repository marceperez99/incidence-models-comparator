# Instalación de dependencias (ejecutar en tu entorno si aún no tienes keras-tuner)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import keras_tuner as kt

# 1. Carga y parseo de fechas
df = pd.read_csv('data/dengue_confirmado_central.csv')

df['date'] = pd.to_datetime(df['date'])
# 1.a Extraer componentes de la fecha
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# 2. Generación de ventanas (look-back=4) y target a 4 semanas
LOOK_BACK = 4
HORIZON = 4

# Crear lags de i_cases
group_cols = ['name', 'level', 'disease', 'classification']
df = df.sort_values(group_cols + ['date'])
for lag in range(1, LOOK_BACK + 1):
    df[f'lag_{lag}'] = df.groupby(group_cols)['i_cases'].shift(lag)

# Target: casos a HORIZON semanas
df['target'] = df.groupby(group_cols)['i_cases'].shift(-HORIZON)

# Eliminar filas con NaN en lags o target
cols_to_dropna = [f'lag_{i}' for i in range(1, LOOK_BACK + 1)] + ['target']
df = df.dropna(subset=cols_to_dropna)

# 3. Hold‑out: último mes para validación
max_date = df['date'].max()
cutoff = max_date - pd.DateOffset(months=1)

# Primero ordena por fecha
df_sorted = df.sort_values('date')

# Calcula índice de corte en el 80%
split_idx = int(len(df_sorted) * 0.8)

# Divide por posición
train_df = df_sorted.iloc[:split_idx]
val_df = df_sorted.iloc[split_idx:]

# 4. Preparar features y etiqueta
# Variables categóricas
cat_cols = []
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[cat_cols])


def build_X(df_subset):
    lag_feats = df_subset[[f'lag_{i}' for i in range(1, LOOK_BACK + 1)]].values  # (n,4)
    static_feats = df_subset[['Population', 'incidence', 'year', 'month', 'day']].values  # (n,2)
    cat_feats = ohe.transform(df_subset[cat_cols])  # puede venir (n,) o (n,k)

    # Si cat_feats acabase siendo 1-D, lo convertimos a 2-D

    return np.hstack([lag_feats, static_feats, cat_feats])  # ahora todos son (n, m_i)


X_train = build_X(train_df)
X_val = build_X(val_df)
y_train = train_df['target'].values
y_val = val_df['target'].values

print("Shapes → X_train:", X_train.shape, "X_val:", X_val.shape)


# 5. Definir builder para Keras Tuner
def model_builder(hp):
    model = keras.Sequential()
    # Capa de entrada implícita

    # Número de capas ocultas: entre 1 y 3
    for i in range(hp.Int("num_layers", 1, 10)):
        units = hp.Int(f"units_{i}", min_value=16, max_value=128, step=16)
        model.add(keras.layers.Dense(units, activation="relu"))
        dropout_rate = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))

    # Capa de salida (una predicción)
    model.add(keras.layers.Dense(1, activation="linear"))

    # Compilación
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsolutePercentageError(name="mape")]
    )
    return model


# 6. Configurar Keras Tuner (Random Search)
tuner = kt.RandomSearch(
    model_builder,
    objective="val_mape",
    max_trials=20,
    executions_per_trial=1,
    directory="tuner_dir",
    project_name="ts_forecast"
)

# 7. Buscar la mejor arquitectura
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[stop_early],
    verbose=1
)

# 8. Resultados
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparámetros encontrados:")
for param, val in best_hp.values.items():
    print(f"  {param}: {val}")

best_model = tuner.get_best_models(num_models=1)[0]
print("\nEvaluación en validación:")
best_model.evaluate(X_val, y_val, verbose=2)

import matplotlib.pyplot as plt

# Si tu modelo está guardado en disco, cárgalo así:
# from tensorflow.keras.models import load_model
# best_model = load_model("best_model.h5")

# 1. Obtener predicciones
y_pred = best_model.predict(X_val).flatten()

# 2. Recuperar fechas (asegúrate de que val_df['date'] sea datetime)
# val_df es el DataFrame de validación que usaste para X_val/y_val
dates = val_df['date']

# 3. Plot
plt.figure(figsize=(14, 6))
plt.plot(dates, y_val, label="Reales", marker='o')
plt.plot(dates, y_pred, label="Predichos", marker='x')

plt.xlabel("Fecha")
plt.ylabel("Número de casos")
plt.title("Casos reales vs. predichos (horizonte " + str(HORIZON) + " semanas)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
