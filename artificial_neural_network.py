import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt

print("🔹 [1] Cargando datos...")
df = pd.read_csv('data/case_data_full.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Parámetros
LOOK_BACK = 8
HORIZON = 4
group_cols = ['name', 'level', 'disease', 'classification']

print("🔹 [2] Generando variables de ventana y objetivo...")
df = df.sort_values(group_cols + ['date'])
for lag in range(1, LOOK_BACK + 1):
    df[f'lag_{lag}'] = df.groupby(group_cols)['i_cases'].shift(lag)
df['target'] = df.groupby(group_cols)['i_cases'].shift(-HORIZON)
cols_to_dropna = [f'lag_{i}' for i in range(1, LOOK_BACK + 1)] + ['target']
df = df.dropna(subset=cols_to_dropna)
print(f"   ↳ Tamaño dataset luego de lags y target: {df.shape}")

# Split temporal por grupo
print("🔹 [3] Dividiendo train/val por grupo geográfico...")
train_df_list, val_df_list = [], []
for name, group in df.groupby(group_cols):
    group = group.sort_values('date')
    split_idx = int(len(group) * 0.8)
    train_df_list.append(group.iloc[:split_idx])
    val_df_list.append(group.iloc[split_idx:])
train_df = pd.concat(train_df_list)
val_df = pd.concat(val_df_list)
print(f"   ↳ Tamaño train: {train_df.shape}, Tamaño val: {val_df.shape}")

# Variables categóricas y features
cat_cols = ['name', 'level', 'disease', 'classification']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[cat_cols])

def build_X(df_subset):
    lag_feats = df_subset[[f'lag_{i}' for i in range(1, LOOK_BACK + 1)]].values
    static_feats = df_subset[['Population', 'incidence', 'year', 'month', 'day']].values
    cat_feats = ohe.transform(df_subset[cat_cols])
    return np.hstack([lag_feats, static_feats, cat_feats])

print("🔹 [4] Preparando features y targets...")
X_train = build_X(train_df)
X_val = build_X(val_df)
y_train = train_df['target'].values
y_val = val_df['target'].values

# Escalado
print("🔹 [5] Escalando datos...")
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
y_train_scaled = scaler_y.transform(y_train.reshape(-1,1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1,1)).flatten()

print("   ↳ Varianza en lags:", np.var(X_train[:, :LOOK_BACK]))
print("   ↳ Varianza en y_train:", np.var(y_train))

# Modelo (dejo tu tuner, pero podría ser un modelo fijo)
def model_builder(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 1, 4)):
        units = hp.Int(f"units_{i}", 2, 64, step=16)
        model.add(keras.layers.Dense(units, activation="relu"))
        dropout_rate = hp.Float(f"dropout_{i}", 0.0, 0.3, step=0.1)
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation="linear"))
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsolutePercentageError(name="mape")]
    )
    return model

print("🔹 [6] Buscando mejor modelo con Keras Tuner...")
tuner = kt.RandomSearch(
    model_builder,
    objective="val_mape",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner_dir",
    project_name="ts_forecast"
)
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=16,
    callbacks=[stop_early],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
print("🔹 [7] Mejor modelo encontrado:", best_hp.values)

print("🔹 [8] Obteniendo predicciones... (solo para CENTRAL-DENGUE-TOTAL)")
# Filtrar solo CENTRAL, DENGUE, TOTAL en val_df para graficar






y_pred_scaled = best_model.predict(X_val_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

dates = val_df['date']

print(f"   ↳ Graficando {len(y_val)} puntos de validación para CENTRAL/DENGUE/TOTAL...")
plt.figure(figsize=(14, 6))
plt.plot(dates, y_val, label="Reales")
plt.plot(dates, y_pred, label="Predichos")
plt.xlabel("Fecha")
plt.ylabel("Número de casos")
plt.title(f"Casos reales vs. predichos (horizonte {HORIZON} semanas)\nCENTRAL, DENGUE, TOTAL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
