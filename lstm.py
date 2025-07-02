import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt

# 0. Exploración inicial de columnas del dataset
df = pd.read_csv('data/case_data_full.csv')
print("🗂️ Columnas del dataset:", df.columns.tolist())
print("🔎 Ejemplo de datos:\n", df.head())
print("📏 Tipos de datos:\n", df.dtypes)

# Sugerencias de nuevas variables posibles
base_cols = set(['name', 'level', 'disease', 'classification', 'date', 'i_cases'])
sugeridas = [col for col in df.columns if col not in base_cols and not col.startswith('Unnamed')]
print("💡 Variables adicionales sugeridas para incluir:", sugeridas)

# 1. Filtro solo para DENGUE
df = df[
    (df['disease'] == 'DENGUE') &
    (df['name'] == 'CENTRAL') &
    (df['classification'] == 'TOTAL')
].copy()

# 2. Parseo de fechas y nuevas variables temporales
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['semana_epi'] = df['date'].dt.isocalendar().week

# 3. Agregamos lags de meteorología y de i_cases más largos
LOOK_BACK = 12  # ahora hasta 12 semanas para i_cases
HORIZON = 4
group_cols = ['name', 'level', 'disease', 'classification']

for lag in range(1, LOOK_BACK + 1):
    df[f'lag_{lag}'] = df.groupby(group_cols)['i_cases'].shift(lag)
    # lags de temperatura y precipitación (7d)
    df[f'temp_7d_lag_{lag}'] = df.groupby(group_cols)['temp_promedio_7d'].shift(lag)
    df[f'precip_7d_lag_{lag}'] = df.groupby(group_cols)['precip_promedio_7d'].shift(lag)

df['target_raw'] = df.groupby(group_cols)['i_cases'].shift(-HORIZON)

# 4. Aplicar transformación logarítmica SOLO para target > 0 (usamos log1p)
df['target'] = np.log1p(df['target_raw'])

# Preparamos columnas para features
cols_to_dropna = [f'lag_{i}' for i in range(1, LOOK_BACK + 1)] + \
    [f'temp_7d_lag_{i}' for i in range(1, LOOK_BACK + 1)] + \
    [f'precip_7d_lag_{i}' for i in range(1, LOOK_BACK + 1)] + ['target']
df = df.dropna(subset=cols_to_dropna)
print(f"✅ Dataset con ventanas + lags meteo (solo DENGUE): {df.shape}")

# 5. División Train/Val por grupo
df_train, df_val = [], []
for _, group in df.groupby(group_cols):
    group = group.sort_values('date')
    split = int(len(group) * 0.8)
    df_train.append(group.iloc[:split])
    df_val.append(group.iloc[split:])
train_df = pd.concat(df_train)
val_df = pd.concat(df_val)
print(f"📊 Tamaño Train: {train_df.shape}, Validación: {val_df.shape}")

# 6. OneHot + Selección features
cat_cols = ['name', 'level', 'disease', 'classification']
static_cols = ['Population', 'incidence', 'year', 'month', 'semana_epi', 'Population', 'temp_promedio_7d', 'precip_promedio_7d']
static_cols += [col for col in sugeridas if col not in static_cols and df[col].dtype != 'O']


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[cat_cols])

def build_X_lstm(df_subset):
    lags = df_subset[[f'lag_{i}' for i in range(1, LOOK_BACK + 1)]].values
    X_seq = lags.reshape(len(lags), LOOK_BACK, 1)
    static_feats = df_subset[static_cols].values
    cat_feats = ohe.transform(df_subset[cat_cols])
    X_aux = np.hstack([static_feats, cat_feats])
    return X_seq, X_aux

X_seq_train, X_aux_train = build_X_lstm(train_df)
X_seq_val, X_aux_val = build_X_lstm(val_df)
y_train = train_df['target'].values
y_val = val_df['target'].values
y_val_raw = val_df['target_raw'].values  # valores reales para métricas

print("🔍 Dimensiones:")
print("   ↳ X_seq_train:", X_seq_train.shape)
print("   ↳ X_aux_train:", X_aux_train.shape)
print("   ↳ y_train:", y_train.shape)

# 7. Escalado
scaler_X_aux = StandardScaler().fit(X_aux_train)
X_aux_train_scaled = scaler_X_aux.transform(X_aux_train)
X_aux_val_scaled = scaler_X_aux.transform(X_aux_val)

print("📈 Varianzas:")
print("   ↳ Varianza lags:", np.var(X_seq_train))
print("   ↳ Varianza estáticos:", np.var(X_aux_train))
print("   ↳ Varianza target:", np.var(y_train))

# 8. Modelo con Dropout y MAE ponderado
def mae_weighted(y_true, y_pred):
    y_true_exp = tf.math.expm1(y_true)  # volvemos a escala original
    abs_error = tf.math.abs(tf.math.expm1(y_pred) - y_true_exp)
    weights = (y_true_exp + 1) / tf.math.reduce_mean(y_true_exp + 1)
    return tf.math.reduce_mean(abs_error * weights)

def model_builder(hp):
    seq_input = layers.Input(shape=(LOOK_BACK, 1))
    x = seq_input
    num_lstm = hp.Int("lstm_layers", 1, 2)
    for i in range(num_lstm):
        units = hp.Int(f"lstm_units_{i}", 32, 64, step=16)
        return_sequences = i < num_lstm - 1
        x = layers.LSTM(units, return_sequences=return_sequences)(x)
        x = layers.Dropout(0.3)(x)  # Dropout en LSTM
    x = layers.Flatten()(x)

    aux_input = layers.Input(shape=(X_aux_train_scaled.shape[1],))
    y = aux_input
    num_dense = hp.Int("dense_layers", 1, 2)
    for i in range(num_dense):
        units = hp.Int(f"dense_units_{i}", 32, 64, step=16)
        y = layers.Dense(units, activation='relu')(y)
        y = layers.Dropout(0.2)(y)  # Dropout en Dense

    merged = layers.Concatenate()([x, y])
    merged = layers.Dense(hp.Choice("merged_units", [16, 32]), activation='relu')(merged)
    output = layers.Dense(1, activation='linear')(merged)

    model = keras.Model(inputs=[seq_input, aux_input], outputs=output)
    lr = hp.Float("learning_rate", 1e-4, 5e-3, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=mae_weighted,  # MAE ponderado
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

# 9. Búsqueda con Keras Tuner
tuner = kt.RandomSearch(
    model_builder,
    objective="val_mae",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner_lstm_dengue",
    project_name="lstm_forecast_dengue"
)

stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
print("🚀 Iniciando búsqueda de hiperparámetros (DENGUE)...")
tuner.search(
    [X_seq_train, X_aux_train_scaled], y_train,
    validation_data=([X_seq_val, X_aux_val_scaled], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[stop_early],
    verbose=0
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
print("🏆 Mejores hiperparámetros encontrados:", best_hp.values)

# 10. Predicciones y des-transformación
y_pred_log = best_model.predict([X_seq_val, X_aux_val_scaled]).flatten()
y_pred = np.expm1(y_pred_log)  # vuelve a escala original
y_val_real = y_val_raw  # target en escala original

plt.figure(figsize=(8,4))
plt.hist(y_val_real, bins=50, alpha=0.7)
plt.title("Histograma de y_val_real (solo DENGUE)")
plt.xlabel("Casos reales (target desescalado)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

print("🔢 Total de muestras de validación:", len(y_val_real))
print("🔢 Casos reales == 0:", np.sum(y_val_real == 0))
print("🔢 Casos reales < 10:", np.sum(y_val_real < 10))
print("🔢 Casos reales >= 10:", np.sum(y_val_real >= 10))

# 11. Métricas robustas
mae = mean_absolute_error(y_val_real, y_pred)
print(f"📏 MAE real (todos los datos): {mae:.2f}")

mask_no_zero = y_val_real > 1e-6
mape_no_zero = mean_absolute_percentage_error(y_val_real[mask_no_zero], y_pred[mask_no_zero])
print(f"📉 MAPE real (sin ceros): {mape_no_zero*100:.2f}%")

mask_gt10 = y_val_real >= 10
if np.sum(mask_gt10) > 0:
    mape_gt10 = mean_absolute_percentage_error(y_val_real[mask_gt10], y_pred[mask_gt10])
    print(f"📉 MAPE solo para casos >= 10: {mape_gt10*100:.2f}%")
else:
    print("No hay suficientes valores >= 10 para evaluar MAPE segmentado.")

# 12. Visualización de error absoluto por bins
bins = [0, 1, 10, 100, 1000]
labels = ['=0', '1-9', '10-99', '>=100']
df_errors = pd.DataFrame({'y_val_real': y_val_real, 'abs_err': np.abs(y_pred - y_val_real)})
df_errors['bin'] = pd.cut(df_errors['y_val_real'], bins=bins, labels=labels, include_lowest=True)
plt.figure(figsize=(8,5))
df_errors.boxplot(column='abs_err', by='bin', showfliers=False)
plt.title("Error absoluto por rango de casos reales")
plt.suptitle("")
plt.xlabel("Rango de casos reales")
plt.ylabel("Error absoluto")
plt.grid(True)
plt.show()



plt.figure(figsize=(14, 6))
plt.plot(val_df['date'], y_val_real, label='Reales', marker='o')
plt.plot(val_df['date'], y_pred, label='Predichos', marker='x')
plt.xlabel('Fecha')
plt.ylabel('Casos de dengue')
plt.title('Evolución temporal: Casos reales vs predichos (CENTRAL - DENGUE - TOTAL)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()