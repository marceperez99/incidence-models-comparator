import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# ==============================
# ⚙️  CONSTANTES CONFIGURABLES
# ==============================
CAT_COLS = ['name', 'level', 'disease', 'classification']
NUM_COLS = ['Population', 'incidence', 'temp_promedio_7d', 'precip_promedio_7d']
DATE_COLS = ['year', 'month', 'day', 'week']
EARLY_STOP_PATIENCE = 4
TUNER_MAX_TRIALS = 50
TUNER_EPOCHS = 35
TUNER_BATCH_SIZE = 16
TUNER_EXECUTIONS = 1
TUNER_DIRECTORY = "temp/mlp_tuner_run"
TUNER_PROJECT = "mlp_forecast"

def mlp_model(dataset, training_window, prediction_window, return_predictions=False):
    from metrics import loss_function  # Tu función custom

    print("🔵 [INICIO] MLP MODEL")
    df = dataset.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week

    print(f"🗂️ Dataset cargado: {df.shape}")

    # 2. Lags y target
    group_cols = CAT_COLS
    LOOK_BACK = training_window
    HORIZON = prediction_window

    for lag in range(1, LOOK_BACK + 1):
        df[f'lag_{lag}'] = df.groupby(group_cols)['i_cases'].shift(lag)
    df['target'] = df.groupby(group_cols)['i_cases'].shift(-HORIZON)

    # 3. Features
    lag_cols = [f'lag_{i}' for i in range(1, LOOK_BACK + 1)]
    feature_cols = NUM_COLS + DATE_COLS + lag_cols

    # 4. Limpieza
    cols_to_dropna = lag_cols + ['target']
    df = df.dropna(subset=cols_to_dropna).reset_index(drop=True)
    print(f"🧹 Dataset tras dropna: {df.shape}")

    # 5. Train/Val Split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]
    print(f"🔶 Train: {train_df.shape} | 🔷 Val: {val_df.shape}")

    # 6. Encoding y Escalado
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[CAT_COLS])
    scaler_X = StandardScaler().fit(train_df[feature_cols])
    scaler_y = StandardScaler().fit(train_df[['target']])

    def build_X(df_sub):
        X_aux = scaler_X.transform(df_sub[feature_cols])
        X_cat = ohe.transform(df_sub[CAT_COLS])
        X_full = np.hstack([X_aux, X_cat])
        return X_full

    X_train = build_X(train_df)
    X_val = build_X(val_df)
    y_train = scaler_y.transform(train_df[['target']]).flatten()
    y_val = scaler_y.transform(val_df[['target']]).flatten()
    y_val_real = val_df['target'].values
    val_dates = val_df['t'].tolist() if 't' in val_df else val_df['date'].tolist()
    print(f"🔺 X_train: {X_train.shape}")
    print(f"🔹 X_val: {X_val.shape}")

    # 7. Keras Tuner Model
    print("🛠️ Buscando mejor arquitectura MLP con Keras Tuner...")

    def model_builder(hp):
        inp = layers.Input(shape=(X_train.shape[1],))
        x = inp
        for i in range(hp.Int("dense_layers", 1, 3)):
            units = hp.Int(f"dense_units_{i}", 32, 256, step=32)
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(rate=hp.Float(f"dropout_dense_{i}", 0.0, 0.4, step=0.1))(x)
        output = layers.Dense(1, activation='linear')(x)

        model = keras.Model(inputs=inp, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling='log')
            ),
            loss='mse'
        )
        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=TUNER_MAX_TRIALS,
        executions_per_trial=TUNER_EXECUTIONS,
        directory=f"{TUNER_DIRECTORY}",
        project_name=f"{TUNER_PROJECT}_{dataset['id_proy'].iloc[0]}_{prediction_window}",
    )

    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE)
    print("🚦 Lanzando tuner.search() ...")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TUNER_EPOCHS,
        batch_size=TUNER_BATCH_SIZE,
        callbacks=[stop_early],
        verbose=0
    )
    print("✅ Keras Tuner terminado")
    best_model = tuner.get_best_models(num_models=1)[0]
    print("🏆 Mejor arquitectura encontrada:")
    tuner.results_summary()

    # 8. Predicción y evaluación
    print("🔮 Generando predicciones de validación...")
    y_pred_scaled = best_model.predict(X_val, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    loss = loss_function.loss_function(list(y_val_real), list(y_pred))
    print(f"🎯 Loss final (custom): {loss:.4f}")
    print(f"🗓️ Fechas de validación: {val_dates[0]} → {val_dates[-1]}")
    print(f"📊 Valores validados: {len(y_val_real)} | Predichos: {len(y_pred)}")

    if return_predictions:
        hp_dict = dict(tuner.get_best_hyperparameters(num_trials=1)[0].values)
        return loss, val_dates, list(y_val_real), list(y_pred), hp_dict
    return loss
