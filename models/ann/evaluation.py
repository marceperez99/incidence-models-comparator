from tensorflow import keras
import matplotlib.pyplot as plt

from metrics.loss_function import loss_function
from .utils import preprocess_ann_data, build_ann


def run_ann(dataset, training_window, prediction_window, architectures, epochs=200, batch_size=32):
    dataset, feature_names = preprocess_ann_data(dataset, training_window, prediction_window)
    train = dataset[dataset['id_proy'] != "CENTRAL-DENGUE-CONFIRMADO"]
    test = dataset[dataset['id_proy'] == "CENTRAL-DENGUE-CONFIRMADO"]

    x_train, x_test, y_train, y_test = train[feature_names].values, test[feature_names].values, train[
        'prediction'].values, test['prediction'].values

    best_loss = float('inf')
    best_model = None
    best_config = None
    best_predictions = None

    for config in architectures:
        print(f"🔍 Probando arquitectura: {config}")

        model = build_ann(len(feature_names), layers_config=config)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(x_test, y_test), callbacks=[early_stop], verbose=0)

        predictions = model.predict(x_test).flatten()
        loss = loss_function(predictions.tolist(), y_test.tolist())

        print(f"📉 Loss actual: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_predictions = predictions.copy()
            best_config = config

        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Observed', marker='o')
        plt.plot(predictions, label='Predicted', marker='x')
        plt.title(f'Mejor Red ANN - Arquitectura: {best_config} | Loss: {best_loss:.4f}')
        plt.xlabel('Muestras')
        plt.ylabel('Número de casos')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"\n✅ Mejor arquitectura: {best_config} con loss: {best_loss}")
    return best_model, best_config, best_loss
