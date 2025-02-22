def autoarima_model(dataset, prediction_window, plot=False):
    dataset = dataset.copy()

    for id_proy in dataset['id_proy'].unique():
        subset = dataset[dataset['id_proy'] == id_proy]
        print(f"Procesando id_proy: {id_proy}, número de filas: {len(subset)}")




