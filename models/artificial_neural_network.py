import keras
from sklearn.preprocessing import OneHotEncoder,  MinMaxScaler
from sklearn.model_selection import train_test_split
import functools
import matplotlib.pyplot as plt
import pandas as pd
import utils


def ann_model(dataset, lagged_number_of_weeks, prediction_window, plot=False):
    # Step 1: One-Hot Encoding for 'disease' and 'name' (nominal data)
    one_hot_encoder = OneHotEncoder(sparse_output=False)  # drop='first' to avoid multicollinearity
    print(dataset)
    encoded_features = one_hot_encoder.fit_transform(dataset[['disease', 'name', 'classification']])
    # Convert the encoded features to a DataFrame
    encoded_df = pd.DataFrame(encoded_features,
                              columns=one_hot_encoder.get_feature_names_out(
                                  ['disease', 'name', 'classification']))

    dataset = pd.concat([dataset, encoded_df], axis=1)

    dataset['timestamp'] = dataset['date'].astype('datetime64[ns]').view('int64') // 10 ** 9
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['prediction'] = dataset['i_cases'].shift(-(prediction_window + 1))

    atributes = ['timestamp', 'i_cases', 'year', 'month', 'day'] + encoded_df.columns.tolist()


    for i in range(lagged_number_of_weeks):
        for i in range(lagged_number_of_weeks):
            dataset[f'i_cases_{i}'] = dataset.groupby(['level', 'name', 'disease', 'classification'])['i_cases'].shift(
                i + 1)
        atributes.append(f'i_cases_{i}')

    dataset = dataset.dropna()

    scaler = MinMaxScaler()
    dataset['timestamp'] = scaler.fit_transform(dataset[['timestamp']]).flatten()


    X = dataset[atributes].values  # Features
    y = dataset['prediction'].values  # Target

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model

    model = keras.Sequential([
        keras.layers.Dense(len(atributes), activation='relu',  input_shape=(len(atributes),)),  # Input layer (1 feature)
        keras.layers.Dense(10, activation='relu'),  # Hidden layer 1
        keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
        keras.layers.Dense(10, activation='relu'),  # Hidden layer 3
        keras.layers.Dense(1)  # Output layer (regression)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=500, batch_size=30, validation_data=(X_test, y_test))

    # Make predictions
    predictions = model.predict(X_test)
    y_pred = [predictions[i][0] for i in range(len(predictions))]
    loss = utils.loss_function(y_pred, y_test)
    if plot:
        predictions = model.predict(X)

        y_pred = [predictions[i][0] for i in range(len(predictions))]


        plt.title(f'Training: {lagged_number_of_weeks}, Prediction: {prediction_window}: {loss}')
        plt.plot(y, label="Real data")
        plt.plot(y_pred, label="Predicted")
        plt.show()




    return loss


def ann_evaluator(dataset):
    @functools.cache
    def ann_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        if individual[1] == 0 or individual[1] > 5: return float('inf')
        return ann_model(dataset, individual[0], individual[1])

    return ann_evaluation
