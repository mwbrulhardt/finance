import numpy as np
import json

from src import market as mkt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.callbacks import History
from keras.layers import Dense, LSTM
from datetime import datetime


def save(model: Sequential, history: History, identifier: str, path: str) -> None:
    """Saves a model and its training history to a path specified by the user.

    Params:
        model (Sequential): a keras sequential model
        history (History): a keras history object
        identifier (str): an identifier for what the model is used for
        path (str): the path that deals specifically with what the model pertains to
    """

    # Serialize model to JSON
    model_json = model.to_json()
    with open(path + identifier + '_model.json', 'w') as fp:
        fp.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(path + identifier + '_weights.h5')

    # Save model history
    with open(path + identifier + '_history.json', 'w') as f:
        json.dump(history.history, f)


def load(identifier: str, path: str='models/') -> Sequential:
    """Loads a model specified to the path and identifier given.

    Params:
        identifier (str): an identifier to specify the function of the model
            (e.g. 'cp' for closing price)
        path (str): directory where the model is located (e.g. 'models/aapl/')
    Return:
         the model that is specific to the path and identifier

    Example:
        >>>load(identifier='cp', path='models/aapl/')
    """

    # Load json and create model
    json_file = open(path + identifier + '_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights into new model
    model.load_weights(path + identifier + '_weights.h5')
    return model


def train(ticker: str, identifier: str, start: datetime=None, units: int=60, depth: int=3) -> (Sequential, History):
    """Trains an attribute of a stock using a LSTM network model.

    Params:
        ticker (str): the abbreviation of the stock to train (e.g. AAPL, SNAP, AAL)
        identifier (str): attribute of stock to train (e.g. High, Low, Close, Open)
        start (str): date to start training from
        units (int): number of units given to each layer
        depth (int): number of layers for model

    Returns:
        model (Sequential): a sequential keras model
        history (History): a keras history object
    """
    # Creating data frame
    stock = mkt.Stock(ticker)
    aspect_data = stock.data()
    aspect_data = aspect_data[identifier].to_frame()
    aspect_data = aspect_data[aspect_data.index >= start]

    # Creating train and test sets
    data_set = aspect_data.values

    # Converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_set)

    x, y = [], []
    for i in range(60, len(data_set)):
        x.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Create and fit the LSTM network
    model = Sequential()
    for _ in range(depth - 1):
        model.add(LSTM(units=units, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy'])

    history = model.fit(x, y, epochs=1, batch_size=1, validation_split=0.1, verbose=2)

    print(ticker, identifier, 'Training Complete.')

    return model, history
