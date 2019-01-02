
import pandas as pd
import numpy as np


from src import finance_data_api as fd, finance_model as fm
from sklearn.preprocessing import MinMaxScaler


class Stock:
    """
    Creates a stock object that has different models in it for prediction.

    Variables:
        ticker (str): the abbreviation of the stock to train (e.g. AAPL, SNAP, AAL)
        model_path (str): the path that the model for the stock is stored
    """

    def __init__(self, ticker: str, model_path: str=None):
        self._ticker = ticker

        # Collect data on stock
        self._data = fd.get_data(ticker)
        self._data['Gap'] = self._data['Close'].values - self._data['Open'].values

        # Make the scaling objects
        self._scalers = {
            'Open': MinMaxScaler(feature_range=(0, 1)),
            'Close': MinMaxScaler(feature_range=(0, 1)),
            'Gap': MinMaxScaler(feature_range=(0, 1))
        }
        for key in self._scalers.keys():
            self._scalers[key].fit(self._data[key].values.reshape(-1,1))

        # Load the models
        try:
            self._models = {
                'Open': fm.load('op', path=model_path),
                'Close': fm.load('cp', path=model_path),
                'Gap':fm.load('gp', path=model_path)
            }
        except FileNotFoundError:
            print(f'No models found for {ticker}.')
        except TypeError:
            print(f'Model path: {ticker}.')

    def data(self) -> pd.DataFrame:
        return self._data

    def get_model(self, model_id):
        return self._models[model_id]

    def predict_next_day(self, model_id: str) -> float:
        inputs = self._data[model_id].values[-60:]
        inputs = inputs.reshape(-1, 1)
        inputs = self._scalers[model_id].transform(inputs)

        X = [inputs]
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        prediction = self._models[model_id].predict(X)
        return self._scalers[model_id].inverse_transform(prediction)[0][0]

    def predict(self, inputs, model_id):
        inputs = inputs.reshape(-1, 1)
        inputs = self._scalers[model_id].transform(inputs)

        X = [inputs]
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        prediction = self._models[model_id].predict(X)
        return self._scalers[model_id].inverse_transform(prediction)[0][0]
