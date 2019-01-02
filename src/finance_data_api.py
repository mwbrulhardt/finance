"""
The purpose of this module is to query financial data from different source
and consolidate it into one place that things will be called from.
"""
import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data

ASPECT_ABV = {
    'High': 'hp',
    'Low': 'lp',
    'Open': 'op',
    'Close': 'cp',
    'Volume': 'v',
    'Adj Close': 'ac'
}


def get_data(ticker: str, start: dt.datetime=None, end: dt.datetime=None, identifier=None) -> pd.DataFrame:

    try:
        start = dt.datetime(1920, 1, 1) if start is None else start

        if identifier is None:
            return data.DataReader(ticker, 'yahoo', start=start, end=end)
        elif identifier in ASPECT_ABV.keys():
            aspect_data = data.DataReader(ticker, 'yahoo', start=start, end=end)
            return aspect_data[identifier].to_frame()
        raise ValueError('Stock has no such identifier.')

    except ValueError as e:
        print(e)
    except KeyError as e:
        print('Date are not in range of when the stock has been tracked.')


def get_last_n_values(n: int, ticker: str, identifier: str, date: dt.datetime=None) -> np.array:
    """
    Params:
        n:
        ticker:
        identifier:
        date:
    Returns:
        a numpy array with shape (n, 1)
    """
    if date is None:
        stock_data = get_data(ticker=ticker, identifier=identifier)
        return stock_data[len(stock_data) - n:].values
    stock_data = get_data(ticker=ticker, end=date, identifier=identifier)
    stock_data = stock_data[stock_data.index <= date]
    return stock_data[len(stock_data) - n:].values


def get_today_date() -> str:
    return str(dt.datetime.today().date())


def get_data_on_date(date: str, ticker: str, start: str, end: str, aspect=None) -> list:
    stock_data = get_data(ticker, start, end, aspect)
    return stock_data[stock_data.index == date].values