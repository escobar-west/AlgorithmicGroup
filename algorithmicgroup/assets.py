import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import requests
from functools import partial

plt.ion()

class Indicator:
    def __init__(self, func):
        self.__pfunc = partial(func)
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        return self.__pfunc(*args, **kwargs)

    def __get__(self, obj, cls=None):
        self.__pfunc = partial(self.__pfunc.func, obj)
        return self


class Panel(Indicator):
    def plot(self, *args, **kwargs):
        fig, ax = plt.subplots()

        self(*args, **kwargs).plot(ax=ax, title=self.__name__, legend=True)
        plt.show()


class Level(Indicator):
    def plot(self, *args, **kwargs):
        df = pd.DataFrame(self(*args, **kwargs), index = self.__pfunc.args[0].df.index)
        df.plot(title=self.__name__, legend=True)
        plt.show()
        

class Asset:
    """
    Abstract class to define assets
    """
    @classmethod
    def __dir__(cls):
        return [val for val in dir(cls) if isinstance(
                                              eval(cls.__name__+'.'+val), Indicator)]

    def __repr__(self):
        return '{} from {} to {}'.format(self.name, self.df.index[0], self.df.index[-1])

    def __init__(self, df, name='Asset'):
        """
        :param df: pandas dataframe of asset information
        :param name: string representing the name
        """
        self.df = df
        self.name = name

    @Panel
    def Close(self):
        return self.df['Close'].copy()

    @Panel
    def Volume(self):
        return self.df['Volume'].copy()

    @Panel
    def SMA(self, windows=15):
        """
        :param windows: list of periods to compute averages for
        :returns: Series if windows is an int, DataFrame if windows is a list
        """
        if isinstance(windows, int):
            SMA = self.df['Close'].rolling(window=windows).mean()
            return SMA.rename('SMA')

        else:
            windows.sort()
            input_dict = {f'SMA_{val}':
                           self.df['Close'].rolling(window=val).mean() for val in windows}

            SMA = pd.DataFrame(input_dict, self.df.index)
            return SMA

    @Panel
    def RSI(self, window=14):
        """
        :param window: window to compute averages
        :returns: Series -- pandas Series of RSI values
        """
        change = self.df['Close'].diff().values

        avg_gain = np.array([np.nan] * self.df.shape[0])
        avg_gain[window] = np.sum(np.maximum(change[1:window+1],0)) / window

        avg_loss = np.array([np.nan] * self.df.shape[0])
        avg_loss[window] = np.sum(np.minimum(change[1:window+1],0)) / window

        for i in range(window+1, self.df.shape[0]):
            avg_gain[i] = ((window-1)*avg_gain[i-1] + np.max([change[i],0])) / window
            avg_loss[i] = ((window-1)*avg_loss[i-1] + np.min([change[i],0])) / window

        RS = -avg_gain/avg_loss
        RSI = 100 - 100 / (1+RS)
        RSI = pd.Series(data = RSI, index = self.df.index, name = 'RSI')
        return RSI

    @Panel 
    def OBV(self):
        """
        :returns: Series -- pandas Series of OBV values
        """
        sign = np.sign(self.df['Close'].diff())
        sign[0] = 0

        OBV = self.df['Volume'] * sign / 1e6
        OBV = np.cumsum(OBV)

        return OBV.rename('OBV')
        
    @Level
    def maxmin(self):
        return {'min': min(self.df['Low']), 'max': max(self.df['High'])}


class Stock(Asset):
    """
    Stock class
    """
    def __repr__(self):
        return f'{self.name} from {self.start} to {self.end}'

    def __init__(self,
                 name='SPY',
                 start=dt.date.today() - dt.timedelta(days=30),
                 end=dt.date.today(),
                 engine='yahoo'):
        """
        :param name: stock name name
        :param start: date object to start the query
        :param end: end period
        :param engine: engine to use for data_reader
        :returns: Stock -- object containing stock info and Indicators
        """
        df = pdr.DataReader(name, engine, start, end)

        # Deletes an extra field from the yahoo engine we don't need
        try:
            del df['Adj Close']
        except KeyError:
            pass

        super().__init__(df, name)
        self.start = self.df.index[0].date()
        self.end = self.df.index[-1].date()

    @Level
    def _dev(self):
        print('This Indicator is for development purposes!')
        return pd.Series(np.arange(self.df.shape[0]), self.df.index).rename('dev')


class FX(Asset):
    def __init__(self,
                 name='BTC-USD',
                 start=dt.date.today() - dt.timedelta(days=30),
                 end=dt.date.today()):

        date_list = [start]
        while start + dt.timedelta(days=200) < end:
            start += dt.timedelta(days=200)
            date_list.append(start)
        date_list.append(end)

        df_list = []
        for start_, end_ in zip(date_list[:-1], date_list[1:]):

            start = '{}/{}/{}'.format(start_.year, start_.month, start_.day)
            end = '{}/{}/{}'.format(end_.year, end_.month, end_.day)

            url = ('https://api.gdax.com/products/' + name
                   + '/candles?start=' + start + '&end=' + end
                   + '&granularity=86400')

            r = requests.get(url)

            data = np.array(r.json())
            data = data[np.argsort(data[:,0]),:]
            df_list.append(pd.DataFrame(data[:,1:], pd.to_datetime(data[:,0], unit='s'),
                              ['Low','High','Open','Close','Volume'])
                          )
        df = pd.concat(df_list)
        df.index.name = 'Date'

        super().__init__(df, name)

        self.start = self.df.index[0]
        self.end = self.df.index[-1]

