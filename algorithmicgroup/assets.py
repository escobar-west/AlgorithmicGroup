import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import requests
import pdb


class Asset:
    """
    Abstract class to define assets
    """
    @classmethod
    def __dir__(cls):
        return [val for val in dir(cls) if val[0] != '_']

    def __repr__(self):
        return '{} from {} to {}'.format(self.ticker, self.df.index[0], self.df.index[-1])

    def __init__(self, df, ticker='Asset'):
        """
        :param df: pandas dataframe of asset information
        :param ticker: string representing the ticker
        """
        self.df = df
        self.ticker = ticker

    def SMA(self, windows = 15):
        """
        :param windows: list of periods to compute averages for
        :returns: Series if windows is an int, DataFrame if windows is a list
        """
        if isinstance(windows, int):
            SMA = self.df.Close.rolling(window=windows).mean()
            return SMA.rename('SMA')

        else:
            windows.sort()
            input_dict = {f'SMA_{val}':
                           self.df.Close.rolling(window=val).mean() for val in windows}

            SMA = pd.DataFrame(input_dict, self.df.index)
            return SMA

    def RSI(self, window=14):
        """
        :param window: window to compute averages
        :returns: Series -- pandas Series of RSI values
        """
        change = self.df.Close.diff().values

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

    def OBV(self):
        """
        :returns: Series -- pandas Series of OBV values
        """
        sign = np.sign(self.df.Close.diff())
        sign[0] = 0

        OBV = self.df.Volume * sign / 1e6
        OBV = np.cumsum(OBV)

        return OBV.rename('OBV')
        
class Stock(Asset):
    """
    Stock class
    """
    def __repr__(self):
        return f'{self.ticker} from {self.start} to {self.end}'

    def __init__(self,
                 ticker='SPY',
                 start=dt.date.today() - dt.timedelta(days=30),
                 end=dt.date.today(),
                 engine='yahoo'):
        """
        :param ticker: stock ticker name
        :param start: date object to start the query
        :param end: end period
        :param engine: engine to use for data_reader
        :returns: Stock -- object containing stock info and indicators
        """
        df = pdr.DataReader(ticker, engine, start, end)
        super().__init__(df, ticker)

        try:
            del self.df['Adj Close']
        except KeyError:
            pass

        self.start = self.df.index[0].date()
        self.end = self.df.index[-1].date()

    def _dev(self):
        print('This indicator is for development purposes!')
        return pd.Series(np.arange(self.df.shape[0]), self.df.index).rename('dev')

class FX(Asset):
    def __init__(self,
                 ticker='BTC-USD',
                 start=dt.date.today() - dt.timedelta(days=30),
                 end=dt.date.today()):

        start = '{}/{}/{}'.format(start.year, start.month, start.day)
        end = '{}/{}/{}'.format(end.year, end.month, end.day)

        url = ('https://api.gdax.com/products/' + ticker
               + '/candles?start=' + start + '&end=' + end
               + '&granularity=86400')

        r = requests.get(url)

        data = np.array(r.json())
        data = data[np.argsort(data[:,0]),:]
        df = pd.DataFrame(data[:,1:], pd.to_datetime(data[:,0], unit='s'),
                          ['Low','High','Open','Close','Volume'])
        df.index.name = 'Date'

        super().__init__(df, 'BTCUSD')

        self.start = self.df.index[0]
        self.end = self.df.index[-1]
        self.ticker = ticker

