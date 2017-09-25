import numpy as np
import pandas as pd

class Signal:
    def __init__(self, func):
        self.__func = func
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

    def backtest(self, principal, assets, *args, **kwargs):
        no_neutral = kwargs.pop('no_neutral', False)
        signals = pd.DataFrame(self(assets, *args, **kwargs))

        if no_neutral:
            for col in signals.columns:
                signals[col].replace(to_replace=0, method='ffill', inplace=True)

        if not isinstance(assets, list):
            assets = [assets]

        principal /= len(assets)
        log_close = [np.log(A.df.Close).rename(A.name+'_Close') for A in assets]
        log_close = pd.concat(log_close, axis=1)

        log_gain = np.zeros(log_close.shape)
        log_gain[0] = np.log(principal)
        log_gain[1:] = log_close.diff().values[1:]*signals.values[:-1].reshape(-1,log_close.shape[1])
        log_gain[np.isnan(log_gain)] = 0

        port_gain = np.exp(log_gain.cumsum(axis=0))
        port_gain = pd.DataFrame(port_gain, index=log_close.index,
                                 columns=[A.name+'_value' for A in assets]
                                )
        if port_gain.shape[1] == 1:
            return pd.concat([np.exp(log_close), signals, port_gain], axis=1)
                   
        else:
            final = pd.Series(port_gain.sum(axis=1),
                              name='final_value',
                              index=port_gain.index
                             )
            return pd.concat([np.exp(log_close), signals, port_gain, final], axis=1)


@Signal
def RSI_signal(asset, window=14, overbought=70.0, oversold=30.0):

    signal = np.array([np.nan] * asset.df.shape[0])
    signal[asset.RSI(window) >= overbought] = -1
    signal[asset.RSI(window) <= oversold] = 1
    signal[np.isnan(signal)] = 0

    signal = pd.Series(signal, asset.df.index, name='RSI_signal')
    return signal


@Signal
def SMA_cross(asset, quick=30, slow=100):
    if quick > slow:
        (quick, slow) = (slow, quick)

    df = asset.SMA([quick, slow])
 
    signal = np.array([np.nan] * df.shape[0])
    signal[df[f'SMA_{quick}'] >= df[f'SMA_{slow}']] =  1
    signal[df[f'SMA_{quick}'] <= df[f'SMA_{slow}']] = -1
    signal[np.isnan(signal)] = 0

    signal = pd.Series(signal, df.index, name=f'SMA_cross_{quick}_{slow}')
    return signal


@Signal
def pairs(assets, tol=1):
    asset1, asset2 = assets
    price_diff = asset2.df.Close - asset1.df.Close
    price_diff = (price_diff - price_diff.mean())/price_diff.std()
    
    signal = np.array([[np.nan, np.nan]] * price_diff.shape[0])
    signal[price_diff >= tol, :] = np.array([1, -1])
    signal[price_diff <= -tol, :] = np.array([-1, 1])
    signal[np.isnan(signal)] = 0

    signal = pd.DataFrame(signal, price_diff.index,
                          columns=[asset1.name + '_pair', asset2.name + '_pair']
                         )
    return signal


@Signal
def buy_hold(assets):
    if not isinstance(assets, list):
        return pd.Series(1, assets.df.index, name='buy_hold')

    else:
        index = pd.concat([A.df.Close for A in assets], axis=1).index
        signal = np.ones([m.shape[0], len(assets)])
        signal = pd.DataFrame(signal, index, [A.name for A in assets])
        return signal
