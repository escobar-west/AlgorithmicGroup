import numpy as np
import pandas as pd

def RSI_signal(asset, window=14, overbought=70.0, oversold=30.0):
    signal = np.array([np.nan] * asset.df.shape[0])
    signal[asset.RSI(window) >= overbought] = -1
    signal[asset.RSI(window) <= oversold] = 1
    signal[np.isnan(signal)] = 0

    signal = pd.Series(signal, asset.df.index, name='RSI_signal')

    return signal

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

def pairs(assets, tol=1):
    asset1, asset2 = assets
    price_diff = asset2.df.Close - asset1.df.Close
    price_diff = (price_diff - price_diff.mean())/price_diff.std()
    
    signal = np.array([[np.nan, np.nan]] * price_diff.shape[0])
    signal[price_diff >= tol, :] = np.array([1, -1])
    signal[price_diff <= -tol, :] = np.array([-1, 1])
    signal[np.isnan(signal)] = 0

    signal = pd.DataFrame(signal, price_diff.index,
                          columns=[asset1.ticker + '_pair', asset2.ticker + '_pair'])

    return signal

def buy_hold(assets):
    if ~isinstance(assets, list):
        print(isinstance(assets, list ))
        return pd.Series(1, assets.df.index, name='buy_hold')

    else:
        blah = pd.concat([A.df.Close for A in assets], axis=1)
        blah = blah.index
        signal = np.ones([m.shape[0], len(assets)])
        signal = pd.DataFrame(signal, blah, [A.ticker for A in assets])

        return signal
