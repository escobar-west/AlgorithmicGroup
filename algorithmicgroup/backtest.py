import numpy as np
import pandas as pd

def backtest_strat(principal, strat, assets, *args, **kwargs):
    if isinstance(assets, list):
        principal /= len(assets)
        df_close = pd.concat(
                       [np.log(A.df.Close.rename(A.name+'_Close')) for A in assets], axis=1)
    else:
        df_close = np.log(assets.df.Close.rename(assets.name+'_Close'))
        df_close = pd.DataFrame(df_close)

    df_signals = strat(assets, *args, **kwargs)
    if df_signals.ndim == 1:
        df_signals = pd.DataFrame(df_signals)

    # TO DO: Fix problem of numpy array operations returning nan    
    port_gains = np.zeros(df_close.shape)
    port_gains[0,:] = np.log(principal)
    port_gains[1:,:] = df_close.diff().values[1:,:] * df_signals.values[:-1,:]
    port_gains = np.exp(port_gains.cumsum(axis=0))

    if port_gains.ndim == 1:
        final = port_gains
    else:
        final = port_gains.sum(axis=1)

    res = (np.exp(df_close)
              .join(df_signals)
              .join(pd.DataFrame(port_gains, index=df_close.index))
              .join(pd.Series(final, name='final', index=df_close.index)))

    return res
