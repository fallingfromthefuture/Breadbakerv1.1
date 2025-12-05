# utils.py — Data & ATR helpers
import pandas as pd
import numpy as np

def resample_df(df, timeframe):
    rule = {'1m':'1T','3m':'3T','5m':'5T','15m':'15T','1h':'1H'}.get(timeframe, '1T')
    return df.resample(rule).agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
