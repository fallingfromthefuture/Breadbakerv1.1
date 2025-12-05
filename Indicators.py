# indicators.py — BOS + FVG (2025 SMC Standard)
def detect_bos(df):
    df = df.copy()
    df['bos_up'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) <= df['high'].shift(2))
    df['bos_down'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) >= df['low'].shift(2))
    return df

def detect_fvg(df):
    df = df.copy()
    df['fvg_up'] = (df['low'] > df['high'].shift(2))
    df['fvg_down'] = (df['high'] < df['low'].shift(2))
    return df
