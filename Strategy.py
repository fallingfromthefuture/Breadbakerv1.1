# strategy.py — Mystery Model + DeepSeek + Qwen Hybrid
from utils import resample_df, atr
from indicators import detect_bos, detect_fvg

class AlphaStrategy:
    def __init__(self):
        pass

    async def signal(self, df_1m):
        df_1h = resample_df(df_1m, '1h')
        df_15m = resample_df(df_1m, '15m')
        df_5m = resample_df(df_1m, '5m')
        df_3m = resample_df(df_1m, '3m')

        # 1h Bias (DeepSeek)
        df_1h = detect_bos(df_1h)
        bull_bias = df_1h['bos_up'].iloc[-1]

        # 5m FVG + Liquidity Grab
        df_5m = detect_fvg(df_5m)
        fvg_touch = df_5m['fvg_up'].iloc[-1]

        # 3m Reversal (BB + WR)
        bb_lower, _, _ = df_3m['close'].rolling(20).mean(), df_3m['close'].rolling(20).std()
        bb_lower = bb_lower - 2 * _
        wr = -100 * (df_3m['high'].rolling(14).max() - df_3m['close']) / (df_3m['high'].rolling(14).max() - df_3m['low'].rolling(14).min())
        reversal = df_3m['close'].iloc[-1] <= bb_lower.iloc[-1] and wr.iloc[-1] < -80

        # 1m Confirmation
        engulf = df_1m['close'].iloc[-1] > df_1m['open'].iloc[-1] and df_1m['close'].iloc[-2] < df_1m['open'].iloc[-2]

        return 1 if bull_bias and fvg_touch and reversal and engulf else 0
