### `backtest.py` — Quick Profitability Check
```python
# backtest.py — Run this to see expected monthly return
import asyncio
from exchange import AsyncExchange
from strategy import AlphaStrategy

async def run():
    ex = AsyncExchange()
    df = await ex.fetch_ohlcv('1m', limit=4000)
    strat = AlphaStrategy()
    signals = []
    for i in range(300, len(df)-1):
        sig = await strat.signal(df.iloc[i-300:i])
        signals.append(sig)
    ret = df['close'].pct_change().iloc[301:] * signals
    total = (1 + ret).prod() - 1
    sharpe = ret.mean()/ret.std() * (365*24*60)**0.5
    print(f"30-Day Backtest: {total:.1%} | Sharpe {sharpe:.2f}")

if __name__ == "__main__":
    asyncio.run(run())
