# exchange.py — Async Hyperliquid (2025 CCXT)
import ccxt.async_support as ccxt
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

class AsyncExchange:
    def __init__(self):
        self.ex = ccxt.hyperliquid({
            'secret': config['hyperliquid']['private_key'],
            'enableRateLimit': True
        })
        if config['paper_trading']:
            self.ex.set_sandbox_mode(True)
        self.symbol = config['symbol']

    async def price(self):
        t = await self.ex.fetch_ticker(self.symbol)
        return t['last']

    async def balance(self):
        b = await self.ex.fetch_balance()
        return b.get('USDC', 10000)

    async def market_order(self, side, amount):
        print(f"[TRADE] {side.upper()} {amount:.6f} {self.symbol}")
        return await self.ex.create_market_order(self.symbol, side, amount)

    async def fetch_ohlcv(self, timeframe='1m', limit=2000):
        raw = await self.ex.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df

    async def close(self):
        await self.ex.close()
