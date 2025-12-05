# main.py — Breadbakerv1- v4.1 — Alpha Arena Profitable Bot
import asyncio
import threading
import yaml
from datetime import datetime
from exchange import AsyncExchange
from strategy import AlphaStrategy
from risk_manager import RiskManager
from utils import atr
from evolution_pro import evolve_forever

# Load config
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

exchange = AsyncExchange()
strategy = AlphaStrategy()
position = None

async def trading_loop():
    global position
    try:
        df = await exchange.fetch_ohlcv('1m', 2000)
        signal = await strategy.signal(df)
        price = await exchange.price()
        bal = await exchange.balance()
        atr_val = atr(df).iloc[-1]

        rm = RiskManager(bal, price, atr_val)
        sl = price - (atr_val * cfg['trading']['atr_multiplier_sl'])
        size = rm.position_size(sl)
        levels = rm.levels(price, sl)

        # ENTRY
        if signal == 1 and not position:
            await exchange.market_order('buy', size)
            position = {'entry': price, 'size': size, 'sl': sl, 'tp1': levels['tp1'], 'tp2': levels['tp2'], 'tp1_hit': False}
            print(f"ENTRY LONG @ {price:.2f}")

        # EXIT & MANAGEMENT
        if position:
            cur_price = await exchange.price()

            # TP1 (50%)
            if not position['tp1_hit'] and cur_price >= position['tp1']:
                await exchange.market_order('sell', position['size'] * 0.5)
                position['size'] *= 0.5
                position['tp1_hit'] = True
                print(f"TP1 HIT → 50% closed @ {cur_price:.2f}")

            # TP2 or SL
            if position['tp1_hit'] and cur_price >= position['tp2']:
                await exchange.market_order('sell', position['size'])
                print(f"FULL EXIT @ {cur_price:.2f} (+1.5R)")
                position = None
            elif cur_price <= position['sl']:
                await exchange.market_order('sell', position['size'])
                print(f"STOP-LOSS @ {cur_price:.2f}")
                position = None
                await asyncio.sleep(cfg['trading']['cooldown_after_loss_seconds'])

        await asyncio.sleep(30)

    except Exception as e:
        print(f"ERROR: {e}")
        await asyncio.sleep(60)

async def main():
    print(f"Breadbakerv1- v4.1 STARTED — {cfg['symbol']} | Paper: {cfg['paper_trading']}")
    threading.Thread(target=lambda: asyncio.run(evolve_forever()), daemon=True).start()
    while True:
        await trading_loop()

if __name__ == "__main__":
    asyncio.run(main())
