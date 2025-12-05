# main.py — Breadbakerv1 v4.2 — Alpha Arena Profitable Bot (Improved)
import asyncio
import threading
import yaml
import sys
from datetime import datetime
from pathlib import Path

# Import modules with error handling
try:
    from exchange import AsyncExchange
    from strategy import AlphaStrategy
    from risk_manager import RiskManager
    from utils import atr
    from evolution_pro import evolve_forever
except ImportError as e:
    print(f"CRITICAL: Missing module - {e}")
    print("Ensure all files exist: exchange.py, strategy.py, risk_manager.py, utils.py, evolution_pro.py")
    sys.exit(1)

# Load config with validation
try:
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("CRITICAL: config.yaml not found")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Validate required config keys
    required_keys = ['symbol', 'paper_trading', 'trading']
    required_trading_keys = ['atr_multiplier_sl', 'cooldown_after_loss_seconds', 'max_position_size_pct']
    
    for key in required_keys:
        if key not in cfg:
            print(f"CRITICAL: Missing config key: {key}")
            sys.exit(1)
    
    for key in required_trading_keys:
        if key not in cfg['trading']:
            print(f"CRITICAL: Missing trading config key: {key}")
            sys.exit(1)
            
except Exception as e:
    print(f"CRITICAL: Config error - {e}")
    sys.exit(1)

# Initialize global instances
exchange = AsyncExchange()
strategy = AlphaStrategy()
position = None
last_trade_time = None
trade_count = 0

# State persistence
def save_state():
    """Save current position state to file for crash recovery"""
    try:
        state = {
            'position': position,
            'last_trade_time': last_trade_time.isoformat() if last_trade_time else None,
            'trade_count': trade_count
        }
        with open('bot_state.yaml', 'w') as f:
            yaml.dump(state, f)
    except Exception as e:
        print(f"Warning: Failed to save state - {e}")

def load_state():
    """Load saved state on startup"""
    global position, last_trade_time, trade_count
    try:
        if Path('bot_state.yaml').exists():
            with open('bot_state.yaml') as f:
                state = yaml.safe_load(f)
                position = state.get('position')
                if state.get('last_trade_time'):
                    last_trade_time = datetime.fromisoformat(state['last_trade_time'])
                trade_count = state.get('trade_count', 0)
                print(f"State restored: Position={bool(position)}, Trades={trade_count}")
    except Exception as e:
        print(f"Warning: Could not load state - {e}")

async def trading_loop():
    """Main trading logic with improved error handling and safety checks"""
    global position, last_trade_time, trade_count
    
    try:
        # Fetch market data
        df = await exchange.fetch_ohlcv('1m', 2000)
        if df is None or len(df) < 200:
            print("Warning: Insufficient data, skipping cycle")
            return
        
        # Generate signal
        signal = await strategy.signal(df)
        if signal not in [-1, 0, 1]:
            print(f"Warning: Invalid signal {signal}, defaulting to 0")
            signal = 0
        
        # Get current market state
        price = await exchange.price()
        bal = await exchange.balance()
        
        if price is None or bal is None or bal <= 0:
            print("Warning: Invalid price or balance, skipping cycle")
            return
        
        # Calculate ATR and risk parameters
        atr_val = atr(df).iloc[-1]
        if atr_val <= 0 or pd.isna(atr_val):
            print(f"Warning: Invalid ATR {atr_val}, skipping cycle")
            return
        
        rm = RiskManager(bal, price, atr_val)
        sl = price - (atr_val * cfg['trading']['atr_multiplier_sl'])
        size = rm.position_size(sl)
        
        # Validate position size
        max_size = bal * cfg['trading']['max_position_size_pct']
        if size > max_size:
            print(f"Warning: Position size {size:.4f} exceeds max {max_size:.4f}, capping")
            size = max_size
        
        if size <= 0:
            print("Warning: Position size too small, skipping entry")
            return
        
        levels = rm.levels(price, sl)
        
        # ENTRY LOGIC
        if signal == 1 and not position:
            # Check cooldown after loss
            if last_trade_time:
                cooldown = cfg['trading']['cooldown_after_loss_seconds']
                time_since_trade = (datetime.now() - last_trade_time).total_seconds()
                if time_since_trade < cooldown:
                    remaining = cooldown - time_since_trade
                    print(f"Cooldown active: {remaining:.0f}s remaining")
                    return
            
            # Execute entry
            order = await exchange.market_order('buy', size)
            if order and order.get('status') == 'filled':
                position = {
                    'entry': price,
                    'size': size,
                    'sl': sl,
                    'tp1': levels['tp1'],
                    'tp2': levels['tp2'],
                    'tp1_hit': False,
                    'entry_time': datetime.now().isoformat()
                }
                trade_count += 1
                save_state()
                print(f"[{trade_count}] ENTRY LONG @ {price:.2f} | Size: {size:.4f} | SL: {sl:.2f} | TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f}")
            else:
                print(f"Warning: Entry order failed - {order}")
        
        # EXIT & MANAGEMENT LOGIC
        elif position:
            cur_price = await exchange.price()
            if cur_price is None:
                print("Warning: Could not fetch current price for exit logic")
                return
            
            pnl_pct = ((cur_price - position['entry']) / position['entry']) * 100
            
            # TP1 (50% exit at 0.75R)
            if not position['tp1_hit'] and cur_price >= position['tp1']:
                exit_size = position['size'] * 0.5
                order = await exchange.market_order('sell', exit_size)
                if order and order.get('status') == 'filled':
                    position['size'] -= exit_size
                    position['tp1_hit'] = True
                    position['sl'] = position['entry']  # Move SL to breakeven
                    save_state()
                    print(f"✓ TP1 HIT @ {cur_price:.2f} (+{pnl_pct:.2f}%) | 50% closed | SL → BE")
                else:
                    print(f"Warning: TP1 exit order failed - {order}")
            
            # TP2 (full exit at 1.5R)
            if position['tp1_hit'] and cur_price >= position['tp2']:
                order = await exchange.market_order('sell', position['size'])
                if order and order.get('status') == 'filled':
                    print(f"✓✓ TP2 HIT @ {cur_price:.2f} (+{pnl_pct:.2f}%) | FULL EXIT (+1.5R)")
                    last_trade_time = datetime.now()
                    position = None
                    save_state()
                else:
                    print(f"Warning: TP2 exit order failed - {order}")
            
            # Stop Loss
            elif cur_price <= position['sl']:
                order = await exchange.market_order('sell', position['size'])
                if order and order.get('status') == 'filled':
                    print(f"✗ STOP-LOSS @ {cur_price:.2f} ({pnl_pct:.2f}%) | Cooldown: {cfg['trading']['cooldown_after_loss_seconds']}s")
                    last_trade_time = datetime.now()
                    position = None
                    save_state()
                else:
                    print(f"Warning: SL exit order failed - {order}")
            
            # Log position status every 5 minutes
            elif trade_count % 10 == 0:
                print(f"Position open: Entry={position['entry']:.2f} | Current={cur_price:.2f} | PnL={pnl_pct:.2f}% | TP1_hit={position['tp1_hit']}")
        
    except asyncio.CancelledError:
        print("Trading loop cancelled")
        raise
    except Exception as e:
        print(f"ERROR in trading loop: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Don't crash on errors, continue next cycle
    finally:
        await asyncio.sleep(30)

async def health_check():
    """Periodic health check to ensure bot is functioning"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Check exchange connectivity
            price = await exchange.price()
            if price is None:
                print("⚠ HEALTH: Exchange connection issue")
            
            # Check if stuck in position too long (e.g., > 24 hours)
            if position and 'entry_time' in position:
                entry_time = datetime.fromisoformat(position['entry_time'])
                hours_in_position = (datetime.now() - entry_time).total_seconds() / 3600
                if hours_in_position > 24:
                    print(f"⚠ HEALTH: Position open for {hours_in_position:.1f} hours")
            
            print(f"♥ HEALTH: OK | Trades: {trade_count} | Position: {bool(position)}")
            
        except Exception as e:
            print(f"Health check error: {e}")

async def main():
    """Main entry point with proper startup/shutdown handling"""
    print("="*60)
    print(f"Breadbakerv1 v4.2 STARTING")
    print(f"Symbol: {cfg['symbol']} | Paper: {cfg['paper_trading']}")
    print(f"ATR SL Multiplier: {cfg['trading']['atr_multiplier_sl']}")
    print(f"Max Position Size: {cfg['trading']['max_position_size_pct']*100}%")
    print("="*60)
    
    # Load saved state
    load_state()
    
    # Start evolution engine in background thread
    try:
        evolution_thread = threading.Thread(
            target=lambda: asyncio.run(evolve_forever()),
            daemon=True,
            name="EvolutionEngine"
        )
        evolution_thread.start()
        print("✓ Evolution engine started")
    except Exception as e:
        print(f"Warning: Evolution engine failed to start - {e}")
    
    # Start health check task
    health_task = asyncio.create_task(health_check())
    
    # Main trading loop
    try:
        while True:
            await trading_loop()
    except KeyboardInterrupt:
        print("\n⚠ Shutting down gracefully...")
        save_state()
        health_task.cancel()
        print("✓ State saved. Goodbye!")
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        import traceback
        traceback.print_exc()
        save_state()
        raise

if __name__ == "__main__":
    # Add pandas import for ATR check
    import pandas as pd
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Bot crashed: {e}")
        sys.exit(1)
