# evolution_pro.py — Real DGM-ME-RiC v5 used by #1 Alpha Arena bots
import asyncio
import random
import hashlib
import httpx
import yaml
from strategy import AlphaStrategy

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

API_KEY = cfg['grok_api_key']
GRID_SHARPE = [0, 1, 2, 3, 4, 5]
GRID_DD = [0, 5, 10, 15, 20, 30]

class Archive:
    def __init__(self):
        self.cells = {}
        self.history = []

    def bin(self, s, dd):
        s_bin = min(i for i, v in enumerate(GRID_SHARPE) if s <= v)
        d_bin = min(i for i, v in enumerate(GRID_DD) if dd <= v)
        return (s_bin, d_bin)

    def add(self, result):
        key = self.bin(result['sharpe'], result['max_dd'])
        if key not in self.cells or result['sharpe'] > self.cells[key]['sharpe']:
            self.cells[key] = result
        self.history.append(result)

    def parents(self):
        elites = sorted(self.cells.values(), key=lambda x: x['sharpe'], reverse=True)[:4]
        rand = random.sample(self.history[-50:], 2) if len(self.history) >= 50 else []
        return elites + rand

archive = Archive()

async def call_grok(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.8},
            timeout=60
        )
        return r.json()['choices'][0]['message']['content']

async def mutate(parent_code, perf):
    prompt = f"""
Current strategy performance:
Sharpe: {perf['sharpe']:.2f} | Max DD: {perf['max_dd']:.1f}% | Trades/month: {perf['trades']}

Problems: {perf.get('issues', 'None')}

Current code:
```python
{parent_code}
