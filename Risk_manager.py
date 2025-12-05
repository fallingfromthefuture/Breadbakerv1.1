# risk_manager.py — Alpha Arena RR Discipline
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['trading']

class RiskManager:
    def __init__(self, balance, price, atr_val):
        self.balance = balance
        self.price = price
        self.atr = atr_val
        self.risk_pct = cfg['risk_percent_per_trade'] / 100

    def position_size(self, sl_price):
        risk_amount = self.balance * self.risk_pct
        distance = abs(self.price - sl_price)
        return round(risk_amount / distance, 6) if distance > 0 else 0

    def levels(self, entry, sl):
        risk = abs(entry - sl)
        tp1 = entry + risk * cfg['rr_tp1']
        tp2 = entry + risk * cfg['rr_tp2']
        return {'sl': sl, 'tp1': tp1, 'tp2': tp2, 'risk': risk}
