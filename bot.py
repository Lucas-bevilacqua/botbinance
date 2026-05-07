"""
🤖 Binance Day Trading Bot
Estratégia: RSI + EMA crossover com seleção automática de par
Risco: Agressivo (10-15% por trade)
"""

import os
import time
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

RISK_PER_TRADE    = 0.12        # 12% do saldo por operação
STOP_LOSS_PCT     = 0.04        # Stop loss 4%
TAKE_PROFIT_PCT   = 0.08        # Take profit 8% (2:1 reward/risk)
SCAN_INTERVAL_SEC = 60          # Verifica mercado a cada 60 segundos
MIN_VOLUME_USDT   = 5_000_000   # Volume mínimo 24h em USDT para considerar par
MAX_OPEN_TRADES   = 1           # Só 1 trade aberto por vez (conta pequena)

# Pares candidatos (stablecoins USDT, líquidos)
CANDIDATE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "MATICUSDT", "LINKUSDT"
]

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── INDICADORES ──────────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def get_signals(df: pd.DataFrame) -> dict:
    """Retorna sinais de compra/venda baseados em RSI + EMA crossover."""
    close = df["close"]

    rsi     = calc_rsi(close, 14)
    ema9    = calc_ema(close, 9)
    ema21   = calc_ema(close, 21)

    last_rsi  = rsi.iloc[-1]
    prev_rsi  = rsi.iloc[-2]
    last_ema9 = ema9.iloc[-1]
    last_ema21= ema21.iloc[-1]
    prev_ema9 = ema9.iloc[-2]
    prev_ema21= ema21.iloc[-2]

    # EMA crossover bullish: EMA9 cruzou acima da EMA21
    bull_cross = (prev_ema9 <= prev_ema21) and (last_ema9 > last_ema21)
    # EMA crossover bearish: EMA9 cruzou abaixo da EMA21
    bear_cross = (prev_ema9 >= prev_ema21) and (last_ema9 < last_ema21)

    buy  = bull_cross and last_rsi < 65 and last_rsi > 40
    sell = bear_cross or last_rsi > 75

    return {
        "buy": buy,
        "sell": sell,
        "rsi": round(last_rsi, 2),
        "ema9": round(last_ema9, 6),
        "ema21": round(last_ema21, 6),
    }


# ─── BINANCE HELPERS ──────────────────────────────────────────────────────────

def get_klines(client: Client, symbol: str, interval="15m", limit=100) -> pd.DataFrame:
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=[
        "time","open","high","low","close","volume",
        "close_time","qa_vol","trades","taker_buy","taker_buy_qa","ignore"
    ])
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def get_usdt_balance(client: Client) -> float:
    info = client.get_asset_balance(asset="USDT")
    return float(info["free"]) if info else 0.0


def get_asset_balance(client: Client, asset: str) -> float:
    info = client.get_asset_balance(asset=asset)
    return float(info["free"]) if info else 0.0


def get_symbol_price(client: Client, symbol: str) -> float:
    return float(client.get_symbol_ticker(symbol=symbol)["price"])


def get_lot_size(client: Client, symbol: str) -> dict:
    info = client.get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            return {
                "min_qty":  float(f["minQty"]),
                "step_size": float(f["stepSize"]),
            }
    return {"min_qty": 0.0001, "step_size": 0.0001}


def round_step(qty: float, step: float) -> float:
    precision = len(str(step).rstrip("0").split(".")[-1])
    return round(int(qty / step) * step, precision)


def pick_best_pair(client: Client) -> str | None:
    """Escolhe o par com maior momentum e volume."""
    scores = {}
    tickers = {t["symbol"]: t for t in client.get_ticker()}

    for symbol in CANDIDATE_PAIRS:
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        volume = float(ticker["quoteVolume"])
        change = float(ticker["priceChangePercent"])
        if volume < MIN_VOLUME_USDT:
            continue
        # Score: volume normalizado * momentum positivo
        scores[symbol] = volume * max(change, 0)

    if not scores:
        return None
    best = max(scores, key=scores.get)
    log.info(f"🔍 Melhor par selecionado: {best} (score={scores[best]:.0f})")
    return best


# ─── TRADE MANAGER ────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self):
        self.client      = Client(API_KEY, API_SECRET)
        self.open_trade  = None   # {"symbol", "qty", "entry_price", "stop", "target"}
        log.info("✅ Bot conectado à Binance")

    def run(self):
        log.info("🚀 Bot iniciado — estratégia: RSI + EMA Crossover (agressivo)")
        while True:
            try:
                self._cycle()
            except BinanceAPIException as e:
                log.error(f"Binance API error: {e}")
            except Exception as e:
                log.error(f"Erro inesperado: {e}")
            time.sleep(SCAN_INTERVAL_SEC)

    def _cycle(self):
        usdt = get_usdt_balance(self.client)
        log.info(f"💰 Saldo USDT: ${usdt:.2f}")

        # ── Gerencia trade aberto ───────────────────────────────────────────
        if self.open_trade:
            t     = self.open_trade
            price = get_symbol_price(self.client, t["symbol"])
            log.info(f"📊 {t['symbol']} preço atual: {price:.6f} | entrada: {t['entry_price']:.6f} | stop: {t['stop']:.6f} | target: {t['target']:.6f}")

            hit_stop   = price <= t["stop"]
            hit_target = price >= t["target"]

            df      = get_klines(self.client, t["symbol"])
            signals = get_signals(df)

            if hit_stop or hit_target or signals["sell"]:
                reason = "STOP" if hit_stop else ("TARGET" if hit_target else "SINAL VENDA")
                self._close_trade(reason, price)
            return

        # ── Busca nova oportunidade ─────────────────────────────────────────
        if usdt < 1.0:
            log.warning("⚠️  Saldo USDT insuficiente para operar.")
            return

        symbol = pick_best_pair(self.client)
        if not symbol:
            log.info("😴 Nenhum par com volume suficiente agora.")
            return

        df      = get_klines(self.client, symbol)
        signals = get_signals(df)
        log.info(f"📈 {symbol} → RSI={signals['rsi']} | EMA9={signals['ema9']} | EMA21={signals['ema21']} | BUY={signals['buy']}")

        if signals["buy"]:
            self._open_trade(symbol, usdt, signals)

    def _open_trade(self, symbol: str, usdt: float, signals: dict):
        capital = usdt * RISK_PER_TRADE
        price   = get_symbol_price(self.client, symbol)
        lot     = get_lot_size(self.client, symbol)
        qty     = round_step(capital / price, lot["step_size"])

        if qty < lot["min_qty"]:
            log.warning(f"⚠️  Quantidade {qty} abaixo do mínimo {lot['min_qty']} para {symbol}")
            return

        try:
            order = self.client.order_market_buy(symbol=symbol, quantity=qty)
            entry = float(order["fills"][0]["price"]) if order["fills"] else price

            self.open_trade = {
                "symbol":      symbol,
                "qty":         qty,
                "entry_price": entry,
                "stop":        round(entry * (1 - STOP_LOSS_PCT), 8),
                "target":      round(entry * (1 + TAKE_PROFIT_PCT), 8),
                "order_id":    order["orderId"],
            }
            log.info(f"🟢 COMPRA {symbol} | qty={qty} | entrada={entry:.6f} | stop={self.open_trade['stop']:.6f} | target={self.open_trade['target']:.6f}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir trade: {e}")

    def _close_trade(self, reason: str, price: float):
        t   = self.open_trade
        qty = t["qty"]

        # Verifica saldo real do ativo antes de vender
        asset = t["symbol"].replace("USDT", "")
        balance = get_asset_balance(self.client, asset)
        lot   = get_lot_size(self.client, t["symbol"])
        qty   = round_step(min(qty, balance), lot["step_size"])

        try:
            self.client.order_market_sell(symbol=t["symbol"], quantity=qty)
            pnl = (price - t["entry_price"]) / t["entry_price"] * 100
            log.info(f"🔴 VENDA {t['symbol']} [{reason}] | saída={price:.6f} | PnL={pnl:+.2f}%")
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar trade: {e}")
        finally:
            self.open_trade = None


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("❌ Configure BINANCE_API_KEY e BINANCE_API_SECRET como variáveis de ambiente.")
        exit(1)
    TradingBot().run()
