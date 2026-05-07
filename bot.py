"""
🤖 Binance Futures Scalping Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estratégia: SuperTrend (ATR 7, mult 2) + RSI(9) + EMA200
Metodologia: Trend-following com filtro de momentum
Backtests documentados: 55-70% win rate, profit factor ~1.8
Timeframe: 3m (scalping) | Alavancagem: 10x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGRAS DE ENTRADA:
  LONG:  SuperTrend verde (flip) + preço > EMA200 + RSI entre 40-60
  SHORT: SuperTrend vermelho (flip) + preço < EMA200 + RSI entre 40-60

REGRAS DE SAÍDA:
  - Stop loss: linha do SuperTrend (dinâmico)
  - Take profit: 2x o risco (ratio 2:1)
  - Timeout: 10 minutos sem atingir alvo
  - Sinal contrário: SuperTrend flipa na direção oposta
"""

import os
import time
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

SUPERTREND_ATR_PERIOD  = 7
SUPERTREND_MULTIPLIER  = 2.0
RSI_PERIOD             = 9
RSI_MIN                = 40
RSI_MAX                = 60
EMA_TREND_PERIOD       = 200
LEVERAGE               = 10
RISK_PER_TRADE         = 0.15
REWARD_RISK_RATIO      = 2.0
MAX_TRADE_DURATION     = 600
SCAN_INTERVAL_SEC      = 20

CANDIDATE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "MATICUSDT"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── INDICADORES ──────────────────────────────────────────────────────────────

def calc_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_supertrend(df, atr_period, multiplier):
    hl2  = (df["high"] + df["low"]) / 2
    atr  = calc_atr(df, atr_period)
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = [np.nan] * len(df)
    direction  = [0] * len(df)
    for i in range(1, len(df)):
        cp = df["close"].iloc[i-1]
        cc = df["close"].iloc[i]
        ub = upper_band.iloc[i] if (upper_band.iloc[i-1] < upper_band.iloc[i] or cp > upper_band.iloc[i-1]) else min(upper_band.iloc[i], upper_band.iloc[i-1])
        lb = lower_band.iloc[i] if (lower_band.iloc[i-1] > lower_band.iloc[i] or cp < lower_band.iloc[i-1]) else max(lower_band.iloc[i], lower_band.iloc[i-1])
        prev_st = supertrend[i-1] if not np.isnan(supertrend[i-1]) else ub
        if prev_st == upper_band.iloc[i-1]:
            if cc <= ub:
                supertrend[i] = ub; direction[i] = -1
            else:
                supertrend[i] = lb; direction[i] = 1
        else:
            if cc >= lb:
                supertrend[i] = lb; direction[i] = 1
            else:
                supertrend[i] = ub; direction[i] = -1
    df = df.copy()
    df["supertrend"] = supertrend
    df["st_dir"] = direction
    return df

def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_signals(df):
    df_st   = calc_supertrend(df, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
    close   = df["close"]
    rsi     = calc_rsi(close, RSI_PERIOD)
    ema200  = calc_ema(close, EMA_TREND_PERIOD)
    curr_dir  = df_st["st_dir"].iloc[-1]
    prev_dir  = df_st["st_dir"].iloc[-2]
    curr_st   = df_st["supertrend"].iloc[-1]
    last_rsi  = rsi.iloc[-1]
    last_ema  = ema200.iloc[-1]
    last_close= close.iloc[-1]
    bull_flip = (prev_dir == -1) and (curr_dir == 1)
    bear_flip = (prev_dir == 1)  and (curr_dir == -1)
    above_ema = last_close > last_ema
    below_ema = last_close < last_ema
    rsi_ok    = RSI_MIN <= last_rsi <= RSI_MAX
    return {
        "buy":        bull_flip and above_ema and rsi_ok,
        "sell":       bear_flip and below_ema and rsi_ok,
        "exit_long":  curr_dir == -1,
        "exit_short": curr_dir == 1,
        "rsi":        round(last_rsi, 2),
        "st_dir":     curr_dir,
        "st_line":    round(curr_st, 6) if not np.isnan(curr_st) else 0,
        "ema200":     round(last_ema, 6),
        "close":      round(last_close, 6),
    }

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_klines(client, symbol, interval="3m", limit=250):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume","close_time","qa_vol","trades","taker_buy","taker_buy_qa","ignore"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

def get_futures_balance(client):
    for b in client.futures_account_balance():
        if b["asset"] == "USDT":
            return float(b["availableBalance"])
    return 0.0

def get_futures_price(client, symbol):
    return float(client.futures_symbol_ticker(symbol=symbol)["price"])

def get_lot_size(client, symbol):
    for s in client.futures_exchange_info()["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return {"min_qty": float(f["minQty"]), "step_size": float(f["stepSize"])}
    return {"min_qty": 0.001, "step_size": 0.001}

def round_step(qty, step):
    if step == 0: return qty
    decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(int(qty / step) * step, decimals)

def set_leverage(client, symbol, leverage):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
    except BinanceAPIException:
        pass

def rank_pairs(client):
    """Retorna todos os pares ordenados por volume x momentum."""
    tickers = {t["symbol"]: t for t in client.futures_ticker()}
    scores  = {}
    for symbol in CANDIDATE_PAIRS:
        t = tickers.get(symbol)
        if not t: continue
        volume = float(t["quoteVolume"])
        change = abs(float(t["priceChangePercent"]))
        if volume < 10_000_000: continue
        scores[symbol] = volume * change
    return sorted(scores, key=scores.get, reverse=True)

# ─── BOT ──────────────────────────────────────────────────────────────────────

class SuperTrendScalpBot:
    def __init__(self):
        self.client     = Client(API_KEY, API_SECRET)
        self.open_trade = None
        log.info("✅ Bot conectado à Binance Futures")
        log.info(f"📐 SuperTrend(ATR={SUPERTREND_ATR_PERIOD}, mult={SUPERTREND_MULTIPLIER}) + RSI({RSI_PERIOD}) + EMA{EMA_TREND_PERIOD}")
        log.info(f"⚙️  Alavancagem: {LEVERAGE}x | Risco: {RISK_PER_TRADE*100:.0f}%/trade | R:R=1:{REWARD_RISK_RATIO}")

    def run(self):
        log.info("🚀 SuperTrend Scalp Bot iniciado")
        while True:
            try:
                self._cycle()
            except BinanceAPIException as e:
                log.error(f"Binance API error: {e}"); time.sleep(10)
            except Exception as e:
                log.error(f"Erro: {e}", exc_info=True); time.sleep(10)
            time.sleep(SCAN_INTERVAL_SEC)

    def _cycle(self):
        balance = get_futures_balance(self.client)
        log.info(f"💰 Saldo: ${balance:.2f} USDT")

        if self.open_trade:
            t       = self.open_trade
            price   = get_futures_price(self.client, t["symbol"])
            elapsed = time.time() - t["opened_at"]
            side    = t["side"]
            pnl     = ((price - t["entry"]) / t["entry"] * 100) * (1 if side == "LONG" else -1) * LEVERAGE
            log.info(f"📊 {side} {t['symbol']} | {price:.4f} | PnL={pnl:+.2f}% | {elapsed:.0f}s")

            hit_stop   = price <= t["stop"]   if side == "LONG" else price >= t["stop"]
            hit_target = price >= t["target"] if side == "LONG" else price <= t["target"]
            timeout    = elapsed >= MAX_TRADE_DURATION

            df      = get_klines(self.client, t["symbol"])
            signals = get_signals(df)
            st_exit = signals["exit_long"] if side == "LONG" else signals["exit_short"]

            if hit_stop or hit_target or timeout or st_exit:
                reason = "STOP" if hit_stop else ("TARGET ✨" if hit_target else ("TIMEOUT" if timeout else "ST_FLIP"))
                self._close_trade(reason, price)
            return

        if balance < 1.0:
            log.warning("⚠️  Saldo insuficiente."); return

        pairs = rank_pairs(self.client)
        if not pairs:
            log.info("😴 Nenhum par com volume suficiente."); return

        log.info(f"🔎 Varrendo {len(pairs)} pares: {', '.join(pairs)}")
        found = False
        for symbol in pairs:
            set_leverage(self.client, symbol, LEVERAGE)
            df      = get_klines(self.client, symbol)
            signals = get_signals(df)
            log.info(f"  {symbol} | close={signals['close']} | ST={'🟢' if signals['st_dir']==1 else '🔴'} | RSI={signals['rsi']} | BUY={signals['buy']} SELL={signals['sell']}")
            if signals["buy"]:
                self._open_trade(symbol, balance, "LONG", signals)
                found = True; break
            elif signals["sell"]:
                self._open_trade(symbol, balance, "SHORT", signals)
                found = True; break
        if not found:
            log.info("😴 Nenhum sinal em nenhum par. Aguardando...")

    def _open_trade(self, symbol, balance, side, signals):
        price     = get_futures_price(self.client, symbol)
        lot       = get_lot_size(self.client, symbol)
        st_line   = signals["st_line"]
        stop_dist = max(abs(price - st_line) / price, 0.005)
        stop      = price * (1 - stop_dist) if side == "LONG" else price * (1 + stop_dist)
        target    = price * (1 + stop_dist * REWARD_RISK_RATIO) if side == "LONG" else price * (1 - stop_dist * REWARD_RISK_RATIO)
        capital   = balance * RISK_PER_TRADE
        qty       = round_step((capital * LEVERAGE) / price, lot["step_size"])
        if qty < lot["min_qty"]:
            log.warning(f"⚠️  Qty {qty} abaixo do mínimo {lot['min_qty']}"); return
        try:
            order = self.client.futures_create_order(symbol=symbol, side="BUY" if side=="LONG" else "SELL", type="MARKET", quantity=qty)
            entry = float(order["avgPrice"]) if float(order.get("avgPrice",0)) > 0 else price
            self.open_trade = {"symbol": symbol, "side": side, "qty": qty, "entry": entry, "stop": round(stop,8), "target": round(target,8), "order_id": order["orderId"], "opened_at": time.time()}
            log.info(f"{'🟢 LONG' if side=='LONG' else '🔴 SHORT'} {symbol} | qty={qty} | entrada={entry:.4f} | stop={stop:.4f} ({stop_dist*100:.2f}%) | target={target:.4f} ({stop_dist*REWARD_RISK_RATIO*100:.2f}%) | {capital:.2f}x{LEVERAGE}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir: {e}")

    def _close_trade(self, reason, price):
        t = self.open_trade
        try:
            self.client.futures_create_order(symbol=t["symbol"], side="SELL" if t["side"]=="LONG" else "BUY", type="MARKET", quantity=t["qty"], reduceOnly=True)
            pnl = ((price - t["entry"]) / t["entry"] * 100) * (1 if t["side"]=="LONG" else -1) * LEVERAGE
            log.info(f"⏹  FECHOU [{reason}] {t['symbol']} | saída={price:.4f} | PnL={pnl:+.2f}%")
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar: {e}")
        finally:
            self.open_trade = None

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("❌ Configure BINANCE_API_KEY e BINANCE_API_SECRET."); exit(1)
    SuperTrendScalpBot().run()
