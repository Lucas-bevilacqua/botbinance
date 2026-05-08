"""
Binance Futures Scalping Bot
Estrategia: EMA Cross (9/21) + RSI(14) + EMA200
LONG:  EMA9 cruza acima EMA21 + preco > EMA200 + RSI 45-65
SHORT: EMA9 cruza abaixo EMA21 + preco < EMA200 + RSI 35-55
Stop: 1.5x ATR | Target: 2x risco | Timeout: 10min
"""

import os
import time
import logging
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np

# CONFIG
API_KEY        = os.getenv("BINANCE_API_KEY", "")
API_SECRET     = os.getenv("BINANCE_API_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMA_FAST         = 9
EMA_SLOW         = 21
EMA_TREND        = 200
RSI_PERIOD       = 14
ATR_PERIOD       = 14
ATR_STOP_MULT    = 1.5
LEVERAGE         = 10
RISK_PER_TRADE   = 0.05
REWARD_RISK      = 2.0
MAX_DURATION     = 600
SCAN_INTERVAL    = 20

CANDIDATE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# INDICADORES

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def get_signals(df):
    close    = df["close"]
    ema_fast = calc_ema(close, EMA_FAST)
    ema_slow = calc_ema(close, EMA_SLOW)
    ema_trend= calc_ema(close, EMA_TREND)
    rsi      = calc_rsi(close, RSI_PERIOD)
    atr      = calc_atr(df, ATR_PERIOD)

    # Cruzamento atual e anterior
    cross_up_now  = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    cross_up_prev = ema_fast.iloc[-2] > ema_slow.iloc[-2]
    cross_dn_now  = ema_fast.iloc[-1] < ema_slow.iloc[-1]
    cross_dn_prev = ema_fast.iloc[-2] < ema_slow.iloc[-2]

    # Sinal de entrada: momento do cruzamento
    bull_cross = cross_up_now and not cross_up_prev
    bear_cross = cross_dn_now and not cross_dn_prev

    last_close = close.iloc[-1]
    last_ema_t = ema_trend.iloc[-1]
    last_rsi   = rsi.iloc[-1]
    last_atr   = atr.iloc[-1]

    above_trend = last_close > last_ema_t
    below_trend = last_close < last_ema_t

    rsi_long  = 45 <= last_rsi <= 65
    rsi_short = 35 <= last_rsi <= 55

    return {
        "buy":        bull_cross and above_trend and rsi_long,
        "sell":       bear_cross and below_trend and rsi_short,
        "exit_long":  cross_dn_now,
        "exit_short": cross_up_now,
        "rsi":        round(last_rsi, 2),
        "ema_fast":   round(ema_fast.iloc[-1], 6),
        "ema_slow":   round(ema_slow.iloc[-1], 6),
        "ema_trend":  round(last_ema_t, 6),
        "atr":        round(last_atr, 6),
        "close":      round(last_close, 6),
    }

# HELPERS

def get_klines(client, symbol, interval="3m", limit=250):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume",
                                      "close_time","qa_vol","trades","taker_buy","taker_buy_qa","ignore"])
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

def ask_gemini(symbol, side, signals, candles):
    if not GEMINI_API_KEY:
        return True
    recent = candles[-5:]
    candle_summary = " | ".join([f"c={float(c[4]):.4f} v={float(c[5]):.0f}" for c in recent])
    prompt = f"""Voce e um analista de futuros de criptomoedas.
Par: {symbol} | Direcao: {side}
EMA9={signals['ema_fast']} EMA21={signals['ema_slow']} EMA200={signals['ema_trend']}
RSI(14): {signals['rsi']} | ATR: {signals['atr']} | Preco: {signals['close']}
Ultimos 5 candles (close | volume): {candle_summary}

Devo entrar em {side} agora?
Responda APENAS: CONFIRMAR ou BLOQUEAR: [motivo curto]"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 60}
        }, timeout=8)
        data = resp.json()
        if "candidates" not in data:
            log.warning(f"Gemini erro: {data.get('error', {}).get('message', str(data))}. Prosseguindo sem filtro.")
            return True
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        log.info(f"Gemini [{symbol}]: {text}")
        return text.upper().startswith("CONFIRMAR")
    except Exception as e:
        log.warning(f"Gemini indisponivel: {e}. Prosseguindo sem filtro.")
        return True

def rank_pairs(client):
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

# BOT

class EMAScalpBot:
    def __init__(self):
        self.client     = Client(API_KEY, API_SECRET)
        self.open_trade = None
        log.info("Bot conectado a Binance Futures")
        log.info(f"Estrategia: EMA{EMA_FAST}/{EMA_SLOW} Cross + RSI({RSI_PERIOD}) + EMA{EMA_TREND}")
        log.info(f"Alavancagem: {LEVERAGE}x | Risco: {RISK_PER_TRADE*100:.0f}%/trade | R:R=1:{REWARD_RISK}")
        self._sync_open_positions()

    def _sync_open_positions(self):
        try:
            positions = self.client.futures_position_information()
            for p in positions:
                amt = float(p["positionAmt"])
                if amt == 0: continue
                symbol = p["symbol"]
                side   = "LONG" if amt > 0 else "SHORT"
                entry  = float(p["entryPrice"])
                qty    = abs(amt)
                stop_dist = 0.008
                stop   = entry * (1 - stop_dist) if side == "LONG" else entry * (1 + stop_dist)
                target = entry * (1 + stop_dist * REWARD_RISK) if side == "LONG" else entry * (1 - stop_dist * REWARD_RISK)
                self.open_trade = {
                    "symbol": symbol, "side": side, "qty": qty,
                    "entry": entry, "stop": round(stop, 8), "target": round(target, 8),
                    "order_id": None, "opened_at": time.time(),
                }
                log.info(f"Posicao existente: {side} {symbol} | qty={qty} | entrada={entry:.4f}")
                break
            if not self.open_trade:
                log.info("Nenhuma posicao aberta. Pronto para operar.")
        except BinanceAPIException as e:
            log.error(f"Erro ao sincronizar: {e}")

    def run(self):
        log.info("EMA Scalp Bot iniciado")
        while True:
            try:
                self._cycle()
            except BinanceAPIException as e:
                log.error(f"Binance API error: {e}"); time.sleep(10)
            except Exception as e:
                log.error(f"Erro: {e}", exc_info=True); time.sleep(10)
            time.sleep(SCAN_INTERVAL)

    def _cycle(self):
        balance = get_futures_balance(self.client)
        log.info(f"Saldo: ${balance:.2f} USDT")

        if self.open_trade:
            t       = self.open_trade
            price   = get_futures_price(self.client, t["symbol"])
            elapsed = time.time() - t["opened_at"]
            side    = t["side"]
            pnl     = ((price - t["entry"]) / t["entry"] * 100) * (1 if side == "LONG" else -1) * LEVERAGE
            log.info(f"{side} {t['symbol']} | {price:.4f} | PnL={pnl:+.2f}% | {elapsed:.0f}s")

            hit_stop   = price <= t["stop"]   if side == "LONG" else price >= t["stop"]
            hit_target = price >= t["target"] if side == "LONG" else price <= t["target"]
            timeout    = elapsed >= MAX_DURATION

            df      = get_klines(self.client, t["symbol"])
            signals = get_signals(df)
            st_exit = signals["exit_long"] if side == "LONG" else signals["exit_short"]

            if hit_stop or hit_target or timeout or st_exit:
                reason = "STOP" if hit_stop else ("TARGET" if hit_target else ("TIMEOUT" if timeout else "EMA_CROSS"))
                self._close_trade(reason, price)
            return

        if balance < 1.0:
            log.warning("Saldo insuficiente."); return

        pairs = rank_pairs(self.client)
        if not pairs:
            log.info("Nenhum par com volume suficiente."); return

        log.info(f"Varrendo {len(pairs)} pares: {', '.join(pairs)}")
        found = False
        for symbol in pairs:
            set_leverage(self.client, symbol, LEVERAGE)
            df      = get_klines(self.client, symbol)
            signals = get_signals(df)
            log.info(f"  {symbol} | close={signals['close']} | EMA9={signals['ema_fast']} EMA21={signals['ema_slow']} | RSI={signals['rsi']} | BUY={signals['buy']} SELL={signals['sell']}")
            if signals["buy"]:
                self._open_trade(symbol, balance, "LONG", signals)
                found = True; break
            elif signals["sell"]:
                self._open_trade(symbol, balance, "SHORT", signals)
                found = True; break
        if not found:
            log.info("Nenhum sinal. Aguardando...")

    def _open_trade(self, symbol, balance, side, signals):
        price    = get_futures_price(self.client, symbol)
        lot      = get_lot_size(self.client, symbol)
        atr      = signals["atr"]
        stop_dist = max(atr * ATR_STOP_MULT / price, 0.005)
        stop     = price * (1 - stop_dist) if side == "LONG" else price * (1 + stop_dist)
        target   = price * (1 + stop_dist * REWARD_RISK) if side == "LONG" else price * (1 - stop_dist * REWARD_RISK)
        capital  = balance * RISK_PER_TRADE
        qty      = round_step((capital * LEVERAGE) / price, lot["step_size"])
        if qty < lot["min_qty"]:
            log.warning(f"Qty {qty} abaixo do minimo {lot['min_qty']}"); return

        raw_candles = get_klines(self.client, symbol).values.tolist()
        if not ask_gemini(symbol, side, signals, raw_candles):
            log.info(f"Gemini bloqueou {side} {symbol}.")
            return

        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side="BUY" if side == "LONG" else "SELL",
                type="MARKET",
                quantity=qty
            )
            entry = float(order["avgPrice"]) if float(order.get("avgPrice", 0)) > 0 else price
            self.open_trade = {
                "symbol": symbol, "side": side, "qty": qty,
                "entry": entry, "stop": round(stop, 8), "target": round(target, 8),
                "order_id": order["orderId"], "opened_at": time.time()
            }
            log.info(f"{'LONG' if side=='LONG' else 'SHORT'} {symbol} | qty={qty} | entrada={entry:.4f} | stop={stop:.4f} ({stop_dist*100:.2f}%) | target={target:.4f} ({stop_dist*REWARD_RISK*100:.2f}%) | {capital:.2f}x{LEVERAGE}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir: {e}")

    def _close_trade(self, reason, price):
        t = self.open_trade
        try:
            self.client.futures_create_order(
                symbol=t["symbol"],
                side="SELL" if t["side"] == "LONG" else "BUY",
                type="MARKET",
                quantity=t["qty"],
                reduceOnly=True
            )
            pnl = ((price - t["entry"]) / t["entry"] * 100) * (1 if t["side"] == "LONG" else -1) * LEVERAGE
            log.info(f"FECHOU [{reason}] {t['symbol']} | saida={price:.4f} | PnL={pnl:+.2f}%")
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar: {e}")
        finally:
            self.open_trade = None

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("Configure BINANCE_API_KEY e BINANCE_API_SECRET."); exit(1)
    EMAScalpBot().run()
