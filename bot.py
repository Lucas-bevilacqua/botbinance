"""
Binance Futures Aggressive Scalping Bot
Estrategia: EMA Cross (5/13) + RSI(7) + Volume Spike
Timeframe: 1min | Alavancagem: 10x
"""

import os
import time
import logging
import sqlite3
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np

# CONFIG
API_KEY        = os.getenv("BINANCE_API_KEY", "")
API_SECRET     = os.getenv("BINANCE_API_SECRET", "")

EMA_FAST       = 5
EMA_SLOW       = 13
EMA_TREND      = 50
RSI_PERIOD     = 7
RSI_MIN        = 35
RSI_MAX        = 65
ATR_PERIOD     = 7
ATR_STOP_MULT  = 1.0
VOL_MA_PERIOD  = 20
VOL_SPIKE_MULT = 1.2
LEVERAGE       = 10
RISK_PER_TRADE = 0.05
REWARD_RISK    = 2.0
MAX_DURATION   = 300
SCAN_INTERVAL  = 10
TIMEFRAME      = "1m"

CANDIDATE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT"
]

DB_PATH = "/app/data/trades.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/app/data/bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── DATABASE ─────────────────────────────────────────────────────────────────

def init_db():
    os.makedirs("/app/data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, side TEXT,
        entry REAL, exit REAL,
        qty REAL, pnl REAL, pnl_pct REAL,
        reason TEXT, duration INTEGER,
        opened_at TEXT, closed_at TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS balance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        balance REAL, ts TEXT
    )""")
    conn.commit()
    conn.close()

def db_log_balance(balance):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO balance_log (balance, ts) VALUES (?, ?)",
                     (balance, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except: pass

def db_log_trade(symbol, side, entry, exit_price, qty, pnl, pnl_pct, reason, duration, opened_at):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT INTO trades
            (symbol, side, entry, exit, qty, pnl, pnl_pct, reason, duration, opened_at, closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, side, entry, exit_price, qty, pnl, pnl_pct, reason, duration,
             opened_at, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except: pass

def db_get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        trades = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY closed_at DESC LIMIT 50").fetchall()]
        balance_history = [dict(r) for r in conn.execute(
            "SELECT balance, ts FROM balance_log ORDER BY ts DESC LIMIT 100").fetchall()]
        total = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()[0]
        wins  = conn.execute("SELECT COUNT(*) as c FROM trades WHERE pnl > 0").fetchone()[0]
        total_pnl = conn.execute("SELECT COALESCE(SUM(pnl),0) as s FROM trades").fetchone()[0]
        conn.close()
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        return {
            "trades": trades,
            "balance_history": list(reversed(balance_history)),
            "stats": {
                "total_trades": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": win_rate,
                "total_pnl": round(total_pnl, 4)
            }
        }
    except:
        return {"trades": [], "balance_history": [], "stats": {}}

# ─── INDICADORES ──────────────────────────────────────────────────────────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi

def calc_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def get_signals(df):
    if len(df) < max(EMA_SLOW, EMA_TREND, VOL_MA_PERIOD) + 5:
        return None

    close    = df["close"]
    volume   = df["volume"]
    ema_fast  = calc_ema(close, EMA_FAST)
    ema_slow  = calc_ema(close, EMA_SLOW)
    ema_trend = calc_ema(close, EMA_TREND)
    rsi       = calc_rsi(close, RSI_PERIOD)
    atr       = calc_atr(df, ATR_PERIOD)
    vol_ma    = volume.rolling(VOL_MA_PERIOD).mean()

    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        return None

    cross_up_now  = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    cross_up_prev = ema_fast.iloc[-2] > ema_slow.iloc[-2]
    cross_dn_now  = ema_fast.iloc[-1] < ema_slow.iloc[-1]
    cross_dn_prev = ema_fast.iloc[-2] < ema_slow.iloc[-2]

    bull_cross = cross_up_now and not cross_up_prev
    bear_cross = cross_dn_now and not cross_dn_prev

    last_close  = close.iloc[-1]
    last_trend  = ema_trend.iloc[-1]
    last_atr    = atr.iloc[-1]
    last_vol    = volume.iloc[-1]
    last_vol_ma = vol_ma.iloc[-1]

    vol_spike   = (not pd.isna(last_vol_ma)) and (last_vol >= last_vol_ma * VOL_SPIKE_MULT)
    rsi_ok      = RSI_MIN <= last_rsi <= RSI_MAX
    above_trend = last_close > last_trend
    below_trend = last_close < last_trend

    return {
        "buy":        bull_cross and above_trend and rsi_ok and vol_spike,
        "sell":       bear_cross and below_trend and rsi_ok and vol_spike,
        "exit_long":  cross_dn_now,
        "exit_short": cross_up_now,
        "rsi":        round(last_rsi, 2),
        "ema_fast":   round(ema_fast.iloc[-1], 6),
        "ema_slow":   round(ema_slow.iloc[-1], 6),
        "ema_trend":  round(last_trend, 6),
        "atr":        round(last_atr, 6),
        "vol_spike":  vol_spike,
        "close":      round(last_close, 6),
    }

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_klines(client, symbol, interval=TIMEFRAME, limit=150):
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

# ─── DASHBOARD HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ SCALP BOT</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080b0f;
    --surface: #0d1117;
    --border: #1e2730;
    --green: #00ff88;
    --red: #ff3355;
    --yellow: #ffcc00;
    --blue: #00aaff;
    --text: #e2e8f0;
    --muted: #4a5568;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    padding: 24px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }
  h1 {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
  }
  h1 span { color: var(--green); }
  .live-dot {
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .card-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }
  .card-value {
    font-size: 28px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
  }
  .card-value.green { color: var(--green); }
  .card-value.red { color: var(--red); }
  .card-value.yellow { color: var(--yellow); }
  .card-value.blue { color: var(--blue); }
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th {
    text-align: left;
    padding: 10px 12px;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 12px;
    border-bottom: 1px solid var(--border);
  }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .badge {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
  }
  .badge-long { background: rgba(0,255,136,0.15); color: var(--green); }
  .badge-short { background: rgba(255,51,85,0.15); color: var(--red); }
  .badge-win { background: rgba(0,255,136,0.15); color: var(--green); }
  .badge-loss { background: rgba(255,51,85,0.15); color: var(--red); }
  .chart-container {
    position: relative;
    height: 120px;
    margin-bottom: 32px;
  }
  canvas { width: 100% !important; }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }
  @media (max-width: 768px) {
    .two-col { grid-template-columns: 1fr; }
    .grid-4 { grid-template-columns: 1fr 1fr; }
  }
  .refresh-time {
    font-size: 11px;
    color: var(--muted);
  }
  .empty { color: var(--muted); text-align: center; padding: 32px; }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
</style>
</head>
<body>
<header>
  <h1>⚡ SCALP<span>BOT</span></h1>
  <div class="refresh-time">
    <span class="live-dot"></span>
    <span id="refresh-time">atualizando...</span>
  </div>
</header>

<div class="grid-4" id="stats-cards">
  <div class="card"><div class="card-label">Saldo</div><div class="card-value blue" id="balance">--</div></div>
  <div class="card"><div class="card-label">Total PnL</div><div class="card-value" id="total-pnl">--</div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value yellow" id="win-rate">--</div></div>
  <div class="card"><div class="card-label">Trades</div><div class="card-value" id="total-trades">--</div></div>
</div>

<div class="section-title">Saldo ao longo do tempo</div>
<div class="chart-container card" style="padding:16px">
  <canvas id="balanceChart"></canvas>
</div>

<div class="two-col">
  <div>
    <div class="section-title">Resumo</div>
    <div class="card">
      <table>
        <tr><td style="color:var(--muted)">Wins</td><td id="wins" class="pnl-pos">--</td></tr>
        <tr><td style="color:var(--muted)">Losses</td><td id="losses" class="pnl-neg">--</td></tr>
        <tr><td style="color:var(--muted)">Estratégia</td><td style="color:var(--yellow)">EMA 5/13 + RSI(7)</td></tr>
        <tr><td style="color:var(--muted)">Timeframe</td><td>1 minuto</td></tr>
        <tr><td style="color:var(--muted)">Alavancagem</td><td>10x</td></tr>
        <tr><td style="color:var(--muted)">Risco/trade</td><td>5%</td></tr>
      </table>
    </div>
  </div>
  <div>
    <div class="section-title">PnL por trade</div>
    <div class="card" style="height:200px;overflow:auto">
      <canvas id="pnlChart"></canvas>
    </div>
  </div>
</div>

<div class="section-title">Últimos trades</div>
<div class="card">
  <table>
    <thead>
      <tr>
        <th>Par</th><th>Lado</th><th>Entrada</th><th>Saída</th>
        <th>PnL%</th><th>PnL $</th><th>Motivo</th><th>Duração</th>
      </tr>
    </thead>
    <tbody id="trades-body">
      <tr><td colspan="8" class="empty">Carregando...</td></tr>
    </tbody>
  </table>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
let balanceChart, pnlChart;

function initCharts() {
  const ctx1 = document.getElementById('balanceChart').getContext('2d');
  balanceChart = new Chart(ctx1, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Saldo USDT', data: [], borderColor: '#00aaff', backgroundColor: 'rgba(0,170,255,0.1)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { ticks: { color: '#4a5568', font: { family: 'Space Mono', size: 10 } }, grid: { color: '#1e2730' } } } }
  });

  const ctx2 = document.getElementById('pnlChart').getContext('2d');
  pnlChart = new Chart(ctx2, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'PnL $', data: [], backgroundColor: [], borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { ticks: { color: '#4a5568', font: { family: 'Space Mono', size: 10 } }, grid: { color: '#1e2730' } } } }
  });
}

async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const data = await r.json();
    const s = data.stats;

    // Cards
    const bal = data.balance_history.length ? data.balance_history[data.balance_history.length-1].balance : 0;
    document.getElementById('balance').textContent = '$' + bal.toFixed(2);
    const pnlEl = document.getElementById('total-pnl');
    pnlEl.textContent = (s.total_pnl >= 0 ? '+' : '') + '$' + (s.total_pnl || 0).toFixed(4);
    pnlEl.className = 'card-value ' + (s.total_pnl >= 0 ? 'green' : 'red');
    document.getElementById('win-rate').textContent = (s.win_rate || 0) + '%';
    document.getElementById('total-trades').textContent = s.total_trades || 0;
    document.getElementById('wins').textContent = s.wins || 0;
    document.getElementById('losses').textContent = s.losses || 0;

    // Balance chart
    const bh = data.balance_history;
    balanceChart.data.labels = bh.map(b => b.ts.slice(11,16));
    balanceChart.data.datasets[0].data = bh.map(b => b.balance);
    balanceChart.update('none');

    // PnL chart
    const trades = data.trades.slice().reverse();
    pnlChart.data.labels = trades.map(t => t.symbol);
    pnlChart.data.datasets[0].data = trades.map(t => t.pnl);
    pnlChart.data.datasets[0].backgroundColor = trades.map(t => t.pnl >= 0 ? 'rgba(0,255,136,0.7)' : 'rgba(255,51,85,0.7)');
    pnlChart.update('none');

    // Trades table
    const tbody = document.getElementById('trades-body');
    if (!data.trades.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">Nenhum trade registrado ainda</td></tr>';
    } else {
      tbody.innerHTML = data.trades.map(t => `
        <tr>
          <td><b>${t.symbol}</b></td>
          <td><span class="badge badge-${t.side.toLowerCase()}">${t.side}</span></td>
          <td>${t.entry.toFixed(4)}</td>
          <td>${t.exit ? t.exit.toFixed(4) : '--'}</td>
          <td class="${t.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${t.pnl_pct >= 0 ? '+' : ''}${(t.pnl_pct||0).toFixed(2)}%</td>
          <td class="${t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${t.pnl >= 0 ? '+' : ''}$${(t.pnl||0).toFixed(4)}</td>
          <td><span class="badge ${t.reason=='STOP'?'badge-loss':'badge-win'}">${t.reason}</span></td>
          <td>${t.duration}s</td>
        </tr>`).join('');
    }

    document.getElementById('refresh-time').textContent = 'atualizado ' + new Date().toLocaleTimeString('pt-BR');
  } catch(e) {
    console.error(e);
  }
}

initCharts();
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""

# ─── FLASK API ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
_current_balance = 0.0

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/stats')
def api_stats():
    data = db_get_stats()
    data["current_balance"] = _current_balance
    return jsonify(data)

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "ts": datetime.utcnow().isoformat()})

def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# ─── BOT ──────────────────────────────────────────────────────────────────────

class AggressiveScalpBot:
    def __init__(self):
        global _current_balance
        self.client     = Client(API_KEY, API_SECRET)
        self.open_trade = None
        init_db()
        log.info("Bot conectado a Binance Futures")
        log.info(f"Estrategia AGRESSIVA: EMA{EMA_FAST}/{EMA_SLOW} + RSI({RSI_PERIOD}) + Volume | TF={TIMEFRAME}")
        log.info(f"Alavancagem: {LEVERAGE}x | Risco: {RISK_PER_TRADE*100:.0f}%/trade | R:R=1:{REWARD_RISK} | Timeout={MAX_DURATION}s")
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
                stop_dist = 0.006
                stop   = entry * (1 - stop_dist) if side == "LONG" else entry * (1 + stop_dist)
                target = entry * (1 + stop_dist * REWARD_RISK) if side == "LONG" else entry * (1 - stop_dist * REWARD_RISK)
                self.open_trade = {
                    "symbol": symbol, "side": side, "qty": qty,
                    "entry": entry, "stop": round(stop, 8), "target": round(target, 8),
                    "order_id": None, "opened_at": time.time(),
                    "opened_at_str": datetime.utcnow().isoformat()
                }
                log.info(f"Posicao existente: {side} {symbol} | entrada={entry:.4f}")
                break
            if not self.open_trade:
                log.info("Nenhuma posicao aberta. Pronto para operar.")
        except BinanceAPIException as e:
            log.error(f"Erro ao sincronizar: {e}")

    def run(self):
        global _current_balance
        log.info("Aggressive Scalp Bot iniciado")
        while True:
            try:
                self._cycle()
            except BinanceAPIException as e:
                log.error(f"Binance API error: {e}"); time.sleep(10)
            except Exception as e:
                log.error(f"Erro: {e}", exc_info=True); time.sleep(10)
            time.sleep(SCAN_INTERVAL)

    def _cycle(self):
        global _current_balance
        balance = get_futures_balance(self.client)
        _current_balance = balance
        log.info(f"Saldo: ${balance:.2f} USDT")
        db_log_balance(balance)

        if self.open_trade:
            t       = self.open_trade
            price   = get_futures_price(self.client, t["symbol"])
            elapsed = int(time.time() - t["opened_at"])
            side    = t["side"]
            pnl_pct = ((price - t["entry"]) / t["entry"] * 100) * (1 if side == "LONG" else -1) * LEVERAGE
            log.info(f"{side} {t['symbol']} | {price:.4f} | PnL={pnl_pct:+.2f}% | {elapsed}s")

            hit_stop   = price <= t["stop"]   if side == "LONG" else price >= t["stop"]
            hit_target = price >= t["target"] if side == "LONG" else price <= t["target"]
            timeout    = elapsed >= MAX_DURATION

            df      = get_klines(self.client, t["symbol"])
            signals = get_signals(df)
            st_exit = False
            if signals:
                st_exit = signals["exit_long"] if side == "LONG" else signals["exit_short"]

            if hit_stop or hit_target or timeout or st_exit:
                reason = "STOP" if hit_stop else ("TARGET" if hit_target else ("TIMEOUT" if timeout else "EMA_CROSS"))
                self._close_trade(reason, price, elapsed, pnl_pct)
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
            if signals is None:
                continue
            log.info(f"  {symbol} | {signals['close']} | RSI={signals['rsi']} | Vol={signals['vol_spike']} | BUY={signals['buy']} SELL={signals['sell']}")
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
        stop_dist = max(atr * ATR_STOP_MULT / price, 0.004)
        stop     = price * (1 - stop_dist) if side == "LONG" else price * (1 + stop_dist)
        target   = price * (1 + stop_dist * REWARD_RISK) if side == "LONG" else price * (1 - stop_dist * REWARD_RISK)
        capital  = balance * RISK_PER_TRADE
        qty      = round_step((capital * LEVERAGE) / price, lot["step_size"])
        if qty < lot["min_qty"]:
            log.warning(f"Qty {qty} abaixo do minimo {lot['min_qty']}"); return

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
                "order_id": order["orderId"], "opened_at": time.time(),
                "opened_at_str": datetime.utcnow().isoformat()
            }
            log.info(f"{'LONG' if side=='LONG' else 'SHORT'} {symbol} | qty={qty} | entrada={entry:.4f} | stop={stop:.4f} ({stop_dist*100:.2f}%) | target={target:.4f} | {capital:.2f}x{LEVERAGE}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir: {e}")

    def _close_trade(self, reason, price, duration, pnl_pct):
        t = self.open_trade
        try:
            self.client.futures_create_order(
                symbol=t["symbol"],
                side="SELL" if t["side"] == "LONG" else "BUY",
                type="MARKET",
                quantity=t["qty"],
                reduceOnly=True
            )
            capital = _current_balance * RISK_PER_TRADE
            pnl_dollar = (pnl_pct / 100) * capital
            log.info(f"FECHOU [{reason}] {t['symbol']} | saida={price:.4f} | PnL={pnl_pct:+.2f}% | ${pnl_dollar:+.4f}")
            db_log_trade(
                t["symbol"], t["side"], t["entry"], price,
                t["qty"], pnl_dollar, pnl_pct, reason, duration,
                t.get("opened_at_str", "")
            )
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar: {e}")
        finally:
            self.open_trade = None

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("Configure BINANCE_API_KEY e BINANCE_API_SECRET."); exit(1)
    # Inicia Flask em thread separada
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info("Dashboard disponivel em http://0.0.0.0:8080")
    AggressiveScalpBot().run()
