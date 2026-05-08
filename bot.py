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
RSI_MIN        = 45
RSI_MAX        = 55
ATR_PERIOD     = 7
ATR_STOP_MULT  = 1.2
VOL_MA_PERIOD  = 20
VOL_SPIKE_MULT = 1.5
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

    # Filtro de momentum: preco deve estar se afastando da EMA50, nao lateralizando
    ema_dist_pct = abs(last_close - last_trend) / last_trend
    momentum_ok  = ema_dist_pct > 0.003  # pelo menos 0.3% de distancia da EMA50

    return {
        "buy":        bull_cross and above_trend and rsi_ok and vol_spike and momentum_ok,
        "sell":       bear_cross and below_trend and rsi_ok and vol_spike and momentum_ok,
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
<title>ScalpBot</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
:root{
  --bg:#0a0c10;--card:#111318;--border:#1e2530;
  --green:#00e676;--red:#ff1744;--yellow:#ffd600;--blue:#2979ff;--purple:#d500f9;
  --text:#e8edf2;--sub:#4a5568;
}
html,body{width:100%;min-height:100vh;overflow-x:hidden}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;padding:12px}

/* HEADER */
.hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 0 14px;border-bottom:1px solid var(--border);margin-bottom:16px}
.hdr-title{font-size:18px;font-weight:700;letter-spacing:-.5px}
.hdr-title b{color:var(--green)}
.hdr-live{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--sub)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* CARDS 2x2 mobile */
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 12px}
.card-top{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.card-num{font-size:22px;font-weight:700;line-height:1}
.c-blue .card-num{color:var(--blue)}
.c-yellow .card-num{color:var(--yellow)}
.c-purple .card-num{color:var(--purple)}
.c-green .card-num{color:var(--green)}
.c-red .card-num{color:var(--red)}

/* SECTION */
.sec{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--sub);margin:16px 0 8px}

/* CHART */
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px;margin-bottom:0}
.chart-inner{position:relative;width:100%;height:100px}
.chart-inner canvas{display:block;width:100%!important;height:100%!important}

/* INFO ROW - stacked no mobile */
.info-row{display:flex;flex-direction:column;gap:8px}
.info-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
.info-line{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px}
.info-line:last-child{border-bottom:none}
.info-line span:first-child{color:var(--sub)}
.info-line span:last-child{font-weight:600}

/* TRADE CARDS */
.trades{display:flex;flex-direction:column;gap:6px}
.tc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center}
.tc-left{}
.tc-sym{font-size:14px;font-weight:700;margin-top:4px}
.tc-meta{font-size:11px;color:var(--sub);margin-top:3px;line-height:1.5}
.tc-right{text-align:right;min-width:60px}
.tc-pct{font-size:15px;font-weight:700}
.tc-usd{font-size:11px;color:var(--sub);margin-top:1px}
.pos{color:var(--green)}
.neg{color:var(--red)}

/* BADGE */
.b{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700}
.b-long{background:rgba(0,230,118,.12);color:var(--green)}
.b-short{background:rgba(255,23,68,.12);color:var(--red)}
.b-ok{background:rgba(0,230,118,.12);color:var(--green)}
.b-stop{background:rgba(255,23,68,.12);color:var(--red)}
.b-time{background:rgba(41,121,255,.12);color:var(--blue)}

.empty{text-align:center;padding:24px;color:var(--sub);font-size:12px}

/* DESKTOP */
@media(min-width:600px){
  body{padding:20px;max-width:800px;margin:0 auto}
  .cards{grid-template-columns:repeat(4,1fr);gap:12px}
  .chart-inner{height:130px}
  .info-row{flex-direction:row}
  .info-card{flex:1}
}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-title">⚡ Scalp<b>Bot</b></div>
  <div class="hdr-live"><span class="dot"></span><span id="ts">--</span></div>
</div>

<div class="cards">
  <div class="card c-blue">
    <div class="card-top">Saldo</div>
    <div class="card-num" id="balance">--</div>
  </div>
  <div class="card">
    <div class="card-top">PnL Total</div>
    <div class="card-num" id="pnl">--</div>
  </div>
  <div class="card c-yellow">
    <div class="card-top">Win Rate</div>
    <div class="card-num" id="wr">--</div>
  </div>
  <div class="card c-purple">
    <div class="card-top">Trades</div>
    <div class="card-num" id="tot">--</div>
  </div>
</div>

<div class="sec">Saldo</div>
<div class="chart-card">
  <div class="chart-inner"><canvas id="balChart"></canvas></div>
</div>

<div class="sec">Resumo</div>
<div class="info-row">
  <div class="info-card">
    <div class="info-line"><span>Wins</span><span class="pos" id="wins">--</span></div>
    <div class="info-line"><span>Losses</span><span class="neg" id="losses">--</span></div>
    <div class="info-line"><span>Estratégia</span><span style="color:var(--yellow)">EMA 5/13</span></div>
    <div class="info-line"><span>Timeframe</span><span>1 min</span></div>
    <div class="info-line"><span>Alavancagem</span><span>10x</span></div>
    <div class="info-line"><span>Risco/trade</span><span>5%</span></div>
  </div>
  <div class="info-card">
    <div class="chart-inner"><canvas id="pnlChart"></canvas></div>
  </div>
</div>

<div class="sec">Últimos trades</div>
<div class="trades" id="trades-list">
  <div class="empty">Carregando...</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
const G={green:'#00e676',red:'#ff1744',blue:'#2979ff',sub:'#4a5568',border:'#1e2530',card:'#111318'};
let BC,PC;

function mkChart(id,type,color,fill){
  const ctx=document.getElementById(id);
  if(!ctx)return null;
  return new Chart(ctx,{
    type,
    data:{labels:[],datasets:[{data:[],borderColor:color,
      backgroundColor:fill?color+'22':color+'44',
      borderWidth:2,pointRadius:0,fill:!!fill,tension:.4,borderRadius:3}]},
    options:{
      responsive:false,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false},tooltip:{enabled:true,backgroundColor:G.card,
        titleColor:'#e8edf2',bodyColor:G.sub,borderColor:G.border,borderWidth:1}},
      scales:{
        x:{display:false},
        y:{ticks:{color:G.sub,font:{size:10},maxTicksLimit:3},grid:{color:G.border},border:{display:false}}
      }
    }
  });
}

function resizeCharts(){
  ['balChart','pnlChart'].forEach(id=>{
    const c=document.getElementById(id);
    if(c&&c.parentElement){
      c.width=c.parentElement.offsetWidth;
      c.height=c.parentElement.offsetHeight;
    }
  });
  if(BC)BC.resize();
  if(PC)PC.resize();
}

function badgeSide(s){return`<span class="b b-${s.toLowerCase()}">${s}</span>`}
function badgeReason(r){
  const m={STOP:'b-stop',TIMEOUT:'b-time'};
  return`<span class="b ${m[r]||'b-ok'}">${r}</span>`;
}

async function refresh(){
  try{
    const r=await fetch('/api/stats');
    const d=await r.json();
    const s=d.stats||{};

    const bal=d.balance_history?.length?d.balance_history[d.balance_history.length-1].balance:0;
    document.getElementById('balance').textContent='$'+bal.toFixed(2);

    const pnl=s.total_pnl||0;
    const pe=document.getElementById('pnl');
    pe.textContent=(pnl>=0?'+':'')+'$'+Math.abs(pnl).toFixed(4);
    pe.className='card-num '+(pnl>=0?'c-green':'c-red');

    document.getElementById('wr').textContent=(s.win_rate||0)+'%';
    document.getElementById('tot').textContent=s.total_trades||0;
    document.getElementById('wins').textContent=s.wins||0;
    document.getElementById('losses').textContent=s.losses||0;

    const bh=d.balance_history||[];
    if(BC){BC.data.labels=bh.map(b=>b.ts.slice(11,16));BC.data.datasets[0].data=bh.map(b=>b.balance);BC.update('none');}

    const tr=(d.trades||[]).slice().reverse();
    if(PC){PC.data.labels=tr.map(t=>t.symbol.replace('USDT',''));PC.data.datasets[0].data=tr.map(t=>t.pnl);
      PC.data.datasets[0].backgroundColor=tr.map(t=>t.pnl>=0?G.green+'88':G.red+'88');PC.update('none');}

    const el=document.getElementById('trades-list');
    if(!d.trades?.length){el.innerHTML='<div class="empty">Nenhum trade ainda</div>';}
    else{
      el.innerHTML=d.trades.map(t=>{
        const pp=t.pnl_pct>=0,p$=t.pnl>=0;
        return`<div class="tc">
          <div><div>${badgeSide(t.side)}</div><div class="tc-sym">${t.symbol.replace('USDT','')}</div></div>
          <div class="tc-meta">
            ${t.opened_at?t.opened_at.slice(11,16):'--'} · ${t.duration}s · ${badgeReason(t.reason)}<br>
            ${t.entry?.toFixed(4)||'--'} → ${t.exit?.toFixed(4)||'--'}
          </div>
          <div class="tc-right">
            <div class="tc-pct ${pp?'pos':'neg'}">${pp?'+':''}${(t.pnl_pct||0).toFixed(2)}%</div>
            <div class="tc-usd ${p$?'pos':'neg'}">${p$?'+':''}$${Math.abs(t.pnl||0).toFixed(4)}</div>
          </div>
        </div>`;
      }).join('');
    }
    document.getElementById('ts').textContent=new Date().toLocaleTimeString('pt-BR');
  }catch(e){console.error(e)}
}

window.addEventListener('load',()=>{
  resizeCharts();
  BC=mkChart('balChart','line',G.blue,true);
  PC=mkChart('pnlChart','bar',G.green,false);
  refresh();
  setInterval(refresh,10000);
});
window.addEventListener('resize',resizeCharts);
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

@app.route('/api/reset', methods=['POST'])
def reset_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM balance_log")
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "message": "Banco limpo"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
            # PnL real = variacao de preco * quantidade
            pnl_dollar = (price - t["entry"]) * t["qty"] * (1 if t["side"] == "LONG" else -1)
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
