"""
Binance Futures Pro Scalping Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estrategia: Double Supertrend + ADX + RSI
  - Filtro direcional: Supertrend(10,3) no 15min
  - Gatilho de entrada: Supertrend(7,2) no 5min
  - Filtros de regime: ADX(14)>20 + RSI(14) 40-60
  - Janela operacional: 13:00-17:00 UTC
  - Stop: 1.5x ATR | Target: 3x risco (R:R 1:2)
  - Risk: 2% por trade | Max 2 losses consecutivos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, time, sqlite3, threading, logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Supertrend duplo
ST_FAST_PERIOD = 7;   ST_FAST_MULT = 2.0   # gatilho 5m
ST_SLOW_PERIOD = 10;  ST_SLOW_MULT = 3.0   # filtro 15m

# Filtros de regime
ADX_PERIOD    = 14;  ADX_MIN = 20          # tendencia minima
RSI_PERIOD    = 14;  RSI_MIN = 40; RSI_MAX = 60  # zona neutra

# Gestao de risco
LEVERAGE       = 10
RISK_PER_TRADE = 0.02   # 2% por trade
REWARD_RISK    = 2.0    # R:R 1:2
ATR_STOP_MULT  = 1.5
MAX_CONSEC_LOSSES = 2   # para apos 2 losses seguidos
MAX_DURATION   = 900    # 15 min timeout

# Janela operacional (UTC)
TRADE_HOUR_START = 13
TRADE_HOUR_END   = 17

# Pares (liquidez alta, notional minimo OK)
CANDIDATE_PAIRS = [
    "SOLUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT",
    "ATOMUSDT", "NEARUSDT", "APTUSDT", "INJUSDT", "UNIUSDT"
]

SCAN_INTERVAL = 30   # varre a cada 30s (15m tem menos urgencia)
DB_PATH       = "/app/data/trades.db"

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/app/data/bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs("/app/data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, side TEXT,
        entry REAL, exit REAL, qty REAL,
        pnl REAL, pnl_pct REAL,
        reason TEXT, duration INTEGER,
        opened_at TEXT, closed_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS balance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        balance REAL, ts TEXT
    )""")
    conn.commit(); conn.close()

def db_log_balance(bal):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO balance_log (balance,ts) VALUES (?,?)",
                     (bal, datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
    except: pass

def db_log_trade(symbol, side, entry, exit_p, qty, pnl, pnl_pct, reason, dur, opened_at):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT INTO trades
            (symbol,side,entry,exit,qty,pnl,pnl_pct,reason,duration,opened_at,closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, side, entry, exit_p, qty, pnl, pnl_pct, reason, dur,
             opened_at, datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
    except: pass

def db_get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        trades  = [dict(r) for r in conn.execute(
            "SELECT * FROM trades ORDER BY closed_at DESC LIMIT 50").fetchall()]
        bal_hist = [dict(r) for r in conn.execute(
            "SELECT balance,ts FROM balance_log ORDER BY ts DESC LIMIT 120").fetchall()]
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins  = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl>0").fetchone()[0]
        tot_pnl = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM trades").fetchone()[0]
        conn.close()
        return {
            "trades": trades,
            "balance_history": list(reversed(bal_hist)),
            "stats": {
                "total_trades": total, "wins": wins, "losses": total-wins,
                "win_rate": round(wins/total*100,1) if total else 0,
                "total_pnl": round(tot_pnl, 4)
            }
        }
    except:
        return {"trades": [], "balance_history": [], "stats": {}}

# ── INDICADORES ───────────────────────────────────────────────────────────────
def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_supertrend(df, period, multiplier):
    atr   = calc_atr(df, period)
    hl2   = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    close = df["close"]
    st    = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)  # 1=bull, -1=bear
    for i in range(1, len(df)):
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        cur_upper  = upper.iloc[i]
        cur_lower  = lower.iloc[i]
        cur_upper  = min(cur_upper, prev_upper) if close.iloc[i-1] <= prev_upper else cur_upper
        cur_lower  = max(cur_lower, prev_lower) if close.iloc[i-1] >= prev_lower else cur_lower
        upper.iloc[i] = cur_upper
        lower.iloc[i] = cur_lower
        prev_trend = trend.iloc[i-1] if i > 1 else 1
        if prev_trend == -1 and close.iloc[i] > prev_upper:
            trend.iloc[i] = 1
        elif prev_trend == 1 and close.iloc[i] < prev_lower:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
        st.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    trend.iloc[0] = 1
    st.iloc[0]    = lower.iloc[0]
    return trend, st

def calc_rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr    = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    dm_p  = (h - h.shift()).clip(lower=0)
    dm_n  = (l.shift() - l).clip(lower=0)
    dm_p[dm_p < dm_n] = 0
    dm_n[dm_n < dm_p] = 0
    atr14 = tr.ewm(alpha=1/period, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(alpha=1/period, adjust=False).mean() / atr14
    di_n  = 100 * dm_n.ewm(alpha=1/period, adjust=False).mean() / atr14
    dx    = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def get_klines(client, symbol, interval, limit=120):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume",
                        "ct","qav","nt","tbbav","tbqav","ignore"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

def get_signals(client, symbol):
    """
    Retorna sinal usando Double Supertrend + ADX + RSI.
    Filtro direcional: ST no 15m
    Gatilho: ST no 5m
    Regime: ADX>20 + RSI 40-60
    """
    try:
        df15 = get_klines(client, symbol, "15m", 120)
        df5  = get_klines(client, symbol, "5m",  60)

        # Supertrend lento no 15m (filtro direcional)
        trend15, _ = calc_supertrend(df15, ST_SLOW_PERIOD, ST_SLOW_MULT)
        dir15      = trend15.iloc[-1]   # 1=bull, -1=bear

        # Supertrend rapido no 5m (gatilho)
        trend5_now, _ = calc_supertrend(df5, ST_FAST_PERIOD, ST_FAST_MULT)
        dir5_now   = trend5_now.iloc[-1]
        dir5_prev  = trend5_now.iloc[-2]

        # Flip no 5m (momento do cruzamento)
        bull_flip = dir5_now == 1 and dir5_prev == -1
        bear_flip = dir5_now == -1 and dir5_prev == 1

        # Filtros de regime no 5m
        adx5 = calc_adx(df5, ADX_PERIOD)
        rsi5 = calc_rsi(df5["close"], RSI_PERIOD)
        atr5 = calc_atr(df5, 14)

        last_adx = adx5.iloc[-1]
        last_rsi = rsi5.iloc[-1]
        last_atr = atr5.iloc[-1]
        last_close = df5["close"].iloc[-1]

        if pd.isna(last_adx) or pd.isna(last_rsi):
            return None

        regime_ok  = last_adx > ADX_MIN and RSI_MIN <= last_rsi <= RSI_MAX
        above_trend = last_close > df5["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        below_trend = last_close < df5["close"].ewm(span=200, adjust=False).mean().iloc[-1]

        return {
            "buy":   bull_flip and dir15 == 1 and regime_ok,
            "sell":  bear_flip and dir15 == -1 and regime_ok,
            "exit_long":  dir5_now == -1,
            "exit_short": dir5_now == 1,
            "rsi":   round(last_rsi, 2),
            "adx":   round(last_adx, 2),
            "atr":   round(last_atr, 6),
            "close": round(last_close, 6),
            "dir15": dir15,
            "dir5":  dir5_now,
        }
    except Exception as e:
        log.error(f"get_signals {symbol}: {e}")
        return None

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_balance(client):
    for b in client.futures_account_balance():
        if b["asset"] == "USDT":
            return float(b["availableBalance"])
    return 0.0

def get_price(client, symbol):
    return float(client.futures_symbol_ticker(symbol=symbol)["price"])

def get_lot(client, symbol):
    for s in client.futures_exchange_info()["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return float(f["minQty"]), float(f["stepSize"])
    return 0.001, 0.001

def round_step(qty, step):
    dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(int(qty / step) * step, dec)

def set_leverage(client, symbol):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
    except: pass

def rank_pairs(client):
    tickers = {t["symbol"]: t for t in client.futures_ticker()}
    scores  = {}
    for sym in CANDIDATE_PAIRS:
        t = tickers.get(sym)
        if not t: continue
        vol = float(t["quoteVolume"])
        chg = abs(float(t["priceChangePercent"]))
        if vol < 20_000_000: continue
        scores[sym] = vol * chg
    return sorted(scores, key=scores.get, reverse=True)

def in_trade_window():
    """Verifica se estamos na janela operacional 13:00-17:00 UTC"""
    h = datetime.now(timezone.utc).hour
    return TRADE_HOUR_START <= h < TRADE_HOUR_END

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScalpBot Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
:root{--bg:#07090c;--card:#0e1318;--border:#1a2230;--green:#00e676;--red:#ff1744;--yellow:#ffd600;--blue:#2979ff;--purple:#d500f9;--text:#e8edf2;--sub:#4a5568}
html,body{width:100%;min-height:100vh;overflow-x:hidden}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;padding:14px;padding-bottom:32px}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 0 14px;border-bottom:1px solid var(--border);margin-bottom:16px}
.hdr-title{font-size:18px;font-weight:700;letter-spacing:-.5px}
.hdr-title b{color:var(--green)}
.hdr-sub{font-size:10px;color:var(--sub);margin-top:2px}
.live{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--sub)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.dot.off{background:var(--red);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 12px}
.card-top{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.card-num{font-size:22px;font-weight:700;line-height:1}
.c-blue .card-num{color:var(--blue)}.c-yellow .card-num{color:var(--yellow)}.c-purple .card-num{color:var(--purple)}
.c-green .card-num{color:var(--green)}.c-red .card-num{color:var(--red)}
.sec{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--sub);margin:16px 0 8px}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
.chart-inner{position:relative;width:100%;height:100px}
.chart-inner canvas{display:block;width:100%!important;height:100%!important}
.info-row{display:flex;flex-direction:column;gap:8px}
.info-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
.info-line{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px}
.info-line:last-child{border-bottom:none}
.info-line span:first-child{color:var(--sub)}
.info-line span:last-child{font-weight:600}
.trades{display:flex;flex-direction:column;gap:6px}
.tc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px;display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center}
.tc-sym{font-size:14px;font-weight:700;margin-top:4px}
.tc-meta{font-size:11px;color:var(--sub);margin-top:3px;line-height:1.5}
.tc-right{text-align:right;min-width:60px}
.tc-pct{font-size:15px;font-weight:700}
.tc-usd{font-size:11px;color:var(--sub);margin-top:1px}
.pos{color:var(--green)}.neg{color:var(--red)}
.b{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700}
.b-long{background:rgba(0,230,118,.12);color:var(--green)}.b-short{background:rgba(255,23,68,.12);color:var(--red)}
.b-ok{background:rgba(0,230,118,.12);color:var(--green)}.b-stop{background:rgba(255,23,68,.12);color:var(--red)}
.b-time{background:rgba(41,121,255,.12);color:var(--blue)}
.window-banner{padding:8px 14px;border-radius:8px;font-size:11px;font-weight:600;text-align:center;margin-bottom:12px}
.window-on{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2)}
.window-off{background:rgba(255,23,68,.08);color:var(--red);border:1px solid rgba(255,23,68,.15)}
.empty{text-align:center;padding:24px;color:var(--sub);font-size:12px}
@media(min-width:600px){body{padding:20px;max-width:800px;margin:0 auto}.cards{grid-template-columns:repeat(4,1fr);gap:12px}.info-row{flex-direction:row}.info-card{flex:1}.chart-inner{height:130px}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <div class="hdr-title">⚡ Scalp<b>Bot</b> <span style="font-size:11px;color:var(--yellow);font-weight:500">PRO</span></div>
    <div class="hdr-sub">Double Supertrend · ADX · RSI · 5m/15m</div>
  </div>
  <div class="live"><span class="dot" id="dot"></span><span id="ts">--</span></div>
</div>

<div class="window-banner" id="window-banner">Verificando janela...</div>

<div class="cards">
  <div class="card c-blue"><div class="card-top">Saldo</div><div class="card-num" id="balance">--</div></div>
  <div class="card"><div class="card-top">PnL Total</div><div class="card-num" id="pnl">--</div></div>
  <div class="card c-yellow"><div class="card-top">Win Rate</div><div class="card-num" id="wr">--</div></div>
  <div class="card c-purple"><div class="card-top">Trades</div><div class="card-num" id="tot">--</div></div>
</div>

<div class="sec">Saldo</div>
<div class="chart-card"><div class="chart-inner"><canvas id="balChart"></canvas></div></div>

<div class="sec">Resumo</div>
<div class="info-row">
  <div class="info-card">
    <div class="info-line"><span>Wins</span><span class="pos" id="wins">--</span></div>
    <div class="info-line"><span>Losses</span><span class="neg" id="losses">--</span></div>
    <div class="info-line"><span>Estratégia</span><span style="color:var(--yellow)">Double ST</span></div>
    <div class="info-line"><span>Timeframe</span><span>5m / 15m</span></div>
    <div class="info-line"><span>Janela UTC</span><span>13:00–17:00</span></div>
    <div class="info-line"><span>R:R</span><span>1:2</span></div>
  </div>
  <div class="info-card"><div class="chart-inner"><canvas id="pnlChart"></canvas></div></div>
</div>

<div class="sec">Últimos trades</div>
<div class="trades" id="trades-list"><div class="empty">Carregando...</div></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
const G={green:'#00e676',red:'#ff1744',blue:'#2979ff',sub:'#4a5568',border:'#1a2230',card:'#0e1318'};
let BC,PC;
function mkChart(id,type,color,fill){
  const ctx=document.getElementById(id);
  if(!ctx)return null;
  return new Chart(ctx,{type,data:{labels:[],datasets:[{data:[],borderColor:color,
    backgroundColor:fill?color+'22':color+'44',borderWidth:2,pointRadius:0,fill:!!fill,tension:.4,borderRadius:3}]},
    options:{responsive:false,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{ticks:{color:G.sub,font:{size:10},maxTicksLimit:3},grid:{color:G.border},border:{display:false}}}}});
}
function resizeCharts(){
  ['balChart','pnlChart'].forEach(id=>{const c=document.getElementById(id);if(c&&c.parentElement){c.width=c.parentElement.offsetWidth;c.height=c.parentElement.offsetHeight;}});
  if(BC)BC.resize();if(PC)PC.resize();
}
function badgeSide(s){return`<span class="b b-${s.toLowerCase()}">${s}</span>`}
function badgeReason(r){const m={STOP:'b-stop',TIMEOUT:'b-time'};return`<span class="b ${m[r]||'b-ok'}">${r}</span>`;}
async function refresh(){
  try{
    const d=await(await fetch('/api/stats')).json();
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
    if(PC){PC.data.labels=tr.map(t=>t.symbol.replace('USDT',''));PC.data.datasets[0].data=tr.map(t=>t.pnl);PC.data.datasets[0].backgroundColor=tr.map(t=>t.pnl>=0?G.green+'88':G.red+'88');PC.update('none');}
    const el=document.getElementById('trades-list');
    if(!d.trades?.length){el.innerHTML='<div class="empty">Nenhum trade ainda</div>';}
    else{el.innerHTML=d.trades.map(t=>{const pp=t.pnl_pct>=0,p$=t.pnl>=0;
      return`<div class="tc"><div><div>${badgeSide(t.side)}</div><div class="tc-sym">${t.symbol.replace('USDT','')}</div></div>
      <div class="tc-meta">${t.opened_at?t.opened_at.slice(11,16):'--'} · ${t.duration}s · ${badgeReason(t.reason)}<br>${t.entry?.toFixed(4)||'--'} → ${t.exit?.toFixed(4)||'--'}</div>
      <div class="tc-right"><div class="tc-pct ${pp?'pos':'neg'}">${pp?'+':''}${(t.pnl_pct||0).toFixed(2)}%</div><div class="tc-usd ${p$?'pos':'neg'}">${p$?'+':''}$${Math.abs(t.pnl||0).toFixed(4)}</div></div></div>`;
    }).join('');}
    // Janela operacional
    const h=new Date().getUTCHours();
    const inWin=h>=13&&h<17;
    const banner=document.getElementById('window-banner');
    banner.className='window-banner '+(inWin?'window-on':'window-off');
    banner.textContent=inWin?'🟢 Janela operacional ativa (13:00–17:00 UTC)':'🔴 Fora da janela (opera 13:00–17:00 UTC)';
    document.getElementById('dot').className='dot'+(inWin?'':' off');
    document.getElementById('ts').textContent=new Date().toLocaleTimeString('pt-BR');
  }catch(e){console.error(e)}
}
window.addEventListener('load',()=>{resizeCharts();BC=mkChart('balChart','line',G.blue,true);PC=mkChart('pnlChart','bar',G.green,false);refresh();setInterval(refresh,10000);});
window.addEventListener('resize',resizeCharts);
</script>
</body>
</html>"""

# ── FLASK ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
_current_balance = 0.0

@app.route('/')
def dashboard(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/stats')
def api_stats():
    d = db_get_stats()
    d["current_balance"] = _current_balance
    return jsonify(d)

@app.route('/api/health')
def health(): return jsonify({"status":"ok","ts":datetime.utcnow().isoformat()})

@app.route('/api/reset')
def reset_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM balance_log")
        conn.commit(); conn.close()
        return jsonify({"status":"ok","message":"Banco limpo"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# ── BOT ───────────────────────────────────────────────────────────────────────
class ProScalpBot:
    def __init__(self):
        global _current_balance
        self.client         = Client(API_KEY, API_SECRET)
        self.open_trade     = None
        self.consec_losses  = 0
        self.daily_losses   = 0
        self.paused         = False
        init_db()
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("ScalpBot PRO iniciado")
        log.info("Estrategia: Double Supertrend + ADX + RSI")
        log.info(f"Timeframes: 5m (gatilho) + 15m (filtro)")
        log.info(f"Janela: {TRADE_HOUR_START}:00-{TRADE_HOUR_END}:00 UTC")
        log.info(f"Risk: {RISK_PER_TRADE*100}%/trade | R:R 1:{REWARD_RISK} | {LEVERAGE}x")
        log.info(f"Max losses consecutivos: {MAX_CONSEC_LOSSES}")
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._sync_positions()

    def _sync_positions(self):
        try:
            for p in self.client.futures_position_information():
                amt = float(p["positionAmt"])
                if amt == 0: continue
                symbol = p["symbol"]
                side   = "LONG" if amt > 0 else "SHORT"
                entry  = float(p["entryPrice"])
                qty    = abs(amt)
                d      = 0.008
                stop   = entry*(1-d) if side=="LONG" else entry*(1+d)
                target = entry*(1+d*REWARD_RISK) if side=="LONG" else entry*(1-d*REWARD_RISK)
                self.open_trade = {"symbol":symbol,"side":side,"qty":qty,
                    "entry":entry,"stop":round(stop,8),"target":round(target,8),
                    "order_id":None,"opened_at":time.time(),"opened_at_str":datetime.utcnow().isoformat()}
                log.info(f"Posicao existente: {side} {symbol} entrada={entry:.4f}")
                break
            if not self.open_trade:
                log.info("Nenhuma posicao aberta. Pronto.")
        except BinanceAPIException as e:
            log.error(f"Sync error: {e}")

    def run(self):
        log.info("Bot loop iniciado")
        while True:
            try: self._cycle()
            except BinanceAPIException as e: log.error(f"API: {e}"); time.sleep(15)
            except Exception as e: log.error(f"Erro: {e}", exc_info=True); time.sleep(15)
            time.sleep(SCAN_INTERVAL)

    def _cycle(self):
        global _current_balance
        bal = get_balance(self.client)
        _current_balance = bal
        db_log_balance(bal)
        log.info(f"Saldo: ${bal:.2f} | Losses consec: {self.consec_losses}/{MAX_CONSEC_LOSSES} | Pausado: {self.paused}")

        # Gerenciar posicao aberta
        if self.open_trade:
            t = self.open_trade
            price   = get_price(self.client, t["symbol"])
            elapsed = int(time.time() - t["opened_at"])
            side    = t["side"]
            pnl_pct = ((price-t["entry"])/t["entry"]*100)*(1 if side=="LONG" else -1)*LEVERAGE
            log.info(f"{side} {t['symbol']} | {price:.4f} | PnL={pnl_pct:+.2f}% | {elapsed}s")

            hit_stop   = price <= t["stop"]   if side=="LONG" else price >= t["stop"]
            hit_target = price >= t["target"] if side=="LONG" else price <= t["target"]
            timeout    = elapsed >= MAX_DURATION
            sig        = get_signals(self.client, t["symbol"])
            st_exit    = False
            if sig:
                st_exit = sig["exit_long"] if side=="LONG" else sig["exit_short"]

            if hit_stop or hit_target or timeout or st_exit:
                reason = "STOP" if hit_stop else ("TARGET" if hit_target else ("TIMEOUT" if timeout else "ST_FLIP"))
                self._close(reason, price, elapsed, pnl_pct)
            return

        # Verificar pausa por losses consecutivos
        if self.paused:
            log.warning(f"Bot PAUSADO apos {MAX_CONSEC_LOSSES} losses consecutivos. Aguardando reset manual ou novo dia.")
            return

        # Verificar janela operacional
        if not in_trade_window():
            log.info(f"Fora da janela operacional (13:00-17:00 UTC). Hora UTC: {datetime.now(timezone.utc).hour}:00")
            return

        if bal < 1.0:
            log.warning("Saldo insuficiente."); return

        pairs = rank_pairs(self.client)
        if not pairs:
            log.info("Nenhum par com volume suficiente."); return

        log.info(f"Varrendo {len(pairs)} pares na janela operacional...")
        found = False
        for symbol in pairs:
            set_leverage(self.client, symbol)
            sig = get_signals(self.client, symbol)
            if not sig:
                continue
            log.info(f"  {symbol} | {sig['close']} | ADX={sig['adx']:.1f} | RSI={sig['rsi']:.1f} | ST15={sig['dir15']} ST5={sig['dir5']} | BUY={sig['buy']} SELL={sig['sell']}")
            if sig["buy"] or sig["sell"]:
                side = "LONG" if sig["buy"] else "SHORT"
                self._open(symbol, bal, side, sig)
                found = True; break

        if not found:
            log.info("Nenhum sinal valido. Aguardando...")

    def _open(self, symbol, balance, side, sig):
        price    = get_price(self.client, symbol)
        min_qty, step = get_lot(self.client, symbol)
        atr      = sig["atr"]
        stop_d   = max(atr * ATR_STOP_MULT / price, 0.005)
        stop     = price*(1-stop_d) if side=="LONG" else price*(1+stop_d)
        target   = price*(1+stop_d*REWARD_RISK) if side=="LONG" else price*(1-stop_d*REWARD_RISK)
        capital  = balance * RISK_PER_TRADE
        qty      = round_step((capital * LEVERAGE) / price, step)
        if qty < min_qty:
            log.warning(f"Qty {qty} < min {min_qty} em {symbol}"); return
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side="BUY" if side=="LONG" else "SELL",
                type="MARKET", quantity=qty)
            entry = float(order["avgPrice"]) if float(order.get("avgPrice",0)) > 0 else price
            self.open_trade = {"symbol":symbol,"side":side,"qty":qty,
                "entry":entry,"stop":round(stop,8),"target":round(target,8),
                "order_id":order["orderId"],"opened_at":time.time(),
                "opened_at_str":datetime.utcnow().isoformat()}
            log.info(f"ABRIU {side} {symbol} | qty={qty} | entrada={entry:.4f} | stop={stop:.4f} ({stop_d*100:.2f}%) | target={target:.4f} | capital={capital:.2f}x{LEVERAGE}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir: {e}")

    def _close(self, reason, price, duration, pnl_pct):
        t = self.open_trade
        try:
            self.client.futures_create_order(
                symbol=t["symbol"],
                side="SELL" if t["side"]=="LONG" else "BUY",
                type="MARKET", quantity=t["qty"], reduceOnly=True)
            pnl_dollar = (price - t["entry"]) * t["qty"] * (1 if t["side"]=="LONG" else -1)
            log.info(f"FECHOU [{reason}] {t['symbol']} | saida={price:.4f} | PnL={pnl_pct:+.2f}% | ${pnl_dollar:+.4f}")
            db_log_trade(t["symbol"], t["side"], t["entry"], price, t["qty"],
                        pnl_dollar, pnl_pct, reason, duration, t.get("opened_at_str",""))

            # Controle de losses consecutivos
            if pnl_dollar < 0:
                self.consec_losses += 1
                if self.consec_losses >= MAX_CONSEC_LOSSES:
                    self.paused = True
                    log.warning(f"⛔ Bot PAUSADO: {MAX_CONSEC_LOSSES} losses consecutivos. Revise manualmente.")
            else:
                self.consec_losses = 0  # reset ao ganhar
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar: {e}")
        finally:
            self.open_trade = None

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("Configure BINANCE_API_KEY e BINANCE_API_SECRET."); exit(1)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info("Dashboard em http://0.0.0.0:8080")
    ProScalpBot().run()
