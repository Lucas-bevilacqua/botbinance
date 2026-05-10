"""
ScalpBot PRO v2
Estrategia: Triple EMA (9/21/50) 5m + EMA200 15m + RSI(9)
LONG:  EMA9>EMA21>EMA50 no 5m + preco>EMA200 no 15m + RSI 45-65
SHORT: EMA9<EMA21<EMA50 no 5m + preco<EMA200 no 15m + RSI 35-55
Stop: 1.5x ATR | Target: 2x risco | R:R 1:2
Opera 24/7 com todos filtros alinhados
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

EMA_FAST   = 9
EMA_MID    = 21
EMA_SLOW   = 50
EMA_TREND  = 200
RSI_PERIOD = 9
RSI_LONG_MIN = 45; RSI_LONG_MAX = 65
RSI_SHORT_MIN = 35; RSI_SHORT_MAX = 55

LEVERAGE       = 10
RISK_PER_TRADE = 0.02
REWARD_RISK    = 2.0
ATR_STOP_MULT  = 1.5
MAX_CONSEC_LOSSES = 3
MAX_DURATION   = 1800   # 30 min
SCAN_INTERVAL  = 20

CANDIDATE_PAIRS = [
    "SOLUSDT","BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT",
    "ADAUSDT","LINKUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "ATOMUSDT","NEARUSDT","APTUSDT","INJUSDT","UNIUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TIAUSDT","DOGEUSDT"
]

DB_PATH = "/app/data/trades.db"

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
            (symbol, side, entry, exit_p, qty, round(pnl,6), round(pnl_pct,4),
             reason, dur, opened_at, datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
    except: pass

def db_get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        trades   = [dict(r) for r in conn.execute(
            "SELECT * FROM trades ORDER BY closed_at DESC LIMIT 50").fetchall()]
        bal_hist = [dict(r) for r in conn.execute(
            "SELECT balance,ts FROM balance_log ORDER BY ts DESC LIMIT 120").fetchall()]
        total    = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins     = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl>0").fetchone()[0]
        tot_pnl  = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM trades").fetchone()[0]
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
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def get_klines(client, symbol, interval, limit=250):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume",
                        "ct","qav","nt","tbbav","tbqav","ignore"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

def get_signals(client, symbol):
    try:
        df5  = get_klines(client, symbol, "5m",  100)
        df15 = get_klines(client, symbol, "15m", 250)

        # EMAs no 5m
        e9  = ema(df5["close"], EMA_FAST)
        e21 = ema(df5["close"], EMA_MID)
        e50 = ema(df5["close"], EMA_SLOW)

        # EMA200 no 15m como filtro de tendencia macro
        e200_15m = ema(df15["close"], EMA_TREND)

        # RSI no 5m
        r = rsi(df5["close"], RSI_PERIOD)
        a = atr(df5, 14)

        last_close  = df5["close"].iloc[-1]
        last_e9     = e9.iloc[-1]
        last_e21    = e21.iloc[-1]
        last_e50    = e50.iloc[-1]
        last_e200   = e200_15m.iloc[-1]
        last_rsi    = r.iloc[-1]
        last_atr    = a.iloc[-1]

        if pd.isna(last_rsi) or pd.isna(last_e200):
            return None

        # Alinhamento das EMAs (tendencia forte)
        bull_align = last_e9 > last_e21 > last_e50   # cascata bullish
        bear_align = last_e9 < last_e21 < last_e50   # cascata bearish

        # Filtro macro: preco vs EMA200 no 15m
        above_macro = last_close > last_e200
        below_macro = last_close < last_e200

        # Confirmacao de momentum via RSI
        rsi_long  = RSI_LONG_MIN  <= last_rsi <= RSI_LONG_MAX
        rsi_short = RSI_SHORT_MIN <= last_rsi <= RSI_SHORT_MAX

        # Sinal de entrada: alinhamento + macro + rsi
        buy  = bull_align and above_macro and rsi_long
        sell = bear_align and below_macro and rsi_short

        # Saida: EMA9 cruza EMA21 na direcao contraria
        exit_long  = last_e9 < last_e21
        exit_short = last_e9 > last_e21

        return {
            "buy": buy, "sell": sell,
            "exit_long": exit_long, "exit_short": exit_short,
            "rsi": round(last_rsi, 2),
            "e9": round(last_e9, 6), "e21": round(last_e21, 6),
            "e50": round(last_e50, 6), "e200": round(last_e200, 6),
            "atr": round(last_atr, 6), "close": round(last_close, 6),
            "bull_align": bull_align, "bear_align": bear_align,
            "above_macro": above_macro,
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

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ScalpBot PRO v2</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
:root{--bg:#07090c;--card:#0e1318;--border:#1a2230;--green:#00e676;--red:#ff1744;--yellow:#ffd600;--blue:#2979ff;--purple:#d500f9;--text:#e8edf2;--sub:#4a5568}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;padding:14px;padding-bottom:32px;min-height:100vh;overflow-x:hidden}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 0 14px;border-bottom:1px solid var(--border);margin-bottom:16px}
.hdr-title{font-size:18px;font-weight:700;letter-spacing:-.5px}
.hdr-title b{color:var(--green)}
.hdr-sub{font-size:10px;color:var(--sub);margin-top:2px}
.live{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--sub)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
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
.b-time{background:rgba(41,121,255,.12);color:var(--blue)}.b-flip{background:rgba(255,214,0,.12);color:var(--yellow)}
.empty{text-align:center;padding:24px;color:var(--sub);font-size:12px}
.status-bar{padding:8px 14px;border-radius:8px;font-size:11px;font-weight:600;text-align:center;margin-bottom:12px;background:rgba(0,230,118,.08);color:var(--green);border:1px solid rgba(0,230,118,.15)}
@media(min-width:600px){body{padding:20px;max-width:820px;margin:0 auto}.cards{grid-template-columns:repeat(4,1fr);gap:12px}.info-row{flex-direction:row}.info-card{flex:1}.chart-inner{height:130px}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <div class="hdr-title">⚡ Scalp<b>Bot</b> <span style="font-size:11px;color:var(--purple);font-weight:600">v2</span></div>
    <div class="hdr-sub">Triple EMA 9/21/50 · EMA200 15m · RSI(9) · 24/7</div>
  </div>
  <div class="live"><span class="dot"></span><span id="ts">--</span></div>
</div>

<div class="status-bar" id="status">🟢 Bot ativo · Opera 24/7 com filtros alinhados</div>

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
    <div class="info-line"><span>Estratégia</span><span style="color:var(--yellow)">Triple EMA</span></div>
    <div class="info-line"><span>Timeframes</span><span>5m + 15m</span></div>
    <div class="info-line"><span>R:R</span><span>1:2</span></div>
    <div class="info-line"><span>Risco/trade</span><span>2%</span></div>
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
  const ctx=document.getElementById(id);if(!ctx)return null;
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
function bSide(s){return`<span class="b b-${s.toLowerCase()}">${s}</span>`}
function bReason(r){const m={STOP:'b-stop',TIMEOUT:'b-time',EMA_FLIP:'b-flip',ST_FLIP:'b-flip'};return`<span class="b ${m[r]||'b-ok'}">${r}</span>`;}
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
    if(PC){PC.data.labels=tr.map(t=>t.symbol.replace('USDT',''));PC.data.datasets[0].data=tr.map(t=>t.pnl);
      PC.data.datasets[0].backgroundColor=tr.map(t=>t.pnl>=0?G.green+'88':G.red+'88');PC.update('none');}
    const el=document.getElementById('trades-list');
    if(!d.trades?.length){el.innerHTML='<div class="empty">Nenhum trade ainda</div>';}
    else{el.innerHTML=d.trades.map(t=>{const pp=t.pnl_pct>=0,p$=t.pnl>=0;
      return`<div class="tc"><div><div>${bSide(t.side)}</div><div class="tc-sym">${t.symbol.replace('USDT','')}</div></div>
      <div class="tc-meta">${t.opened_at?t.opened_at.slice(11,16):'--'} · ${t.duration}s · ${bReason(t.reason)}<br>${t.entry?.toFixed(4)||'--'} → ${t.exit?.toFixed(4)||'--'}</div>
      <div class="tc-right"><div class="tc-pct ${pp?'pos':'neg'}">${pp?'+':''}${(t.pnl_pct||0).toFixed(2)}%</div>
      <div class="tc-usd ${p$?'pos':'neg'}">${p$?'+':''}$${Math.abs(t.pnl||0).toFixed(4)}</div></div></div>`;
    }).join('');}
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
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# ── BOT ───────────────────────────────────────────────────────────────────────
class ScalpBotV2:
    def __init__(self):
        global _current_balance
        self.client        = Client(API_KEY, API_SECRET)
        self.open_trade    = None
        self.consec_losses = 0
        self.paused        = False
        init_db()
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("ScalpBot PRO v2 iniciado")
        log.info("Triple EMA 9/21/50 (5m) + EMA200 (15m) + RSI(9)")
        log.info(f"Risk: {RISK_PER_TRADE*100}% | R:R 1:{REWARD_RISK} | {LEVERAGE}x | 24/7")
        log.info(f"Max losses consecutivos: {MAX_CONSEC_LOSSES}")
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
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
                self.open_trade = {
                    "symbol": symbol, "side": side, "qty": qty,
                    "entry": entry, "stop": round(stop,8), "target": round(target,8),
                    "order_id": None, "opened_at": time.time(),
                    "opened_at_str": datetime.utcnow().isoformat()
                }
                log.info(f"Posicao existente: {side} {symbol} | entrada={entry:.4f}")
                break
            if not self.open_trade:
                log.info("Nenhuma posicao aberta. Pronto para operar 24/7.")
        except BinanceAPIException as e:
            log.error(f"Sync error: {e}")

    def run(self):
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
        log.info(f"Saldo: ${bal:.2f} | Losses: {self.consec_losses}/{MAX_CONSEC_LOSSES} | Pausado: {self.paused}")

        if self.open_trade:
            t       = self.open_trade
            price   = get_price(self.client, t["symbol"])
            elapsed = int(time.time() - t["opened_at"])
            side    = t["side"]
            pnl_pct = ((price-t["entry"])/t["entry"]*100)*(1 if side=="LONG" else -1)*LEVERAGE
            log.info(f"{side} {t['symbol']} | {price:.4f} | PnL={pnl_pct:+.2f}% | {elapsed}s")

            hit_stop   = price <= t["stop"]   if side=="LONG" else price >= t["stop"]
            hit_target = price >= t["target"] if side=="LONG" else price <= t["target"]
            timeout    = elapsed >= MAX_DURATION
            sig        = get_signals(self.client, t["symbol"])
            ema_exit   = False
            if sig:
                ema_exit = sig["exit_long"] if side=="LONG" else sig["exit_short"]

            if hit_stop or hit_target or timeout or ema_exit:
                reason = "STOP" if hit_stop else ("TARGET" if hit_target else ("TIMEOUT" if timeout else "EMA_FLIP"))
                self._close(reason, price, elapsed, pnl_pct)
            return

        if self.paused:
            log.warning(f"Bot PAUSADO ({MAX_CONSEC_LOSSES} losses consecutivos). Reset em /api/reset")
            return

        if bal < 1.0:
            log.warning("Saldo insuficiente."); return

        pairs = rank_pairs(self.client)
        log.info(f"Varrendo {len(pairs)} pares...")
        found = False
        for symbol in pairs:
            set_leverage(self.client, symbol)
            sig = get_signals(self.client, symbol)
            if not sig: continue
            log.info(f"  {symbol} | {sig['close']} | RSI={sig['rsi']} | "
                     f"align={'BULL' if sig['bull_align'] else ('BEAR' if sig['bear_align'] else 'NONE')} | "
                     f"macro={'↑' if sig['above_macro'] else '↓'} | "
                     f"BUY={sig['buy']} SELL={sig['sell']}")
            if sig["buy"] or sig["sell"]:
                side = "LONG" if sig["buy"] else "SHORT"
                self._open(symbol, bal, side, sig)
                found = True; break

        if not found:
            log.info("Nenhum sinal. Aguardando...")

    def _open(self, symbol, balance, side, sig):
        price      = get_price(self.client, symbol)
        min_qty, step = get_lot(self.client, symbol)
        stop_dist  = max(sig["atr"] * ATR_STOP_MULT / price, 0.005)
        stop       = price*(1-stop_dist) if side=="LONG" else price*(1+stop_dist)
        target     = price*(1+stop_dist*REWARD_RISK) if side=="LONG" else price*(1-stop_dist*REWARD_RISK)
        capital    = balance * RISK_PER_TRADE
        qty        = round_step((capital * LEVERAGE) / price, step)
        if qty < min_qty:
            log.warning(f"Qty {qty} < min {min_qty} ({symbol})"); return
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side="BUY" if side=="LONG" else "SELL",
                type="MARKET", quantity=qty)
            entry = float(order["avgPrice"]) if float(order.get("avgPrice",0)) > 0 else price
            self.open_trade = {
                "symbol": symbol, "side": side, "qty": qty,
                "entry": entry, "stop": round(stop,8), "target": round(target,8),
                "order_id": order["orderId"], "opened_at": time.time(),
                "opened_at_str": datetime.utcnow().isoformat()
            }
            log.info(f"ABRIU {side} {symbol} | qty={qty} | entrada={entry:.4f} | "
                     f"stop={stop:.4f} ({stop_dist*100:.2f}%) | target={target:.4f}")
        except BinanceAPIException as e:
            log.error(f"Erro ao abrir: {e}")

    def _close(self, reason, price, duration, pnl_pct):
        t = self.open_trade
        try:
            self.client.futures_create_order(
                symbol=t["symbol"],
                side="SELL" if t["side"]=="LONG" else "BUY",
                type="MARKET", quantity=t["qty"], reduceOnly=True)
            pnl = (price - t["entry"]) * t["qty"] * (1 if t["side"]=="LONG" else -1)
            log.info(f"FECHOU [{reason}] {t['symbol']} | saida={price:.4f} | PnL={pnl_pct:+.2f}% | ${pnl:+.4f}")
            db_log_trade(t["symbol"], t["side"], t["entry"], price, t["qty"],
                        pnl, pnl_pct, reason, duration, t.get("opened_at_str",""))
            if pnl < 0:
                self.consec_losses += 1
                if self.consec_losses >= MAX_CONSEC_LOSSES:
                    self.paused = True
                    log.warning(f"⛔ PAUSADO: {MAX_CONSEC_LOSSES} losses consecutivos!")
            else:
                self.consec_losses = 0
        except BinanceAPIException as e:
            log.error(f"Erro ao fechar: {e}")
        finally:
            self.open_trade = None

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("Configure BINANCE_API_KEY e BINANCE_API_SECRET."); exit(1)
    threading.Thread(target=run_flask, daemon=True).start()
    log.info("Dashboard em http://0.0.0.0:8080")
    ScalpBotV2().run()
