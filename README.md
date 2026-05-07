# 🤖 Binance Day Trading Bot

Estratégia: **RSI 14 + EMA 9/21 Crossover**  
Risco por trade: **12% do saldo**  
Stop Loss: **4%** | Take Profit: **8%**  
Seleção de par: **automática** (maior volume + momentum)

---

## ⚙️ Como funciona

1. A cada 60s o bot escaneia os 10 pares mais líquidos
2. Seleciona o par com maior volume e momentum positivo
3. Analisa RSI + cruzamento de EMAs no candle de 15 minutos
4. **Compra** quando: EMA9 cruza acima da EMA21 + RSI entre 40-65
5. **Vende** quando: hit no stop (−4%), hit no target (+8%) ou sinal de venda
6. Só 1 trade aberto por vez (ideal para conta pequena)

---

## 🚀 Deploy no Railway (gratuito)

### Passo 1 — Criar chave de API na Binance
1. Acesse binance.com → Perfil → Gerenciamento de API
2. Crie nova chave → habilite apenas **"Spot & Margin Trading"**
3. Adicione restrição de IP se possível (mais seguro)
4. Salve a **API Key** e o **Secret Key**

### Passo 2 — Subir no GitHub
```bash
git init
git add .
git commit -m "trading bot"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/trading-bot.git
git push -u origin main
```

### Passo 3 — Deploy no Railway
1. Acesse **railway.app** → New Project → Deploy from GitHub
2. Selecione seu repositório
3. Vá em **Variables** e adicione:
   - `BINANCE_API_KEY` = sua chave
   - `BINANCE_API_SECRET` = seu secret
4. Railway detecta o Dockerfile automaticamente e faz o deploy
5. Pronto! O bot roda 24/7 gratuitamente

---

## 📊 Acompanhar logs

No Railway: clique no serviço → aba **Logs** → você vê tudo em tempo real

No arquivo local: `bot.log` é criado automaticamente

---

## ⚠️ Avisos importantes

- **Nunca** compartilhe suas chaves de API com ninguém
- Comece em modo de **teste** descomentando o Testnet (ver abaixo)
- Com $10, o bot opera de forma conservadora em capital — uma sequência ruim pode zerar
- Não é garantia de lucro. Mercado cripto é altamente volátil.

---

## 🧪 Testar sem dinheiro real (Testnet)

Substitua no `bot.py`:
```python
self.client = Client(API_KEY, API_SECRET, testnet=True)
```
E use chaves do testnet: https://testnet.binance.vision/

---

## 📈 Parâmetros ajustáveis (bot.py)

| Variável | Padrão | Descrição |
|---|---|---|
| `RISK_PER_TRADE` | 0.12 | 12% do saldo por trade |
| `STOP_LOSS_PCT` | 0.04 | Stop loss em 4% |
| `TAKE_PROFIT_PCT` | 0.08 | Take profit em 8% |
| `SCAN_INTERVAL_SEC` | 60 | Intervalo entre scans |
| `MIN_VOLUME_USDT` | 5.000.000 | Volume mínimo do par |
