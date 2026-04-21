# Simple Momentum BTC

A heuristic momentum trading algorithm for BTC/USDT built in pure Python.

---

## Architecture

```
Simple-Momentum-BTC/
├── config.py              ← All tuneable parameters (change here, not in src/)
├── main.py                ← Run this: orchestrates pipeline + generates plots
├── requirements.txt
├── src/
│   ├── data_fetcher.py    ← CCXT / Binance OHLCV ingestion (incremental cache)
│   ├── indicators.py      ← pandas-ta indicator computation
│   ├── scoring.py         ← Fixed-weight momentum scoring + regime filter
│   └── backtest.py        ← Event-driven backtest engine
├── data/                  ← Auto-created; holds cached CSV (git-ignored)
└── plots/                 ← Auto-created; holds generated PNGs (git-ignored)
```

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (first run downloads ~17,000 1h candles — takes ~2 min)
python main.py

# 4. Force a fresh data download (ignores cache)
python main.py --refresh
```

Plots are written to `plots/`:

| File | Contents |
|---|---|
| `01_price_chart.png` | BTC price + EMA-9/20 + Bollinger Bands + trade markers |
| `02_equity_curve.png` | Equity curve & drawdown |
| `03_momentum_score.png` | Per-bar momentum score with regime filter overlay |

---

## Phase 1 — Data & Features

**Data**: 1-hour BTC/USDT candles from Binance (≈ 2 years, ~17,000 bars).  
Incremental caching — only new candles are downloaded on subsequent runs.

**Indicators** (via `pandas-ta`):

| Indicator | Config key | Purpose |
|---|---|---|
| RSI(14) | `RSI_LENGTH` | Momentum speed |
| MACD(12,26,9) | `MACD_FAST/SLOW/SIGNAL` | Trend direction |
| Stochastic(14,3,3) | `STOCH_K/D` | Range placement |
| EMA-9 | `EMA_FAST` | Fast dynamic S/R |
| EMA-20 | `EMA_SLOW` | Slow S/R + trailing stop anchor |
| ATR(14) | `ATR_LENGTH` | Volatility + TP/SL sizing |
| Bollinger Bands(20,2) | `BB_LENGTH/STD` | Volatility regime |

---

## Phase 2 — Scoring & Backtest

### Scoring System (max ±6)

| Signal | Long | Short |
|---|---|---|
| MACD histogram on positive/negative side | +2 | −2 |
| RSI > 50 / < 50 | +1 | −1 |
| Close > EMA-20 / < EMA-20 | +1 | −1 |
| BB width expanding vs prior bar | +1 | −1 |
| Stochastic %K > %D / %K < %D | +1 | −1 |

### Regime Filter
- **ATR < 25th-percentile (100-bar rolling)** → score = 0
- **BB width < 25th-percentile (100-bar rolling)** → score = 0

### Entry / Exit Rules
- **Long entry**: score ≥ +5 → buy at next bar's open
- **Short entry**: score ≤ −5 → sell at next bar's open
- **Take-profit**: entry ± 2 × ATR (long: +, short: −)
- **Trailing stop**: 20-EMA value at each bar while in trade

### Risk Management
- 10% of current equity allocated per trade (`POSITION_SIZE_PCT`)
- Starting capital: $10,000 (`INITIAL_CAPITAL`)

---

## Key Config Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `TIMEFRAME` | `"1h"` | Candle size |
| `ENTRY_THRESHOLD` | `5` | Min score to enter (long ≥ +5, short ≤ −5) |
| `TP_ATR_MULT` | `2.0` | Take-profit distance in ATR multiples |
| `POSITION_SIZE_PCT` | `0.10` | Equity fraction per trade |
| `REGIME_PERCENTILE` | `25` | ATR / BB-width percentile cut-off |

---

## Roadmap

- **Phase 3**: Replace fixed weights with an ML model (XGBoost / LSTM).
- **Phase 4**: Walk-forward validation + Sharpe-based hyperparameter tuning.
- **Phase 5**: Live paper trading via CCXT.
