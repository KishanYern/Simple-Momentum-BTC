"""
config.py — Single source of truth for all tuneable parameters.

Modify values here to adjust data range, indicator settings,
scoring weights, regime filter thresholds, and risk management rules
without touching any core logic files.
"""

from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
EXCHANGE_ID       = "kucoin"
SYMBOL            = "BTC/USDT"
TIMEFRAME         = "1h"          # 1-hour candles

# Pull ~2 years of history from today's date
_NOW              = datetime.now(timezone.utc)
START_DATE        = (_NOW - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%SZ")
END_DATE          = _NOW.strftime("%Y-%m-%dT%H:%M:%SZ")

DATA_DIR          = "data"
DATA_CACHE_FILE   = f"{DATA_DIR}/btc_ohlcv_1h.csv"
FETCH_LIMIT       = 1500         # candles per CCXT request (KuCoin max)

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------
RSI_LENGTH        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
STOCH_K           = 14
STOCH_D           = 3
STOCH_SMOOTH_K    = 3
EMA_FAST               = 20
EMA_SLOW               = 50
TRAILING_STOP_EMA_LEN  = 100   # wider EMA used only for the trailing stop anchor
ATR_LENGTH        = 14
BB_LENGTH         = 20
BB_STD            = 2.0

# ---------------------------------------------------------------------------
# Scoring system
# ---------------------------------------------------------------------------
# Individual signal weights (long perspective; signs are flipped for shorts)
SCORE_MACD_CROSS  = 2   # MACD histogram crosses zero
SCORE_RSI_SIDE    = 1   # RSI > 50 (long) / RSI < 50 (short)
SCORE_PRICE_EMA   = 1   # Close > EMA_20 (long) / Close < EMA_20 (short)
SCORE_BB_EXPAND   = 1   # Bollinger Bands widening
SCORE_STOCH_CROSS = 1   # Stoch %K > %D (long) / %K < %D (short)

SCORE_MAX         = (
    SCORE_MACD_CROSS + SCORE_RSI_SIDE + SCORE_PRICE_EMA +
    SCORE_BB_EXPAND  + SCORE_STOCH_CROSS
)  # = 6

# Minimum absolute score required to enter a trade
ENTRY_THRESHOLD   = 5   # score >=5 long, score <=-5 short

# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------
# Rolling window (in bars) for computing ATR / BB-width percentile thresholds
REGIME_WINDOW     = 100
# Percentile below which the market is considered "too quiet" to trade
REGIME_PERCENTILE = 50

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
# Take-profit = entry price ± (TP_ATR_MULT × ATR at entry)
TP_ATR_MULT       = 2.2
# Trailing stop anchor: the EMA_TRAILING_STOP_LEN EMA, checked every bar while in trade.
# Deliberately wider than EMA_SLOW so the stop and the entry-scoring EMA are decoupled.

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
INITIAL_CAPITAL   = 10_000.0      # USD
POSITION_SIZE_PCT = 0.10          # risk 10% of current equity per trade

# Output directory for generated plots
PLOT_DIR          = "plots"
