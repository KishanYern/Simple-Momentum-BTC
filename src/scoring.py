"""
scoring.py — Fixed-weight momentum scoring system (Phase 2, Step 2.1 & 2.2).

Produces a signed integer *momentum_score* column for every bar in the
DataFrame.  Positive scores indicate bullish momentum; negative scores
indicate bearish momentum.  A regime filter hard-resets the score to 0
when the market is too quiet to trade.

Scoring table
-------------
Signal                             Long (+)    Short (−)
─────────────────────────────────  ─────────   ─────────
MACD histogram zero-line cross        +2           −2
RSI directional bias (>/<50)          +1           −1
Close vs 20-EMA                       +1           −1
Bollinger Band width expanding        +1           −1
Stochastic %K vs %D                   +1           −1
Maximum absolute score                 6            6

Regime filter (hard override)
─────────────────────────────
ATR < rolling 25th-percentile  → score = 0
BB width < rolling 25th-percentile → score = 0

Entry triggers (defined in config)
───────────────────────────────────
Long  entry: score >= +ENTRY_THRESHOLD  (default 5)
Short entry: score <= −ENTRY_THRESHOLD  (default −5)
"""

import logging

import numpy as np
import pandas as pd

import config
from src.indicators import get_column_names

logger = logging.getLogger(__name__)


def _macd_cross_signal(hist: pd.Series) -> pd.Series:
    """
    Return +1 when the MACD histogram crosses from negative to positive
    (or is positive after a recent cross), −1 for the opposite, 0 otherwise.

    We use a simple sign-change detection: a bullish cross is registered when
    the histogram is positive *and* was negative on the prior bar.  The signal
    persists as +1 / −1 for as long as the histogram stays on the same side,
    decaying back to 0 only when it crosses again.  This gives the cross
    condition its full +2 weight during a sustained trend.
    """
    sign = np.sign(hist)
    # Forward-fill the sign so the crossover value persists
    signal = sign.replace(0, np.nan).ffill().fillna(0).astype(int)
    return signal


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the signed momentum score for every bar and append it as
    'momentum_score' and 'long_signal' / 'short_signal' boolean columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by indicators.add_indicators().

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        - momentum_score  : int in [−6, 6]
        - long_signal     : bool  (score >= +ENTRY_THRESHOLD)
        - short_signal    : bool  (score <= −ENTRY_THRESHOLD)
        - regime_ok       : bool  (False when regime filter blocks trading)
    """
    df = df.copy()
    cols = get_column_names(df)

    # ------------------------------------------------------------------
    # Individual signal components  [each ∈ {−1, 0, +1} before weighting]
    # ------------------------------------------------------------------

    # 1. MACD histogram crossover direction (+2 / −2)
    macd_hist = df[cols["macd_hist"]]
    macd_dir  = _macd_cross_signal(macd_hist)           # +1 or −1
    macd_score = macd_dir * config.SCORE_MACD_CROSS     # +2 or −2

    # 2. RSI bias: >50 bullish, <50 bearish (+1 / −1)
    rsi       = df[cols["rsi"]]
    rsi_score = np.where(rsi > 50, 1, np.where(rsi < 50, -1, 0)) * config.SCORE_RSI_SIDE

    # 3. Price vs 20-EMA (+1 / −1)
    ema_slow  = df[cols["ema_slow"]]
    ema_score = np.where(
        df["close"] > ema_slow, 1, np.where(df["close"] < ema_slow, -1, 0)
    ) * config.SCORE_PRICE_EMA

    # 4. Bollinger Band width expanding vs prior bar (+1 / −1 / 0)
    bb_width  = df[cols["bb_width"]]
    bb_prev   = df[cols["bb_width_prev"]]
    bb_score  = np.where(
        bb_width > bb_prev, 1, np.where(bb_width < bb_prev, -1, 0)
    ) * config.SCORE_BB_EXPAND

    # 5. Stochastic %K vs %D (+1 / −1)
    stoch_k   = df[cols["stoch_k"]]
    stoch_d   = df[cols["stoch_d"]]
    stoch_score = np.where(
        stoch_k > stoch_d, 1, np.where(stoch_k < stoch_d, -1, 0)
    ) * config.SCORE_STOCH_CROSS

    # ------------------------------------------------------------------
    # Raw momentum score  (signed, range [−6, +6])
    # ------------------------------------------------------------------
    raw_score = (
        macd_score.values
        + rsi_score
        + ema_score
        + bb_score
        + stoch_score
    ).astype(int)

    # ------------------------------------------------------------------
    # Regime filter — suppress trading in low-volatility / choppy markets
    # ------------------------------------------------------------------
    atr       = df[cols["atr"]]
    atr_thresh = atr.rolling(config.REGIME_WINDOW, min_periods=1).quantile(
        config.REGIME_PERCENTILE / 100
    )

    bb_w       = df[cols["bb_width"]]
    bb_thresh  = bb_w.rolling(config.REGIME_WINDOW, min_periods=1).quantile(
        config.REGIME_PERCENTILE / 100
    )

    regime_ok = (atr >= atr_thresh) & (bb_w >= bb_thresh)

    # Zero out score when regime filter blocks trading
    momentum_score = pd.Series(
        np.where(regime_ok, raw_score, 0),
        index=df.index,
        dtype=int,
    )

    df["momentum_score"] = momentum_score
    df["regime_ok"]      = regime_ok
    df["long_signal"]    = momentum_score >= config.ENTRY_THRESHOLD
    df["short_signal"]   = momentum_score <= -config.ENTRY_THRESHOLD

    n_long  = df["long_signal"].sum()
    n_short = df["short_signal"].sum()
    logger.info(
        "Scoring complete: %d long signals, %d short signals "
        "(regime filter blocked %d bars).",
        n_long, n_short, (~regime_ok).sum(),
    )

    return df
