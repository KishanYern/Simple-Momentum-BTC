"""
indicators.py — Technical indicator feature generation using pandas-ta.

All indicators are appended in-place to the DataFrame via the pandas-ta
extension API (df.ta.<indicator>(append=True)).  After generation, any
warm-up NaN rows are dropped so the backtest always operates on fully
populated data.
"""

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401  — registers the .ta accessor

import config

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators required by the scoring system and
    append them as new columns to *df*.

    Indicators added
    ----------------
    RSI_14            — Relative Strength Index (speed / momentum)
    MACD_12_26_9      — MACD line
    MACDh_12_26_9     — MACD histogram  (crossover signal source)
    MACDs_12_26_9     — MACD signal line
    STOCHk_14_3_3     — Stochastic %K  (range placement)
    STOCHd_14_3_3     — Stochastic %D
    EMA_9             — Fast exponential moving average
    EMA_20            — Slow EMA (trailing stop anchor)
    ATRr_14           — Average True Range  (volatility / stop sizing)
    BBL_20_2.0        — Bollinger Lower Band
    BBM_20_2.0        — Bollinger Middle Band
    BBU_20_2.0        — Bollinger Upper Band
    BBB_20_2.0        — Bollinger Band Width  (regime filter)
    bb_width_prev     — Lagged BB width for expansion detection

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame (index=UTC DatetimeIndex,
        columns=[open, high, low, close, volume]).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with indicator columns appended and warm-up NaNs
        removed.
    """
    df = df.copy()

    # Ensure column names are lowercase (pandas-ta requirement)
    df.columns = [c.lower() for c in df.columns]

    logger.info("Computing RSI(%d) …", config.RSI_LENGTH)
    df.ta.rsi(length=config.RSI_LENGTH, append=True)

    logger.info(
        "Computing MACD(%d,%d,%d) …",
        config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL,
    )
    df.ta.macd(
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
        append=True,
    )

    logger.info(
        "Computing Stochastic(%d,%d,%d) …",
        config.STOCH_K, config.STOCH_SMOOTH_K, config.STOCH_D,
    )
    df.ta.stoch(
        k=config.STOCH_K,
        d=config.STOCH_D,
        smooth_k=config.STOCH_SMOOTH_K,
        append=True,
    )

    logger.info("Computing EMA(%d) and EMA(%d) …", config.EMA_FAST, config.EMA_SLOW)
    df.ta.ema(length=config.EMA_FAST, append=True)
    df.ta.ema(length=config.EMA_SLOW, append=True)

    logger.info("Computing ATR(%d) …", config.ATR_LENGTH)
    df.ta.atr(length=config.ATR_LENGTH, append=True)

    logger.info("Computing Bollinger Bands(%d, %.1f) …", config.BB_LENGTH, config.BB_STD)
    df.ta.bbands(length=config.BB_LENGTH, std=config.BB_STD, append=True)

    # Derived column: previous bar's BB width (used to detect expansion)
    # pandas-ta >= 0.4.x names BB width as BBB_{length}_{std}_{std}
    bb_width_col = f"BBB_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"
    # Fallback for older pandas-ta naming (BBB_{length}_{std})
    if bb_width_col not in df.columns:
        bb_width_col = f"BBB_{config.BB_LENGTH}_{config.BB_STD}"
    if bb_width_col in df.columns:
        df["bb_width_prev"] = df[bb_width_col].shift(1)

    # Drop warm-up rows where any core indicator is still NaN
    # Detect actual BB column name (handles both pandas-ta versions)
    bb_width_col_new = f"BBB_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"
    bb_width_col_old = f"BBB_{config.BB_LENGTH}_{config.BB_STD}"
    bb_width_col = bb_width_col_new if bb_width_col_new in df.columns else bb_width_col_old

    # Detect actual BB upper/lower/mid column names
    bb_prefix = f"BB{{}}_{config.BB_LENGTH}_{config.BB_STD}"
    bb_upper_col = f"BBU_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" if f"BBU_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" in df.columns else f"BBU_{config.BB_LENGTH}_{config.BB_STD}"
    bb_lower_col = f"BBL_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" if f"BBL_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" in df.columns else f"BBL_{config.BB_LENGTH}_{config.BB_STD}"
    bb_mid_col   = f"BBM_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" if f"BBM_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}" in df.columns else f"BBM_{config.BB_LENGTH}_{config.BB_STD}"

    # Store resolved column names as DataFrame attrs for downstream access
    df.attrs["bb_width_col"]  = bb_width_col
    df.attrs["bb_upper_col"]  = bb_upper_col
    df.attrs["bb_lower_col"]  = bb_lower_col
    df.attrs["bb_mid_col"]    = bb_mid_col

    core_cols = [
        f"RSI_{config.RSI_LENGTH}",
        f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}",
        f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}",
        f"STOCHk_{config.STOCH_K}_{config.STOCH_SMOOTH_K}_{config.STOCH_D}",
        f"EMA_{config.EMA_FAST}",
        f"EMA_{config.EMA_SLOW}",
        f"ATRr_{config.ATR_LENGTH}",
        bb_width_col,
        "bb_width_prev",
    ]
    before = len(df)
    df = df.dropna(subset=core_cols)
    logger.info(
        "Dropped %d warm-up rows; %d bars remain.",
        before - len(df), len(df),
    )

    return df


def get_column_names(df: "pd.DataFrame | None" = None) -> dict[str, str]:
    """
    Return a mapping of logical name → actual DataFrame column name.

    If *df* is supplied, the Bollinger Band column names are resolved from
    df.attrs (set by add_indicators) to handle the pandas-ta version
    difference (BBB_20_2.0 vs BBB_20_2.0_2.0).  When *df* is None, the
    fallback names for the newer pandas-ta version are returned.
    """
    if df is not None and "bb_width_col" in df.attrs:
        bb_lower = df.attrs["bb_lower_col"]
        bb_mid   = df.attrs["bb_mid_col"]
        bb_upper = df.attrs["bb_upper_col"]
        bb_width = df.attrs["bb_width_col"]
    else:
        # Default to pandas-ta >= 0.4.x naming
        bb_lower = f"BBL_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"
        bb_mid   = f"BBM_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"
        bb_upper = f"BBU_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"
        bb_width = f"BBB_{config.BB_LENGTH}_{config.BB_STD}_{config.BB_STD}"

    return {
        "rsi":           f"RSI_{config.RSI_LENGTH}",
        "macd_hist":     f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}",
        "macd_line":     f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}",
        "stoch_k":       f"STOCHk_{config.STOCH_K}_{config.STOCH_SMOOTH_K}_{config.STOCH_D}",
        "stoch_d":       f"STOCHd_{config.STOCH_K}_{config.STOCH_SMOOTH_K}_{config.STOCH_D}",
        "ema_fast":      f"EMA_{config.EMA_FAST}",
        "ema_slow":      f"EMA_{config.EMA_SLOW}",
        "atr":           f"ATRr_{config.ATR_LENGTH}",
        "bb_lower":      bb_lower,
        "bb_mid":        bb_mid,
        "bb_upper":      bb_upper,
        "bb_width":      bb_width,
        "bb_width_prev": "bb_width_prev",
    }
