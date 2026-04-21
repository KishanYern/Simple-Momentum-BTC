"""
data_fetcher.py — Historical OHLCV ingestion via CCXT.

Connects to Binance (no API key required for public market data),
paginates through fetch_ohlcv to collect 1-hour candles for the full
date range defined in config, and caches the result to CSV.

On subsequent runs only the candles after the last cached timestamp are
fetched, keeping network usage minimal.
"""

import os
import time
import logging

import ccxt
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_exchange() -> ccxt.Exchange:
    """
    Instantiate and configure the exchange object.

    We skip load_markets() deliberately — it contacts a geo-restricted
    endpoint on Binance/Bybit.  fetch_ohlcv() does not require market
    info to be pre-loaded.
    """
    exchange = getattr(ccxt, config.EXCHANGE_ID)({
        "enableRateLimit": True,
        "options": {
            # Bybit: tell CCXT which market type to use without
            # calling load_markets().  Ignored by other exchanges.
            "defaultType": "linear",
        },
    })
    return exchange


def _fetch_range(
    exchange: ccxt.Exchange,
    since_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Paginate CCXT fetch_ohlcv calls between *since_ms* and *end_ms*
    (both in milliseconds).  Returns a DataFrame with a tz-aware UTC
    DatetimeIndex and columns [open, high, low, close, volume].
    """
    all_rows: list[list] = []

    while since_ms < end_ms:
        logger.info(
            "Fetching %s %s candles from %s …",
            config.SYMBOL,
            config.TIMEFRAME,
            pd.Timestamp(since_ms, unit="ms", tz="UTC"),
        )
        batch = exchange.fetch_ohlcv(
            config.SYMBOL,
            timeframe=config.TIMEFRAME,
            since=since_ms,
            limit=config.FETCH_LIMIT,
        )
        if not batch:
            logger.info("No more data returned — stopping pagination.")
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # Compute the millisecond increment for one candle
        tf_ms = exchange.parse_timeframe(config.TIMEFRAME) * 1000
        since_ms = last_ts + tf_ms

        # Politely obey exchange rate limits
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] < pd.Timestamp(end_ms, unit="ms", tz="UTC")]
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a complete OHLCV DataFrame for the configured symbol / timeframe.

    If the cache file exists and *force_refresh* is False, only the candles
    newer than the last cached timestamp are downloaded and appended.
    Otherwise the full date range is re-fetched from scratch.

    Parameters
    ----------
    force_refresh : bool
        When True, ignore any existing cache and re-download everything.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: tz-aware UTC DatetimeIndex named 'timestamp'.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    cache_path = config.DATA_CACHE_FILE

    exchange = _build_exchange()
    start_ms  = exchange.parse8601(config.START_DATE)
    end_ms    = exchange.parse8601(config.END_DATE)

    # ------------------------------------------------------------------
    # Attempt incremental update from cache
    # ------------------------------------------------------------------
    cached_df: pd.DataFrame | None = None

    if not force_refresh and os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(
                cache_path,
                index_col="timestamp",
                parse_dates=True,
            )
            # Ensure timezone-aware index
            if cached_df.index.tz is None:
                cached_df.index = cached_df.index.tz_localize("UTC")

            last_ts_ms = int(cached_df.index.max().timestamp() * 1000)
            tf_ms      = exchange.parse_timeframe(config.TIMEFRAME) * 1000
            fetch_from = last_ts_ms + tf_ms

            logger.info(
                "Cache found (%d rows). Fetching incremental update from %s.",
                len(cached_df),
                pd.Timestamp(fetch_from, unit="ms", tz="UTC"),
            )
        except Exception as exc:
            logger.warning("Could not read cache (%s). Fetching from scratch.", exc)
            cached_df  = None
            fetch_from = start_ms
    else:
        fetch_from = start_ms

    # ------------------------------------------------------------------
    # Fetch new candles
    # ------------------------------------------------------------------
    new_df = _fetch_range(exchange, fetch_from, end_ms)

    # ------------------------------------------------------------------
    # Merge and persist
    # ------------------------------------------------------------------
    if cached_df is not None and not new_df.empty:
        df = pd.concat([cached_df, new_df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    elif cached_df is not None:
        df = cached_df
    else:
        df = new_df

    if df.empty:
        raise RuntimeError(
            f"No OHLCV data could be fetched for {config.SYMBOL}. "
            "Check your network connection and config dates."
        )

    # Trim to configured window (in case cached data predates START_DATE)
    df = df[df.index >= pd.Timestamp(config.START_DATE, tz="UTC")]

    logger.info("Saving %d rows to %s", len(df), cache_path)
    df.to_csv(cache_path)

    logger.info(
        "OHLCV ready: %d bars  [%s → %s]",
        len(df),
        df.index[0],
        df.index[-1],
    )
    return df
