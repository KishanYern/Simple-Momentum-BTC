"""
backtest.py — Event-driven backtest engine (Phase 2, Step 2.3).

Implements a simple bar-by-bar simulation with the following rules:

Entry
─────
• Long  : momentum_score >= +ENTRY_THRESHOLD, no open position → BUY  at open of next bar.
• Short : momentum_score <= −ENTRY_THRESHOLD, no open position → SELL at open of next bar.
• Only one position at a time (no pyramiding).

Take-Profit (TP)
────────────────
• Long  TP = entry_price + TP_ATR_MULT × ATR_at_entry
• Short TP = entry_price − TP_ATR_MULT × ATR_at_entry

Trailing Stop-Loss
──────────────────
• Long  stop = 20-EMA of the *current* bar (checked every bar while in trade).
  Exit if bar's low  <= EMA_20.  Exit price = EMA_20 value.
• Short stop = 20-EMA of the *current* bar.
  Exit if bar's high >= EMA_20.  Exit price = EMA_20 value.

Exit priority within a bar: TP checked against high/low first; if not
triggered, trailing stop is checked.

Position sizing
───────────────
Each trade risks POSITION_SIZE_PCT of current equity.  Notional value
per trade = equity × POSITION_SIZE_PCT / entry_price (units of BTC).
PnL is computed as (exit_price − entry_price) × units for longs,
and (entry_price − exit_price) × units for shorts.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

import config
from src.indicators import get_column_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    direction:   Literal["long", "short"]
    entry_ts:    pd.Timestamp
    entry_price: float
    tp_price:    float
    atr_at_entry:float
    units:       float           # BTC quantity
    exit_ts:     pd.Timestamp    = field(default=None)
    exit_price:  float           = field(default=None)
    exit_reason: str             = field(default=None)  # "tp" | "trailing_stop" | "end_of_data"
    pnl_usd:     float           = field(default=None)

    def close(
        self,
        ts: pd.Timestamp,
        price: float,
        reason: str,
    ) -> None:
        self.exit_ts     = ts
        self.exit_price  = price
        self.exit_reason = reason
        if self.direction == "long":
            self.pnl_usd = (price - self.entry_price) * self.units
        else:
            self.pnl_usd = (self.entry_price - price) * self.units


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate the momentum strategy on *df* bar by bar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV + indicator + score columns
        (output of scoring.compute_scores).

    Returns
    -------
    trades_df : pd.DataFrame
        One row per completed trade with entry/exit metadata and PnL.
    equity_curve : pd.Series
        Equity value at the *close* of every bar (index = UTC timestamps).
    """
    cols       = get_column_names(df)
    ema_col    = cols["ema_slow"]
    atr_col    = cols["atr"]

    equity     = config.INITIAL_CAPITAL
    open_trade: Trade | None = None
    completed:  list[Trade]  = []
    equity_values: list[float] = []

    # We iterate over bars[1:] so we can use the signal on bar i to enter
    # at the open of bar i+1 (no look-ahead bias).
    bars = df.reset_index()   # gives us integer positional indexing

    for i in range(len(bars)):
        row = bars.iloc[i]
        ts  = row["timestamp"]

        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        ema_val     = row[ema_col]
        atr_val     = row[atr_col]

        # ── Manage open position ──────────────────────────────────────────
        if open_trade is not None:
            tp     = open_trade.tp_price
            exited = False

            if open_trade.direction == "long":
                # Check TP (high reaches tp)
                if h >= tp:
                    open_trade.close(ts, tp, "tp")
                    exited = True
                # Check trailing stop (low touches or crosses EMA_20)
                elif l <= ema_val:
                    open_trade.close(ts, ema_val, "trailing_stop")
                    exited = True
            else:  # short
                # Check TP (low reaches tp)
                if l <= tp:
                    open_trade.close(ts, tp, "tp")
                    exited = True
                # Check trailing stop (high touches or crosses EMA_20)
                elif h >= ema_val:
                    open_trade.close(ts, ema_val, "trailing_stop")
                    exited = True

            if exited:
                equity += open_trade.pnl_usd
                completed.append(open_trade)
                logger.debug(
                    "[%s] %s trade closed via %s  PnL=%.2f  equity=%.2f",
                    ts, open_trade.direction, open_trade.exit_reason,
                    open_trade.pnl_usd, equity,
                )
                open_trade = None

        # ── Entry logic (signal from this bar → enter at open of NEXT bar) ─
        # We look ahead by one to simulate entering at the next bar's open.
        # The signal is evaluated *after* managing any open trade so we never
        # enter and exit on the same bar.
        if open_trade is None and i + 1 < len(bars):
            next_row        = bars.iloc[i + 1]
            next_open       = next_row["open"]
            next_ts         = next_row["timestamp"]
            long_signal     = bool(row.get("long_signal",  False))
            short_signal    = bool(row.get("short_signal", False))

            if long_signal or short_signal:
                direction = "long" if long_signal else "short"
                entry_p   = next_open
                # Use current bar's ATR for stop / TP sizing
                tp_price  = (
                    entry_p + config.TP_ATR_MULT * atr_val
                    if direction == "long"
                    else entry_p - config.TP_ATR_MULT * atr_val
                )
                units = (equity * config.POSITION_SIZE_PCT) / entry_p

                open_trade = Trade(
                    direction    = direction,
                    entry_ts     = next_ts,
                    entry_price  = entry_p,
                    tp_price     = tp_price,
                    atr_at_entry = atr_val,
                    units        = units,
                )
                logger.debug(
                    "[%s] ENTER %s @ %.2f  TP=%.2f  units=%.6f",
                    next_ts, direction, entry_p, tp_price, units,
                )

        equity_values.append(equity)

    # Close any position still open at end of data
    if open_trade is not None:
        last_row = bars.iloc[-1]
        open_trade.close(last_row["timestamp"], last_row["close"], "end_of_data")
        equity += open_trade.pnl_usd
        completed.append(open_trade)
        equity_values[-1] = equity

    # ------------------------------------------------------------------
    # Build output DataFrames
    # ------------------------------------------------------------------
    equity_curve = pd.Series(equity_values, index=df.index, name="equity")

    if not completed:
        logger.warning("No trades were completed during the backtest period.")
        trades_df = pd.DataFrame(columns=[
            "direction", "entry_ts", "entry_price", "tp_price",
            "atr_at_entry", "units", "exit_ts", "exit_price",
            "exit_reason", "pnl_usd",
        ])
        return trades_df, equity_curve

    trades_df = pd.DataFrame([t.__dict__ for t in completed])

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    n_total  = len(trades_df)
    n_wins   = (trades_df["pnl_usd"] > 0).sum()
    win_rate = n_wins / n_total if n_total else 0.0
    avg_win  = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"].mean()
    avg_loss = trades_df.loc[trades_df["pnl_usd"] <= 0, "pnl_usd"].mean()
    total_pnl = trades_df["pnl_usd"].sum()

    peak    = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd  = drawdown.min()

    # Annualised Sharpe (assume 365 × 24 hourly bars per year)
    bars_per_year = 365 * 24
    returns       = equity_curve.pct_change().dropna()
    sharpe        = (
        (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        if returns.std() > 0 else np.nan
    )

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("Total trades    : %d", n_total)
    logger.info("Win rate        : %.1f%%", win_rate * 100)
    logger.info("Avg win (USD)   : %.2f", avg_win  if not np.isnan(avg_win)  else 0)
    logger.info("Avg loss (USD)  : %.2f", avg_loss if not np.isnan(avg_loss) else 0)
    logger.info("Total PnL (USD) : %.2f", total_pnl)
    logger.info("Final equity    : %.2f", equity)
    logger.info("Max drawdown    : %.2f%%", max_dd * 100)
    logger.info("Sharpe ratio    : %.2f", sharpe if not np.isnan(sharpe) else 0)
    logger.info("=" * 60)

    return trades_df, equity_curve


def summary_stats(trades_df: pd.DataFrame, equity_curve: pd.Series) -> dict:
    """
    Return a dictionary of key performance metrics for programmatic access.
    """
    n_total  = len(trades_df)
    if n_total == 0:
        return {}

    n_wins   = (trades_df["pnl_usd"] > 0).sum()
    peak     = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    returns  = equity_curve.pct_change().dropna()
    bars_per_year = 365 * 24
    sharpe   = (
        (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        if returns.std() > 0 else np.nan
    )

    return {
        "n_trades":         n_total,
        "n_longs":          (trades_df["direction"] == "long").sum(),
        "n_shorts":         (trades_df["direction"] == "short").sum(),
        "win_rate":         n_wins / n_total,
        "avg_win_usd":      trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"].mean(),
        "avg_loss_usd":     trades_df.loc[trades_df["pnl_usd"] <= 0, "pnl_usd"].mean(),
        "total_pnl_usd":    trades_df["pnl_usd"].sum(),
        "final_equity":     equity_curve.iloc[-1],
        "max_drawdown_pct": drawdown.min() * 100,
        "sharpe_ratio":     sharpe,
        "tp_exits":         (trades_df["exit_reason"] == "tp").sum(),
        "sl_exits":         (trades_df["exit_reason"] == "trailing_stop").sum(),
        "eod_exits":        (trades_df["exit_reason"] == "end_of_data").sum(),
    }
