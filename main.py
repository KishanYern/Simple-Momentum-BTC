"""
main.py — Orchestrator for the BTC Momentum Algorithm (Phases 1 & 2).

Run with:
    python main.py               # incremental data update + full pipeline
    python main.py --refresh     # force re-download of all OHLCV data

Outputs
-------
• Console / log: data fetch progress, indicator computation, backtest stats.
• plots/01_price_chart.png     : Price + EMAs + Bollinger Bands + trade markers.
• plots/02_equity_curve.png    : Equity curve + drawdown.
• plots/03_momentum_score.png  : Momentum score bar chart with regime filter overlay.
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import config
from src.data_fetcher import load_ohlcv
from src.indicators   import add_indicators, get_column_names
from src.scoring      import compute_scores
from src.backtest     import run_backtest, summary_stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    path = os.path.join(config.PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", path)
    plt.close(fig)


def plot_price_chart(df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    """
    Candlestick-style price chart with EMAs, Bollinger Bands, and trade
    entry / exit markers.  For large datasets we plot line-style OHLC to
    keep the figure readable.
    """
    cols = get_column_names(df)
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    idx = df.index

    # Price line (close)
    ax.plot(idx, df["close"], color="#c9d1d9", linewidth=0.6, label="BTC/USDT Close", zorder=2)

    # EMAs
    ax.plot(idx, df[cols["ema_fast"]], color="#58a6ff", linewidth=1.0,
            label=f"EMA-{config.EMA_FAST}", zorder=3)
    ax.plot(idx, df[cols["ema_slow"]], color="#f78166", linewidth=1.0,
            label=f"EMA-{config.EMA_SLOW} (signal)", zorder=3)
    ax.plot(idx, df[cols["trailing_stop_ema"]], color="#d2a8ff", linewidth=1.0,
            linestyle="--", label=f"EMA-{config.TRAILING_STOP_EMA_LEN} (stop)", zorder=3)

    # Bollinger Bands
    bb_u = df[cols["bb_upper"]]
    bb_l = df[cols["bb_lower"]]
    ax.fill_between(idx, bb_l, bb_u, color="#30363d", alpha=0.4, label="BB (20,2)", zorder=1)
    ax.plot(idx, bb_u, color="#8b949e", linewidth=0.4, linestyle="--")
    ax.plot(idx, bb_l, color="#8b949e", linewidth=0.4, linestyle="--")

    # Trade markers
    if not trades_df.empty:
        long_entries  = trades_df[trades_df["direction"] == "long"]
        short_entries = trades_df[trades_df["direction"] == "short"]
        tp_exits      = trades_df[trades_df["exit_reason"] == "tp"]
        sl_exits      = trades_df[trades_df["exit_reason"] == "trailing_stop"]

        # Entries
        ax.scatter(long_entries["entry_ts"],  long_entries["entry_price"],
                   marker="^", color="#3fb950", s=60, zorder=5, label="Long entry")
        ax.scatter(short_entries["entry_ts"], short_entries["entry_price"],
                   marker="v", color="#f85149", s=60, zorder=5, label="Short entry")
        # Exits
        ax.scatter(tp_exits["exit_ts"],  tp_exits["exit_price"],
                   marker="*", color="#e3b341", s=80, zorder=5, label="TP exit")
        ax.scatter(sl_exits["exit_ts"],  sl_exits["exit_price"],
                   marker="x", color="#d2a8ff", s=50, zorder=5, label="SL exit")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.set_title("BTC/USDT — Price Chart with Momentum Entries & Exits (1h)",
                 color="#c9d1d9", fontsize=13, pad=12)
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Price (USDT)", color="#8b949e")
    legend = ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
                       fontsize=8, loc="upper left")
    _save(fig, "01_price_chart.png")


def plot_equity_curve(equity_curve: pd.Series, stats: dict) -> None:
    """
    Two-panel plot: equity curve above, drawdown below.
    """
    peak     = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")

    # Equity
    ax1.plot(equity_curve.index, equity_curve.values, color="#3fb950", linewidth=1.2)
    ax1.axhline(config.INITIAL_CAPITAL, color="#8b949e", linewidth=0.6, linestyle="--")
    ax1.fill_between(equity_curve.index, config.INITIAL_CAPITAL, equity_curve.values,
                     where=(equity_curve.values >= config.INITIAL_CAPITAL),
                     color="#3fb950", alpha=0.12)
    ax1.fill_between(equity_curve.index, config.INITIAL_CAPITAL, equity_curve.values,
                     where=(equity_curve.values < config.INITIAL_CAPITAL),
                     color="#f85149", alpha=0.12)
    ax1.yaxis.set_tick_params(colors="#8b949e")

    # Stats annotation
    sr   = stats.get("sharpe_ratio", float("nan"))
    mdd  = stats.get("max_drawdown_pct", float("nan"))
    wr   = stats.get("win_rate", float("nan"))
    nt   = stats.get("n_trades", 0)
    anno = (
        f"Trades: {nt}  |  Win rate: {wr:.1%}  |  "
        f"Sharpe: {sr:.2f}  |  Max DD: {mdd:.1f}%  |  "
        f"Final equity: ${stats.get('final_equity', 0):,.0f}"
    )
    ax1.set_title(anno, color="#c9d1d9", fontsize=10, pad=8)
    ax1.set_ylabel("Equity (USD)", color="#8b949e")

    # Drawdown
    ax2.fill_between(equity_curve.index, drawdown.values, 0,
                     color="#f85149", alpha=0.5)
    ax2.plot(equity_curve.index, drawdown.values, color="#f85149", linewidth=0.7)
    ax2.set_ylabel("Drawdown %", color="#8b949e")
    ax2.yaxis.set_tick_params(colors="#8b949e")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    fig.suptitle("BTC Momentum Strategy — Equity Curve & Drawdown",
                 color="#c9d1d9", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "02_equity_curve.png")


def plot_momentum_score(df: pd.DataFrame) -> None:
    """
    Bar chart of momentum_score over time, coloured by direction,
    with a shaded overlay where the regime filter is active.
    """
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    idx    = df.index
    scores = df["momentum_score"]

    # Use sampled data if dataset is huge (plotting 17k bars is slow)
    sample_step = max(1, len(df) // 2000)
    idx_s    = idx[::sample_step]
    scores_s = scores.iloc[::sample_step]
    regime_s = df["regime_ok"].iloc[::sample_step]

    colors = np.where(scores_s > 0, "#3fb950", np.where(scores_s < 0, "#f85149", "#8b949e"))
    ax.bar(idx_s, scores_s, color=colors, width=pd.Timedelta(hours=sample_step * 1.5),
           linewidth=0, zorder=2)

    # Regime filter shading (grey when blocked)
    ax.fill_between(idx_s, -config.SCORE_MAX, config.SCORE_MAX,
                    where=~regime_s,
                    color="#30363d", alpha=0.5, zorder=1, label="Regime filter active")

    # Entry thresholds
    ax.axhline( config.ENTRY_THRESHOLD, color="#e3b341", linewidth=0.8,
               linestyle="--", label=f"Long threshold (+{config.ENTRY_THRESHOLD})")
    ax.axhline(-config.ENTRY_THRESHOLD, color="#d2a8ff", linewidth=0.8,
               linestyle="--", label=f"Short threshold (−{config.ENTRY_THRESHOLD})")
    ax.axhline(0, color="#8b949e", linewidth=0.5)

    ax.set_ylim(-config.SCORE_MAX - 0.5, config.SCORE_MAX + 0.5)
    ax.set_title("Momentum Score (1h bars)  — green=bullish, red=bearish, grey=regime-filtered",
                 color="#c9d1d9", fontsize=11, pad=8)
    ax.set_ylabel("Score", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    legend = ax.legend(facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#c9d1d9", fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "03_momentum_score.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def print_summary(stats: dict) -> None:
    """Pretty-print backtest summary to stdout."""
    divider = "─" * 50
    print(f"\n{divider}")
    print("  BTC MOMENTUM STRATEGY — BACKTEST SUMMARY")
    print(divider)
    items = [
        ("Total trades",       f"{stats.get('n_trades', 0)}"),
        ("  ↳ Longs",          f"{stats.get('n_longs', 0)}"),
        ("  ↳ Shorts",         f"{stats.get('n_shorts', 0)}"),
        ("Win rate",           f"{stats.get('win_rate', 0):.1%}"),
        ("Avg win  (USD)",     f"${stats.get('avg_win_usd',  0):,.2f}"),
        ("Avg loss (USD)",     f"${stats.get('avg_loss_usd', 0):,.2f}"),
        ("Total PnL (USD)",    f"${stats.get('total_pnl_usd', 0):,.2f}"),
        ("Initial equity",     f"${config.INITIAL_CAPITAL:,.2f}"),
        ("Final equity",       f"${stats.get('final_equity', 0):,.2f}"),
        ("Max drawdown",       f"{stats.get('max_drawdown_pct', 0):.2f}%"),
        ("Sharpe ratio",       f"{stats.get('sharpe_ratio', float('nan')):.2f}"),
        ("TP exits",           f"{stats.get('tp_exits', 0)}"),
        ("Trailing stop exits",f"{stats.get('sl_exits', 0)}"),
        ("End-of-data exits",  f"{stats.get('eod_exits', 0)}"),
    ]
    for label, value in items:
        print(f"  {label:<24} {value}")
    print(divider + "\n")


def main(force_refresh: bool = False) -> None:
    logger.info("━━━━ Phase 1: Data Ingestion ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df = load_ohlcv(force_refresh=force_refresh)

    logger.info("━━━━ Phase 1: Feature Generation ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df = add_indicators(df)

    logger.info("━━━━ Phase 2: Momentum Scoring ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df = compute_scores(df)

    logger.info("━━━━ Phase 2: Backtest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    trades_df, equity_curve = run_backtest(df)

    stats = summary_stats(trades_df, equity_curve)
    print_summary(stats)

    logger.info("━━━━ Generating Plots ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    plot_price_chart(df, trades_df)
    plot_equity_curve(equity_curve, stats)
    plot_momentum_score(df)

    logger.info("Done.  Charts saved to ./%s/", config.PLOT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Momentum Algorithm — Phases 1 & 2")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of all OHLCV data (ignores cache).",
    )
    args = parser.parse_args()
    main(force_refresh=args.refresh)
