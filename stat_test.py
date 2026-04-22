"""
stat_test.py (top-level runner) — Statistical significance analysis for the
BTC Momentum Strategy backtest.

Usage
-----
    # Run main.py first to populate the data cache, then:
    source venv/bin/activate && python3 stat_test.py

Outputs
-------
  plots/04_bootstrap_equity_curves.png
      1 000 synthetic equity curves (grey, transparent) overlaid with the
      95% CI band (blue shading), the bootstrap median (white), and the
      observed equity curve (green).  X-axis = trade number.

  plots/05_bootstrap_distributions.png
      2×2 histogram grid for Total PnL, Win Rate, Sharpe, and Max Drawdown.
      Each panel shows the bootstrap distribution with CI and observed lines.

  Stdout: CI table + t-test result.
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config
from src.data_fetcher import load_ohlcv
from src.indicators   import add_indicators
from src.scoring      import compute_scores
from src.backtest     import run_backtest
from src.stat_test    import run_bootstrap

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

DARK_BG   = "#0d1117"
GRID_COL  = "#21262d"
TEXT_COL  = "#c9d1d9"
MUTED_COL = "#8b949e"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"
PURPLE    = "#d2a8ff"
YELLOW    = "#e3b341"
RED       = "#f85149"


def _save(fig, filename: str) -> None:
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    path = os.path.join(config.PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: Bootstrap equity curves
# ---------------------------------------------------------------------------

def plot_bootstrap_equity_curves(
    boot_pnl_curves: np.ndarray,
    observed_pnl: "np.ndarray",
    n_trades: int,
) -> None:
    """
    Overlay all bootstrap equity curves with the 95% CI band and observed curve.

    X-axis is trade number (not time) because resampled trades are out of
    chronological order — time-indexing would be misleading.
    """
    # Convert cumulative PnL → cumulative USD gain (already in USD terms)
    # boot_pnl_curves shape: (n_iter, n_trades + 1), first column = 0
    # We plot cumulative PnL (gain/loss relative to start), not absolute equity.
    cum_pnl_boot = boot_pnl_curves - config.INITIAL_CAPITAL  # shift to 0-based

    trade_nums = np.arange(n_trades + 1)

    # Percentiles at each trade step
    ci_lo  = np.percentile(cum_pnl_boot, 2.5,  axis=0)
    ci_hi  = np.percentile(cum_pnl_boot, 97.5, axis=0)
    median = np.percentile(cum_pnl_boot, 50,   axis=0)

    obs_curve = np.concatenate([[0.0], np.cumsum(observed_pnl)])

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=MUTED_COL)
    ax.yaxis.set_tick_params(colors=MUTED_COL)
    ax.grid(color=GRID_COL, linewidth=0.4, zorder=0)

    # 1 000 individual bootstrap curves — very faint
    n_iter = cum_pnl_boot.shape[0]
    for i in range(n_iter):
        ax.plot(trade_nums, cum_pnl_boot[i], color="#8b949e", alpha=0.025,
                linewidth=0.4, zorder=1)

    # 95% CI band
    ax.fill_between(trade_nums, ci_lo, ci_hi,
                    color=BLUE, alpha=0.20, zorder=2, label="95% CI band")

    # Median bootstrap curve
    ax.plot(trade_nums, median, color=TEXT_COL, linewidth=1.5,
            linestyle="--", zorder=3, label="Bootstrap median")

    # Zero reference
    ax.axhline(0, color=MUTED_COL, linewidth=0.6, linestyle=":", zorder=2)

    # Observed equity curve
    ax.plot(trade_nums, obs_curve, color=GREEN, linewidth=2.2,
            zorder=4, label="Observed strategy")

    # Annotate final CI values
    ax.annotate(f"CI hi: ${ci_hi[-1]:+.0f}",
                xy=(n_trades, ci_hi[-1]), xytext=(n_trades - 6, ci_hi[-1] + 4),
                color=BLUE, fontsize=8)
    ax.annotate(f"CI lo: ${ci_lo[-1]:+.0f}",
                xy=(n_trades, ci_lo[-1]), xytext=(n_trades - 6, ci_lo[-1] - 8),
                color=BLUE, fontsize=8)
    ax.annotate(f"Observed: ${obs_curve[-1]:+.0f}",
                xy=(n_trades, obs_curve[-1]), xytext=(n_trades - 10, obs_curve[-1] + 4),
                color=GREEN, fontsize=8)

    ax.set_title(
        "Bootstrap Analysis — 1 000 Synthetic Equity Curves  (x-axis = trade #, not time)",
        color=TEXT_COL, fontsize=12, pad=10,
    )
    ax.set_xlabel("Trade number", color=MUTED_COL)
    ax.set_ylabel("Cumulative PnL (USD)", color=MUTED_COL)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "04_bootstrap_equity_curves.png")


# ---------------------------------------------------------------------------
# Plot 2: Metric distributions (2×2 histogram grid)
# ---------------------------------------------------------------------------

def plot_bootstrap_distributions(
    results: dict,
    boot_pnl_curves: np.ndarray,
    n_trades: int,
) -> None:
    """2×2 histogram panels for Total PnL, Win Rate, Sharpe, Max Drawdown."""

    # Re-derive per-iteration values from boot_pnl_curves for plotting
    # (results dict has observed/CI but not the raw bootstrap arrays —
    #  we need those for the histograms)
    pass  # handled by passing raw arrays below; see caller


def _plot_hist_panel(
    ax,
    samples: np.ndarray,
    observed: float,
    ci_lo: float,
    ci_hi: float,
    title: str,
    xlabel: str,
    fmt: str = ".2f",
) -> None:
    """Render a single histogram panel with CI and observed lines."""
    valid = samples[~np.isnan(samples)]

    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=MUTED_COL)
    ax.grid(color=GRID_COL, linewidth=0.4, axis="y", zorder=0)

    ax.hist(valid, bins=50, color=BLUE, alpha=0.55, edgecolor="none", zorder=2)

    ax.axvline(observed, color=GREEN,  linewidth=2.0, zorder=3,
               label=f"Observed: {observed:{fmt}}")
    ax.axvline(ci_lo,    color=YELLOW, linewidth=1.4, linestyle="--", zorder=3,
               label=f"2.5%: {ci_lo:{fmt}}")
    ax.axvline(ci_hi,    color=YELLOW, linewidth=1.4, linestyle="--", zorder=3,
               label=f"97.5%: {ci_hi:{fmt}}")

    ax.set_title(title, color=TEXT_COL, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, color=MUTED_COL, fontsize=8)
    ax.set_ylabel("Count", color=MUTED_COL, fontsize=8)
    ax.tick_params(colors=MUTED_COL, labelsize=7)

    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=7)


# ---------------------------------------------------------------------------
# CI table printer
# ---------------------------------------------------------------------------

def print_ci_table(results: dict, ttest) -> None:
    """Pretty-print the confidence interval table and t-test result."""
    divider = "─" * 72
    print(f"\n{divider}")
    print("  BOOTSTRAP SIGNIFICANCE TEST  —  BTC Momentum Strategy (N=1 000)")
    print(divider)
    print(f"  {'Metric':<24} {'Observed':>12} {'CI 2.5%':>12} {'CI 97.5%':>12}")
    print(f"  {'':─<24} {'':─>12} {'':─>12} {'':─>12}")

    fmt_map = {
        "total_pnl_usd":    ("Total PnL (USD)",       "${:,.2f}",  "${:,.2f}",  "${:,.2f}"),
        "win_rate":         ("Win rate",               "{:.1%}",    "{:.1%}",    "{:.1%}"),
        "avg_win_usd":      ("Avg win (USD)",          "${:,.2f}",  "${:,.2f}",  "${:,.2f}"),
        "avg_loss_usd":     ("Avg loss (USD)",         "${:,.2f}",  "${:,.2f}",  "${:,.2f}"),
        "expectancy_usd":   ("Expectancy/trade (USD)", "${:,.2f}",  "${:,.2f}",  "${:,.2f}"),
        "max_drawdown_pct": ("Max drawdown",           "{:.2f}%",   "{:.2f}%",   "{:.2f}%"),
        "sharpe_ratio":     ("Sharpe ratio",           "{:.3f}",    "{:.3f}",    "{:.3f}"),
    }

    for key, (label, f_obs, f_lo, f_hi) in fmt_map.items():
        obs, lo, hi = results[key]
        obs_s = f_obs.format(obs) if not (isinstance(obs, float) and np.isnan(obs)) else "N/A"
        lo_s  = f_lo.format(lo)   if not (isinstance(lo,  float) and np.isnan(lo))  else "N/A"
        hi_s  = f_hi.format(hi)   if not (isinstance(hi,  float) and np.isnan(hi))  else "N/A"
        print(f"  {label:<24} {obs_s:>12} {lo_s:>12} {hi_s:>12}")

    print(divider)
    print(f"\n  T-TEST  (H₀: mean per-trade PnL = 0, two-sided)")
    print(f"  t-statistic : {ttest.statistic:.4f}")
    print(f"  p-value     : {ttest.pvalue:.4f}  "
          f"({'SIGNIFICANT at α=0.05 ✓' if ttest.pvalue < 0.05 else 'NOT significant at α=0.05 — insufficient evidence of edge'})")
    print(f"\n{divider}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("━━━━ Loading data & running backtest ━━━━━━━━━━━━━━━━━━━━━━━━━")
    df          = load_ohlcv(force_refresh=False)
    df          = add_indicators(df)
    df          = compute_scores(df)
    trades_df, equity_curve = run_backtest(df)

    if trades_df.empty:
        logger.error("No trades — cannot run bootstrap.  Run main.py first.")
        sys.exit(1)

    logger.info("━━━━ Running bootstrap (N=1 000) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    results, boot_pnl_curves, ttest = run_bootstrap(trades_df, n_iter=1_000, seed=42)

    print_ci_table(results, ttest)

    # ── Collect raw bootstrap arrays for histogram panels ─────────────────
    from src.stat_test import _metrics_from_pnl  # internal helper
    pnl_obs  = trades_df["pnl_usd"].values.astype(float)
    n_trades = len(pnl_obs)
    rng      = np.random.default_rng(42)

    boot_total_pnl  = []
    boot_win_rate   = []
    boot_sharpe     = []
    boot_max_dd     = []

    for i in range(boot_pnl_curves.shape[0]):
        # Re-derive from the already-computed curves (avoids re-running bootstrap)
        cum = boot_pnl_curves[i]           # shape (n_trades+1,)
        pnl = np.diff(cum)                 # per-trade PnL for this resample
        m   = _metrics_from_pnl(pnl)
        boot_total_pnl.append(m["total_pnl_usd"])
        boot_win_rate.append(m["win_rate"])
        boot_sharpe.append(m["sharpe_ratio"])
        boot_max_dd.append(m["max_drawdown_pct"])

    logger.info("━━━━ Generating plots ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Plot 1: equity curve fan
    plot_bootstrap_equity_curves(boot_pnl_curves, pnl_obs, n_trades)

    # Plot 2: distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Bootstrap Metric Distributions — BTC Momentum Strategy (N=1 000 resamples)",
        color=TEXT_COL, fontsize=13, y=1.01,
    )

    panels = [
        (axes[0, 0], np.array(boot_total_pnl), results["total_pnl_usd"],
         "Total PnL (USD)", "USD", ".2f"),
        (axes[0, 1], np.array(boot_win_rate) * 100,
         (results["win_rate"][0]*100, results["win_rate"][1]*100, results["win_rate"][2]*100),
         "Win Rate", "%", ".1f"),
        (axes[1, 0], np.array(boot_sharpe), results["sharpe_ratio"],
         "Sharpe Ratio", "", ".3f"),
        (axes[1, 1], np.array(boot_max_dd),  results["max_drawdown_pct"],
         "Max Drawdown", "%", ".2f"),
    ]

    for ax, samples, (obs, lo, hi), title, xlabel, fmt in panels:
        _plot_hist_panel(ax, samples, obs, lo, hi, title, xlabel, fmt)

    fig.tight_layout()
    _save(fig, "05_bootstrap_distributions.png")

    logger.info("Done.  Plots saved to ./%s/", config.PLOT_DIR)


if __name__ == "__main__":
    main()
