"""
stat_test.py — Bootstrap significance test engine (Phase 3).

Implements a trade-level iid bootstrap to estimate confidence intervals on
all key backtest metrics and assess whether observed edge is statistically
significant.

Why iid bootstrap (not block bootstrap):
    With ~77 trades, block size ≈ √77 ≈ 9, leaving fewer than 9 blocks —
    making block bootstrap degenerate.  Simple iid resampling with replacement
    is the appropriate and honest choice at this sample size.

Exports
-------
run_bootstrap(trades_df, n_iter=1_000, seed=42)
    -> results : dict[str, tuple[float, float, float]]
                 {metric: (observed, ci_low_2.5%, ci_high_97.5%)}
       boot_pnl_curves : np.ndarray  shape (n_iter, n_trades + 1)
                         Cumulative PnL curve for each resample (starts at 0).
       ttest_result    : scipy.stats.TtestResult
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _equity_curve_from_pnl(pnl: np.ndarray) -> np.ndarray:
    """Return cumulative equity starting at INITIAL_CAPITAL."""
    return config.INITIAL_CAPITAL + np.concatenate([[0.0], np.cumsum(pnl)])


def _metrics_from_pnl(pnl: np.ndarray) -> dict[str, float]:
    """Compute all scalar metrics for one (possibly resampled) PnL vector."""
    n       = len(pnl)
    wins    = pnl[pnl > 0]
    losses  = pnl[pnl <= 0]
    win_rate    = len(wins) / n if n > 0 else float("nan")
    avg_win     = wins.mean()   if len(wins)   > 0 else float("nan")
    avg_loss    = losses.mean() if len(losses) > 0 else float("nan")
    expectancy  = (win_rate * avg_win + (1 - win_rate) * avg_loss
                   if not (np.isnan(avg_win) or np.isnan(avg_loss)) else float("nan"))

    equity = _equity_curve_from_pnl(pnl)

    # Drawdown on the synthetic equity curve
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd   = drawdown.min() * 100  # expressed as %

    # Annualised Sharpe from bar-level equity changes
    # Synthetic equity is indexed by trade, not bar, so we use a
    # per-trade Sharpe as a relative comparator (same for all resamples).
    returns = np.diff(equity) / equity[:-1]
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))
    else:
        sharpe = float("nan")

    return {
        "total_pnl_usd":   pnl.sum(),
        "win_rate":        win_rate,
        "avg_win_usd":     avg_win,
        "avg_loss_usd":    avg_loss,
        "expectancy_usd":  expectancy,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio":    sharpe,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_bootstrap(
    trades_df: pd.DataFrame,
    n_iter: int = 1_000,
    seed: int = 42,
) -> tuple[dict, np.ndarray, object]:
    """
    Run a trade-level iid bootstrap significance test.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of backtest.run_backtest — must contain 'pnl_usd' column.
    n_iter : int
        Number of bootstrap resamples (default 1 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Mapping of metric name → (observed, ci_low, ci_high).
    boot_pnl_curves : np.ndarray  shape (n_iter, n_trades + 1)
        Cumulative USD PnL for each resample (first element = 0).
    ttest_result : scipy.stats.TtestResult
        Two-sided one-sample t-test of per-trade PnL vs μ = 0.
    """
    if trades_df.empty:
        raise ValueError("trades_df is empty — run the backtest first.")

    pnl_obs  = trades_df["pnl_usd"].values.astype(float)
    n_trades = len(pnl_obs)
    rng      = np.random.default_rng(seed)

    # ── Observed metrics ─────────────────────────────────────────────────────
    observed = _metrics_from_pnl(pnl_obs)

    # ── Bootstrap ────────────────────────────────────────────────────────────
    boot_metrics: dict[str, list[float]] = {k: [] for k in observed}
    boot_pnl_curves = np.empty((n_iter, n_trades + 1), dtype=float)

    for i in range(n_iter):
        sample = rng.choice(pnl_obs, size=n_trades, replace=True)
        m      = _metrics_from_pnl(sample)
        for k, v in m.items():
            boot_metrics[k].append(v)
        boot_pnl_curves[i] = _equity_curve_from_pnl(sample)

    # ── Confidence intervals (2.5 – 97.5 percentile) ─────────────────────────
    results: dict[str, tuple[float, float, float]] = {}
    for metric, obs_val in observed.items():
        samples = np.array(boot_metrics[metric])
        # Drop NaNs before computing percentiles
        valid   = samples[~np.isnan(samples)]
        ci_lo   = float(np.percentile(valid, 2.5))  if len(valid) else float("nan")
        ci_hi   = float(np.percentile(valid, 97.5)) if len(valid) else float("nan")
        results[metric] = (obs_val, ci_lo, ci_hi)

    # ── Two-sided t-test: H₀ μ_trade_pnl = 0 ─────────────────────────────────
    ttest_result = stats.ttest_1samp(pnl_obs, popmean=0.0, alternative="two-sided")

    logger.info(
        "Bootstrap complete: %d iterations, %d trades.  "
        "Observed PnL: %.2f  95%% CI: [%.2f, %.2f]  p-value: %.4f",
        n_iter, n_trades,
        results["total_pnl_usd"][0],
        results["total_pnl_usd"][1],
        results["total_pnl_usd"][2],
        ttest_result.pvalue,
    )

    return results, boot_pnl_curves, ttest_result
