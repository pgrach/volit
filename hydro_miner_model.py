"""
Pure-Python engine invoked by the Streamlit UI.
Keeps all heavy lifting out of app.py so you can unit-test in isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ──────────────────────────────── I/O HELPERS ────────────────────────────────

def _load_price_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["Close"].sort_index()

def _load_difficulty_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["Difficulty"].sort_index()

def _load_hydro_series(path: Path) -> np.ndarray:
    df = pd.read_excel(path)
    # ► assert the expected column exists
    if "Generator Power (kW)" not in df.columns:
        raise KeyError("'Generator Power (kW)' column not found in hydro file")
    return df["Generator Power (kW)"].to_numpy()


# ───────────────────────────── STOCHASTIC FIT ────────────────────────────────

def _fit_stochastic(price: pd.Series, diff: pd.Series):
    """Fit a correlated GBM (price) and log-linear (difficulty) model."""
    common = price.index.intersection(diff.index)
    if len(common) < 30:               # ► sanity-check overlap
        raise ValueError("Price and difficulty series barely overlap")

    r_p = np.log(price[common]).diff().dropna()
    r_d = np.log(diff[common]).diff().dropna()

    mu_p, sigma_p = r_p.mean(), r_p.std()

    X = sm.add_constant(r_p)
    alpha, beta = sm.OLS(r_d, X).fit().params
    sigma_e = (r_d - (alpha + beta * r_p)).std()

    return mu_p, sigma_p, alpha, beta, sigma_e


# ──────────────────────────  MAIN MONTE-CARLO ENTRY  ─────────────────────────

def run_monte_carlo(
    *,
    asic: Dict[str, Any],
    discount_rate: float,
    resale_pct: float,
    n_paths: int,
    horizon_years: int,
    price_csv: Path,
    diff_csv: Path,
    hydro_xlsx: Path,
    fleet_sizes: Sequence[int],
) -> Dict[str, Any]:
    """
    Returns
    -------
    dict with
      • 'summary' – DataFrame of median NPV, pay-back, etc.
      • 'npv_paths_all' – list[ ndarray ], one array of NPVs per fleet size
    """

    # 1 ─ Load historical data & fit stochastic processes
    price = _load_price_series(price_csv)
    diff  = _load_difficulty_series(diff_csv)
    mu_p, sigma_p, alpha, beta, sigma_e = _fit_stochastic(price, diff)

    # 2 ─ Simulate correlated paths
    horizon = horizon_years * 365
    rng = np.random.default_rng(42)

    Zp = rng.standard_normal((n_paths, horizon))
    Ze = rng.standard_normal((n_paths, horizon))

    r_p = (mu_p - 0.5 * sigma_p**2) + sigma_p * Zp
    price_paths = np.exp(np.log(price.iloc[-1]) + np.cumsum(r_p, axis=1))

    r_d = alpha + beta * r_p + sigma_e * Ze
    diff_paths = np.exp(np.log(diff.iloc[-1]) + np.cumsum(r_d, axis=1))

    # 3 ─ Daily BTC revenue per TH
    BLOCKS_PER_DAY = 144
    BR_PRE, BR_POST = 3.125, 1.5625
    halving_day = (pd.Timestamp("2028-04-22") - pd.Timestamp.today()).days
    block_reward = np.where(np.arange(horizon) < halving_day, BR_PRE, BR_POST)
    btc_per_th_day = block_reward * BLOCKS_PER_DAY / diff_paths
    usd_per_th_day = price_paths * btc_per_th_day                   # (paths,days)

    # 4 ─ Hydro-constrained TH/s
    hydro_kw_hist = _load_hydro_series(hydro_xlsx)
    th_per_kw = 1000 / asic["watts_per_th"]
    hydro_kw = np.resize(hydro_kw_hist, horizon)                    # repeat history

    npv_paths_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for N in fleet_sizes:
        cap_kw  = N * asic["hash_rate_th"] * asic["watts_per_th"] / 1000
        capex   = N * asic["hash_rate_th"] * asic["price_usd_per_th"]
        salvage = resale_pct * capex

        utilised_kw = np.minimum(hydro_kw, cap_kw)
        th_day = utilised_kw * th_per_kw                            # (days,)

        rev_paths = usd_per_th_day * th_day                         # (paths,days)
        rev_paths[:, 3 * 365] += salvage                            # resale at t = 3 y

        disc = (1 + discount_rate) ** (-np.arange(horizon) / 365)
        npv  = -capex + (rev_paths * disc).sum(axis=1)
        npv_paths_all.append(npv)

        cum_cash = -capex + rev_paths.cumsum(axis=1)
        payback_days = np.nanmin(np.where(cum_cash >= 0, np.arange(horizon), np.nan), axis=1)

        rows.append(
            {
                "ASICs": N,
                "Median NPV (USD)": int(np.median(npv)),
                "NPV p5":           int(np.percentile(npv, 5)),
                "NPV p95":          int(np.percentile(npv, 95)),
                "Probability NPV >0 (%)": round(100 * (npv > 0).mean(), 1),
                "Median Pay-back (days)": int(np.nanmedian(payback_days)),
            }
        )

    summary = pd.DataFrame(rows)
    return {"summary": summary, "npv_paths_all": npv_paths_all}
