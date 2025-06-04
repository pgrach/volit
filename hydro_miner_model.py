"""
Fixed hydro_miner_model.py with proper cost accounting and utilization logic
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
    if "Generator Power (kW)" not in df.columns:
        raise KeyError("'Generator Power (kW)' column not found in hydro file")
    return df["Generator Power (kW)"].to_numpy()


# ───────────────────────────── STOCHASTIC FIT ────────────────────────────────

def _fit_stochastic(price: pd.Series, diff: pd.Series):
    """Fit a correlated GBM (price) and log-linear (difficulty) model."""
    common = price.index.intersection(diff.index)
    if len(common) < 30:
        raise ValueError("Price and difficulty series barely overlap")

    r_p = np.log(price[common]).diff().dropna()
    r_d = np.log(diff[common]).diff().dropna()

    mu_p, sigma_p = r_p.mean(), r_p.std()

    X = sm.add_constant(r_p)
    alpha, beta = sm.OLS(r_d, X).fit().params
    sigma_e = (r_d - (alpha + beta * r_p)).std()

    return mu_p, sigma_p, alpha, beta, sigma_e


# ──────────────────── POWER UTILIZATION CALCULATION ──────────────────────

def calculate_daily_effective_hashrate(
    fleet_size: int, 
    asic: Dict[str, Any], 
    hydro_kw_daily: np.ndarray
) -> np.ndarray:
    """
    Calculate daily effective hashrate based on fleet size and daily hydro availability.
    Preserves daily variation and seasonality.
    
    Returns
    -------
    np.ndarray
        Daily effective hashrate in TH/s
    """
    if fleet_size <= 0:
        return np.zeros_like(hydro_kw_daily)
    
    # Total fleet hashrate at 100% capacity
    fleet_hashrate_th = fleet_size * asic["hash_rate_th"]
    
    # Power required for the fleet at 100% capacity
    fleet_power_kw = fleet_hashrate_th * asic["watts_per_th"] / 1000
    
    if fleet_power_kw <= 0:  # Edge case guard
        return np.zeros_like(hydro_kw_daily)
    
    # Daily utilization factor (0 to 1)
    daily_utilization = np.minimum(hydro_kw_daily / fleet_power_kw, 1.0)
    
    # Daily effective hashrate
    return fleet_hashrate_th * daily_utilization


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
    annual_fixed_cost: float = 60000,  # Annual hydro operating cost
) -> Dict[str, Any]:
    """
    Fixed version with proper cost accounting and utilization logic.
    """
    
    # 1 ─ Load historical data & fit stochastic processes
    price = _load_price_series(price_csv)
    diff = _load_difficulty_series(diff_csv)
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
    
    # Network hashrate from difficulty (TH/s)
    # Difficulty = Network_Hashrate * 2^32 / 600 / 1e12
    network_hashrate = diff_paths * 600 * 1e12 / (2**32)
    
    # BTC per TH per day
    btc_per_th_day = (block_reward * BLOCKS_PER_DAY) / network_hashrate
    usd_per_th_day = price_paths * btc_per_th_day

    # 4 ─ Load hydro data and prepare daily pattern
    hydro_kw_hist = _load_hydro_series(hydro_xlsx)
    
    # Extend hydro data to cover the full horizon by repeating historical pattern
    # This preserves seasonality (winter peaks, summer lows)
    n_repeats = (horizon // len(hydro_kw_hist)) + 1
    hydro_kw_extended = np.tile(hydro_kw_hist, n_repeats)[:horizon]

    npv_paths_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for N in fleet_sizes:
        # Calculate CAPEX
        total_hashrate = N * asic["hash_rate_th"]
        capex = total_hashrate * asic["price_usd_per_th"]
        salvage = resale_pct * capex
        
        # Calculate daily effective hashrate based on hydro constraints
        daily_effective_th = calculate_daily_effective_hashrate(N, asic, hydro_kw_extended)
        
        # Daily revenue for all paths (paths x days)
        # Broadcasting: (paths, days) * (days,) = (paths, days)
        rev_paths = usd_per_th_day * daily_effective_th[np.newaxis, :]
        
        # Subtract daily operating costs
        daily_fixed_cost = annual_fixed_cost / 365
        rev_paths = rev_paths - daily_fixed_cost
        
        # Add salvage value at year 3
        salvage_idx = min(3 * 365 - 1, horizon - 1)
        if salvage_idx < rev_paths.shape[1]:
            rev_paths[:, salvage_idx] += salvage

        # Calculate NPV
        disc = (1 + discount_rate) ** (-np.arange(horizon) / 365)
        npv = -capex + (rev_paths * disc[np.newaxis, :]).sum(axis=1)
        npv_paths_all.append(npv)

        # Calculate payback period
        cum_cash = -capex + rev_paths.cumsum(axis=1)
        payback_days = np.full(n_paths, np.nan)
        for i in range(n_paths):
            positive_days = np.where(cum_cash[i] >= 0)[0]
            if len(positive_days) > 0:
                payback_days[i] = positive_days[0]

        pb = np.nanmedian(payback_days)
        row_pb = None if np.isnan(pb) else int(pb)
        
        # Calculate average utilization for reporting
        fleet_power_kw = N * asic["hash_rate_th"] * asic["watts_per_th"] / 1000
        if fleet_power_kw > 0:
            avg_utilization = (daily_effective_th.mean() / (N * asic["hash_rate_th"]))
        else:
            avg_utilization = 0.0

        rows.append({
            "ASICs": N,
            "Median NPV (USD)": int(np.median(npv)),
            "NPV p5": int(np.percentile(npv, 5)),
            "NPV p95": int(np.percentile(npv, 95)),
            "Probability NPV >0 (%)": round(100 * (npv > 0).mean(), 1),
            "Median Pay-back (days)": row_pb,
            "Avg Utilization (%)": round(avg_utilization * 100, 1),
            "CAPEX": int(capex),
            "Annual Fixed Cost": int(annual_fixed_cost),
        })

    summary = pd.DataFrame(rows)
    return {"summary": summary, "npv_paths_all": npv_paths_all}