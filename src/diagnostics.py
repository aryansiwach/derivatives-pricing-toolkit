"""
Per-model diagnostics:

- BS: put-call parity on quoted prices; analytical vs. finite-difference
  Greeks agreement.
- Binomial: convergence to BS in the European/no-dividend limit (see
  notebook 02 for the full H1 regression; this module provides the
  reusable finite-N check).
- Cross-model: binomial vs. market price gap (early-exercise premium) vs.
  BS vs. market price gap, same contracts, side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from black_scholes import bs_price, bs_greeks
from binomial import crr_binomial_price


def put_call_parity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Check C - P = S e^{-qT} - K e^{-rT} on matched call/put pairs sharing
    the same (ticker, expiry, strike). Requires columns: ticker, expiry,
    strike, option_type, mid, spot, q, r_cc, T_years."""
    calls = df[df.option_type == "call"][["ticker", "expiry", "strike", "mid", "spot", "q", "r_cc", "T_years"]]
    puts = df[df.option_type == "put"][["ticker", "expiry", "strike", "mid"]]
    merged = calls.merge(puts, on=["ticker", "expiry", "strike"], suffixes=("_call", "_put"))
    merged["lhs"] = merged["mid_call"] - merged["mid_put"]
    merged["rhs"] = merged["spot"] * np.exp(-merged["q"] * merged["T_years"]) - \
        merged["strike"] * np.exp(-merged["r_cc"] * merged["T_years"])
    merged["parity_gap"] = merged["lhs"] - merged["rhs"]
    return merged[["ticker", "expiry", "strike", "mid_call", "mid_put", "lhs", "rhs", "parity_gap"]]


def finite_difference_greeks(S, K, r, q, sigma, T, option_type="call", h_S=None, h_sigma=1e-4, h_T=1e-5, h_r=1e-5):
    """Central-difference cross-check for the analytical Greeks in
    black_scholes.py. This is the diagnostic cross-check; the primary
    computation is the closed-form analytical one."""
    h_S = h_S or S * 1e-4

    price_up = bs_price(S + h_S, K, r, q, sigma, T, option_type)
    price_dn = bs_price(S - h_S, K, r, q, sigma, T, option_type)
    price_mid = bs_price(S, K, r, q, sigma, T, option_type)
    delta_fd = (price_up - price_dn) / (2 * h_S)
    gamma_fd = (price_up - 2 * price_mid + price_dn) / (h_S ** 2)

    vega_fd = (bs_price(S, K, r, q, sigma + h_sigma, T, option_type) -
               bs_price(S, K, r, q, sigma - h_sigma, T, option_type)) / (2 * h_sigma)

    # black_scholes.bs_theta returns calendar-time theta (dV/dt, decay as
    # calendar time passes), not dV/dT_remaining -- since T_remaining
    # decreases as calendar time advances, dV/dt = -dV/dT_remaining, so the
    # finite-difference version needs the same sign flip to match.
    theta_fd = -(bs_price(S, K, r, q, sigma, T + h_T, option_type) -
                 bs_price(S, K, r, q, sigma, T - h_T, option_type)) / (2 * h_T)

    rho_fd = (bs_price(S, K, r + h_r, q, sigma, T, option_type) -
              bs_price(S, K, r - h_r, q, sigma, T, option_type)) / (2 * h_r)

    return {"delta": delta_fd, "gamma": gamma_fd, "vega": vega_fd, "theta": theta_fd, "rho": rho_fd}


def greeks_cross_check(S, K, r, q, sigma, T, option_type="call") -> dict:
    analytical = bs_greeks(S, K, r, q, sigma, T, option_type)
    fd = finite_difference_greeks(S, K, r, q, sigma, T, option_type)
    max_abs_diff = {g: abs(analytical[g] - fd[g]) for g in fd}
    return {"analytical": analytical, "finite_difference": fd, "max_abs_diff": max_abs_diff}


def early_exercise_premium(df: pd.DataFrame, N: int = 500) -> pd.DataFrame:
    """Cross-model comparison: for each American-style market quote, compute
    the BS European price and the CRR American price, and report both gaps
    to the market mid (H5). Requires columns: ticker, expiry, strike,
    option_type, mid, spot, q, r_cc, T_years."""
    rows = []
    for _, row in df.iterrows():
        S, K, r, q, T = row["spot"], row["strike"], row["r_cc"], row["q"], row["T_years"]
        opt = row["option_type"]
        bs = bs_price(S, K, r, q, row["iv_computed"], T, opt) if "iv_computed" in row and not pd.isna(row.get("iv_computed", np.nan)) else np.nan
        binom_euro = crr_binomial_price(S, K, r, q, row["iv_computed"], T, N, opt, "european") if not pd.isna(bs) else np.nan
        binom_amer = crr_binomial_price(S, K, r, q, row["iv_computed"], T, N, opt, "american") if not pd.isna(bs) else np.nan
        rows.append({
            "ticker": row["ticker"], "expiry": row["expiry"], "strike": K, "option_type": opt,
            "market_mid": row["mid"], "bs_european": bs,
            "binomial_european": binom_euro, "binomial_american": binom_amer,
            "market_minus_bs_gap": row["mid"] - bs if not pd.isna(bs) else np.nan,
            "market_minus_binomial_gap": row["mid"] - binom_amer if not pd.isna(binom_amer) else np.nan,
            "binomial_early_exercise_premium": binom_amer - binom_euro if not pd.isna(binom_amer) else np.nan,
        })
    return pd.DataFrame(rows)
