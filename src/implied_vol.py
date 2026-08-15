"""
Implied volatility solver: given a market price, back out the Black-Scholes
sigma that reproduces it. Primary method is Brent's method (bracketed,
guaranteed to converge if a root exists in the bracket); falls back to
Newton-Raphson (using the analytical BS vega) only if Brent's bracket is
invalid, and reports failures explicitly rather than silently returning a
placeholder value (unlike yfinance's own IV field -- see notebook 01 EDA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton

from black_scholes import bs_price, bs_vega

SIGMA_LO = 1e-4
SIGMA_HI = 5.0


def solve_implied_vol(price, S, K, r, q, T, option_type="call") -> dict:
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"iv": np.nan, "method": None, "success": False,
                "message": "invalid inputs (non-positive price/T/S/K)"}

    def f(sigma):
        return bs_price(S, K, r, q, sigma, T, option_type) - price

    f_lo, f_hi = f(SIGMA_LO), f(SIGMA_HI)
    if f_lo <= 0 <= f_hi:
        try:
            iv = brentq(f, SIGMA_LO, SIGMA_HI, xtol=1e-8, maxiter=200)
            return {"iv": float(iv), "method": "brent", "success": True, "message": None}
        except Exception:
            pass  # fall through to Newton

    # Brent bracket invalid (price outside the [SIGMA_LO, SIGMA_HI] range of
    # attainable BS prices). This almost always means the price itself is
    # infeasible (e.g. an arbitrage-violating quote, or a bad candidate
    # price thrown up mid-optimization by a calibration search), in which
    # case Newton has no hope of finding a valid answer either and will
    # just diverge to huge sigma (overflow) while burning time. Only
    # attempt the Newton fallback when the bracket violation is small
    # (a genuine near-boundary numerical case), and fail fast otherwise.
    near_boundary_tol = 1e-6 * max(abs(price), 1.0)
    if not (f_lo <= near_boundary_tol and f_hi >= -near_boundary_tol):
        reason = "price below min attainable (near-zero vol)" if f_lo > 0 else \
                 "price above max attainable (near-max vol)" if f_hi < 0 else \
                 "brent failed for an unknown reason within a valid bracket"
        return {"iv": np.nan, "method": None, "success": False, "message": reason}

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            iv = newton(f, x0=0.3, fprime=lambda s: bs_vega(S, K, r, q, s, T, option_type),
                        tol=1e-8, maxiter=50)
        if SIGMA_LO < iv < SIGMA_HI and np.isfinite(iv) and abs(f(iv)) < 1e-4:
            return {"iv": float(iv), "method": "newton_fallback", "success": True, "message": None}
        return {"iv": np.nan, "method": "newton_fallback", "success": False,
                "message": f"Newton converged outside valid range or residual too large (iv={iv!r})"}
    except Exception as e:
        return {"iv": np.nan, "method": None, "success": False, "message": str(e)}


def add_implied_vol(df: pd.DataFrame, price_col: str = "mid") -> tuple[pd.DataFrame, dict]:
    """Row-by-row application over a processed options dataframe (each row
    needs its own bracketed root solve, so this isn't vectorizable the way
    the batch Heston pricer is). Uses itertuples, not iterrows, since
    itertuples avoids rebuilding a Series per row and is materially faster
    at the row counts this pipeline sees (hundreds to low thousands).
    Requires columns: spot, strike, r_cc, q, T_years, option_type, <price_col>.
    Returns (df_with_iv_columns, summary_report)."""
    ivs, methods, successes, messages = [], [], [], []
    for row in df.itertuples(index=False):
        result = solve_implied_vol(
            price=getattr(row, price_col), S=row.spot, K=row.strike,
            r=row.r_cc, q=row.q, T=row.T_years, option_type=row.option_type,
        )
        ivs.append(result["iv"])
        methods.append(result["method"])
        successes.append(result["success"])
        messages.append(result["message"])

    out = df.copy()
    out["iv_computed"] = ivs
    out["iv_method"] = methods
    out["iv_success"] = successes
    out["iv_message"] = messages

    n = len(out)
    n_success = int(out["iv_success"].sum())
    report = {
        "n_total": n,
        "n_success": n_success,
        "n_failed": n - n_success,
        "pct_success": round(100.0 * n_success / n, 1) if n else 0.0,
        "n_brent": int((out["iv_method"] == "brent").sum()),
        "n_newton_fallback": int((out["iv_method"] == "newton_fallback").sum()),
    }
    return out, report
