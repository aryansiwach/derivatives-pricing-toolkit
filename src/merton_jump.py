"""
Merton (1976) jump-diffusion closed-form price: an infinite sum of
Black-Scholes terms weighted by Poisson jump probabilities.

    C = sum_{n=0}^inf [ e^{-lambda' T} (lambda' T)^n / n! ] * BS(S, K, r_n, sigma_n, T)

    lambda' = lambda * (1 + k),  k = exp(mu_J + sigma_J^2/2) - 1
    r_n     = r - lambda*k + n * ln(1+k) / T
    sigma_n = sqrt(sigma^2 + n * sigma_J^2 / T)

Truncated at a documented number of terms with a stated truncation-error
bound: each term is weighted by a Poisson(lambda' T) pmf, so truncating at
N terms leaves a tail mass of 1 - CDF(N; lambda' T), which bounds the
truncation error since each BS term is itself bounded by max(S, K).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

from black_scholes import bs_price


def merton_jump_price(S, K, r, q, sigma, T, lam, mu_j, sigma_j, option_type="call",
                       max_terms=50, trunc_tol=1e-10):
    """Merton jump-diffusion price with dividend yield q folded into the
    base drift, truncated once the remaining Poisson tail mass < trunc_tol."""
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    lam_prime = lam * (1.0 + k)

    total = 0.0
    for n in range(max_terms):
        weight = poisson.pmf(n, lam_prime * T)
        if weight < 1e-300:
            continue
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        r_n = r - lam * k + n * np.log(1.0 + k) / T
        term = weight * bs_price(S, K, r_n, q, sigma_n, T, option_type)
        total += term

        tail_mass = 1.0 - poisson.cdf(n, lam_prime * T)
        if tail_mass < trunc_tol and n >= 5:
            break

    n_terms_used = n + 1
    tail_mass_remaining = 1.0 - poisson.cdf(n, lam_prime * T)
    return total, {"n_terms_used": n_terms_used, "tail_mass_remaining": tail_mass_remaining}


def merton_jump_price_value(S, K, r, q, sigma, T, lam, mu_j, sigma_j, option_type="call"):
    price, _ = merton_jump_price(S, K, r, q, sigma, T, lam, mu_j, sigma_j, option_type)
    return price


def merton_jump_price_batch(K_array, S, r, q, T, sigma, lam, mu_j, sigma_j, option_type="call",
                             max_terms=50, trunc_tol=1e-10):
    """Vectorized pricing across strikes for a single maturity T -- for a
    fixed T, r_n and sigma_n depend only on the jump-count index n, not on
    K, so the sum over n can broadcast across the whole strike array
    instead of re-running the sum once per strike (this is what makes
    calibration tractable, same motivation as heston_price_batch)."""
    K_array = np.asarray(K_array, dtype=float)
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    lam_prime = lam * (1.0 + k)

    total = np.zeros_like(K_array)
    for n in range(max_terms):
        weight = poisson.pmf(n, lam_prime * T)
        if weight < 1e-300:
            continue
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        r_n = r - lam * k + n * np.log(1.0 + k) / T
        total += weight * bs_price(S, K_array, r_n, q, sigma_n, T, option_type)

        tail_mass = 1.0 - poisson.cdf(n, lam_prime * T)
        if tail_mass < trunc_tol and n >= 5:
            break
    return total


def _single_merton_fit(market_df, spot, r, q, x0, bounds, maxiter):
    from implied_vol import solve_implied_vol

    groups = {T: g for T, g in market_df.groupby("T_years")}

    def objective(params):
        sigma, lam, mu_j, sigma_j = params
        sq_errors = []
        for T, g in groups.items():
            for opt, sub in g.groupby("option_type"):
                model_prices = merton_jump_price_batch(
                    sub["strike"].values, spot, r, q, T, sigma, lam, mu_j, sigma_j, opt
                )
                for K, price, mkt_iv in zip(sub["strike"].values, model_prices, sub["iv_computed"].values):
                    iv_result = solve_implied_vol(price, spot, K, r, q, T, opt)
                    model_iv = iv_result["iv"] if iv_result["success"] else mkt_iv  # neutral fallback
                    sq_errors.append((model_iv - mkt_iv) ** 2)
        return float(np.mean(sq_errors)) if sq_errors else np.inf

    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-10})

    sigma, lam, mu_j, sigma_j = result.x
    return {
        "sigma": float(sigma), "lam": float(lam), "mu_j": float(mu_j), "sigma_j": float(sigma_j),
        "success": bool(result.success), "message": str(result.message),
        "n_iter": int(result.nit),
        "final_rmse_iv": float(np.sqrt(result.fun)) if np.isfinite(result.fun) else float("nan"),
        "x0_used": list(x0),
    }


def calibrate_merton(market_df, spot, r, q, bounds=None, x0=None, n_starts=1, random_seed=42, maxiter=200):
    """Calibrate (sigma, lambda, mu_J, sigma_J) to minimize mean squared IV
    error (model price -> model-implied vol vs. market-implied vol),
    matching the metric convention used elsewhere (IV RMSE, not price RMSE,
    since price errors are dominated by ITM contracts).

    market_df must have columns: strike, T_years, option_type, iv_computed.

    With n_starts > 1, runs from n_starts different initial points (the
    given/default x0 plus (n_starts-1) randomized draws from a reasonable
    domain) and keeps the best by final RMSE, since L-BFGS-B is a local
    optimizer and a single start cannot rule out a local minimum. All
    per-start RMSEs are returned in `all_start_rmse` so the spread across
    starts is visible rather than hidden behind just the best one.
    """
    bounds = bounds or [(0.01, 1.5), (0.001, 5.0), (-0.5, 0.5), (0.001, 1.0)]
    base_x0 = x0 or [0.20, 0.5, -0.1, 0.15]

    rng = np.random.default_rng(random_seed)
    starts = [base_x0]
    for _ in range(n_starts - 1):
        starts.append([
            rng.uniform(0.05, 0.5),   # sigma
            rng.uniform(0.05, 2.0),   # lam
            rng.uniform(-0.4, 0.1),   # mu_j
            rng.uniform(0.05, 0.5),   # sigma_j
        ])

    fits = [_single_merton_fit(market_df, spot, r, q, s, bounds, maxiter) for s in starts]
    finite_fits = [f for f in fits if np.isfinite(f["final_rmse_iv"])]
    best = min(finite_fits or fits, key=lambda f: f["final_rmse_iv"] if np.isfinite(f["final_rmse_iv"]) else np.inf)

    best = dict(best)
    best["n_starts"] = n_starts
    best["all_start_rmse"] = [f["final_rmse_iv"] for f in fits]
    return best
