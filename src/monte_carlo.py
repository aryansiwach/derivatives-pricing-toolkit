"""
Risk-neutral Monte Carlo pricing of European options under GBM:

    S_T = S_0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0,1)

Three estimators, all at a fixed total path budget `n_paths` so their
standard errors are comparable at equal cost (H3):

  - plain:      naive MC, SE = std(payoff) / sqrt(n_paths)
  - antithetic: paired (Z, -Z) draws, averaged within each pair before
                taking the sample SE, using n_paths total simulated draws
                (n_paths/2 pairs)
  - control:    control variate using the terminal stock price S_T, whose
                risk-neutral mean E[S_T] = S_0 exp((r-q)T) is known exactly
                from the same GBM dynamics that underlie Black-Scholes.
                (Using the option's own BS price as its own control would
                be degenerate here since MC and BS estimate the identical
                quantity under identical GBM assumptions, so a control
                coefficient of 1 gives zero variance trivially, which
                proves nothing about the technique. S_T is the standard,
                non-degenerate choice: correlated with the payoff, cheap,
                and its mean is exact.)

American MC pricing (Longstaff-Schwartz) is out of scope for this toolkit;
this engine only prices European payoffs.
"""
from __future__ import annotations

import numpy as np


def _payoff(ST, K, option_type):
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def simulate_terminal(S, r, q, sigma, T, n_paths, seed=None, antithetic=False):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z_half = rng.standard_normal(half)
        Z = np.concatenate([z_half, -z_half])
    else:
        Z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S * np.exp(drift + diffusion)
    return ST


def mc_price_plain(S, K, r, q, sigma, T, n_paths, option_type="call", seed=None):
    ST = simulate_terminal(S, r, q, sigma, T, n_paths, seed=seed, antithetic=False)
    disc_payoff = np.exp(-r * T) * _payoff(ST, K, option_type)
    price = float(disc_payoff.mean())
    se = float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))
    return {"method": "plain", "price": price, "se": se, "n_paths": n_paths, "seed": seed}


def mc_price_antithetic(S, K, r, q, sigma, T, n_paths, option_type="call", seed=None):
    half = n_paths // 2
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(half)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diff_term = sigma * np.sqrt(T)
    ST_up = S * np.exp(drift + diff_term * Z)
    ST_dn = S * np.exp(drift - diff_term * Z)
    payoff_up = np.exp(-r * T) * _payoff(ST_up, K, option_type)
    payoff_dn = np.exp(-r * T) * _payoff(ST_dn, K, option_type)
    W = 0.5 * (payoff_up + payoff_dn)  # one value per antithetic pair
    price = float(W.mean())
    se = float(W.std(ddof=1) / np.sqrt(half))
    return {"method": "antithetic", "price": price, "se": se, "n_paths": 2 * half, "seed": seed}


def mc_price_control_variate(S, K, r, q, sigma, T, n_paths, option_type="call", seed=None):
    ST = simulate_terminal(S, r, q, sigma, T, n_paths, seed=seed, antithetic=False)
    Y = np.exp(-r * T) * _payoff(ST, K, option_type)          # target: discounted payoff
    C = np.exp(-r * T) * ST                                    # control: discounted S_T
    C_true_mean = S * np.exp(-q * T)                            # E[e^{-rT} S_T] exactly

    cov = np.cov(Y, C, ddof=1)[0, 1]
    var_C = np.var(C, ddof=1)
    c_star = cov / var_C if var_C > 0 else 0.0

    Y_cv = Y - c_star * (C - C_true_mean)
    price = float(Y_cv.mean())
    se = float(Y_cv.std(ddof=1) / np.sqrt(n_paths))
    return {
        "method": "control_variate", "price": price, "se": se, "n_paths": n_paths,
        "seed": seed, "c_star": float(c_star),
        "corr_Y_C": float(np.corrcoef(Y, C)[0, 1]),
    }


def mc_price(S, K, r, q, sigma, T, n_paths, option_type="call", seed=None, method="plain"):
    if method == "plain":
        return mc_price_plain(S, K, r, q, sigma, T, n_paths, option_type, seed)
    elif method == "antithetic":
        return mc_price_antithetic(S, K, r, q, sigma, T, n_paths, option_type, seed)
    elif method == "control":
        return mc_price_control_variate(S, K, r, q, sigma, T, n_paths, option_type, seed)
    raise ValueError(f"method must be 'plain', 'antithetic', or 'control', got {method!r}")
