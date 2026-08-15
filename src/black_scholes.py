"""
Black-Scholes-Merton closed-form European option pricing with a continuous
dividend yield q, plus analytical Greeks.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

Greeks below are the exact closed-form derivatives (see Hull, Options,
Futures, and Other Derivatives, "The Greek Letters"), not finite differences.
Finite-difference cross-checks live in diagnostics.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, q, sigma, T):
    S, K, r, q, sigma, T = map(np.asarray, (S, K, r, q, sigma, T))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, r, q, sigma, T, option_type="call"):
    """European option price. option_type in {'call', 'put'}."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(S, K, r, q, sigma, T, option_type="call"):
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_gamma(S, K, r, q, sigma, T, option_type="call"):
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, r, q, sigma, T, option_type="call"):
    """Per unit change in sigma (not per 1% vol point)."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, r, q, sigma, T, option_type="call"):
    """Per year (not per calendar day)."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        return term1 + term2 + term3
    elif option_type == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        return term1 + term2 + term3
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_rho(S, K, r, q, sigma, T, option_type="call"):
    """Per unit change in r (not per 1% rate point)."""
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_greeks(S, K, r, q, sigma, T, option_type="call") -> dict:
    return {
        "price": bs_price(S, K, r, q, sigma, T, option_type),
        "delta": bs_delta(S, K, r, q, sigma, T, option_type),
        "gamma": bs_gamma(S, K, r, q, sigma, T, option_type),
        "vega": bs_vega(S, K, r, q, sigma, T, option_type),
        "theta": bs_theta(S, K, r, q, sigma, T, option_type),
        "rho": bs_rho(S, K, r, q, sigma, T, option_type),
    }
