"""
Cox-Ross-Rubinstein (CRR) binomial lattice for European and American options.

    dt = T / N
    u  = exp(sigma * sqrt(dt)),  d = 1/u
    p  = (exp((r - q) dt) - d) / (u - d)      risk-neutral up probability
    discount per step = exp(-r dt)

Backward induction with an early-exercise check at each node for American
options: V = max(continuation value, intrinsic exercise value). Vectorized
over nodes at each time step (not a Python loop over nodes) so this stays
usable at N in the thousands for the H1 convergence study.
"""
from __future__ import annotations

import numpy as np


def crr_binomial_price(S, K, r, q, sigma, T, N, option_type="call", exercise="european"):
    if N < 1:
        raise ValueError("N must be >= 1")
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside [0,1] for N={N}, "
            f"dt={dt:.6g}, sigma={sigma}, r={r}, q={q} -- arbitrage/step-size violation "
            f"(u={u:.4f}, d={d:.4f}, growth={growth:.4f})"
        )
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:i + 2] + (1.0 - p) * V[0:i + 1])
        if exercise == "american":
            j_i = np.arange(i + 1)
            S_i = S * (u ** j_i) * (d ** (i - j_i))
            if option_type == "call":
                exercise_val = np.maximum(S_i - K, 0.0)
            else:
                exercise_val = np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exercise_val)
        elif exercise != "european":
            raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")

    return float(V[0])
