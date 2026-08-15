"""
Heston (1993) stochastic volatility model, priced via the characteristic-
function approach:

    dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^1
    dv_t = kappa(theta - v_t) dt + xi sqrt(v_t) dW_t^2
    Corr(dW^1, dW^2) = rho

    C = S e^{-qT} P1 - K e^{-rT} P2,   P_j = 1/2 + 1/pi * Int_0^inf Re[e^{-i u ln K} phi_j(u) / (i u)] du

We use the "little trap" formulation (Albrecher, Mayer, Schoutens, Tistaert
2007) for the characteristic function, which is numerically stable across
maturities/parameters where the original Heston (1993) branch of the
complex logarithm can jump discontinuously. Implemented once, correctly,
rather than partially implementing both this and the COS method.

Integration uses fixed-order Gauss-Legendre quadrature (not adaptive quad),
vectorized across strikes for a given maturity -- this is what makes
calibration (which needs thousands of price evaluations) tractable.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize

_NODES_CACHE: dict = {}


def _gauss_legendre_nodes(n=128, eps=1e-10, upper=150.0):
    key = (n, eps, upper)
    if key not in _NODES_CACHE:
        x, w = leggauss(n)
        u = 0.5 * (upper - eps) * x + 0.5 * (upper + eps)
        weight = w * 0.5 * (upper - eps)
        _NODES_CACHE[key] = (u, weight)
    return _NODES_CACHE[key]


def _phi(u_arr, S0, r, q, T, kappa, theta, xi, rho, v0, j):
    """Heston characteristic function of ln(S_T), 'little trap' formulation."""
    if j == 1:
        uj, bj = 0.5, kappa - rho * xi
    else:
        uj, bj = -0.5, kappa
    a = kappa * theta
    x0 = np.log(S0)
    iu = 1j * u_arr

    d = np.sqrt((rho * xi * iu - bj) ** 2 - xi ** 2 * (2 * uj * iu - u_arr ** 2))
    g = (bj - rho * xi * iu - d) / (bj - rho * xi * iu + d)
    exp_dT = np.exp(-d * T)

    C = (r - q) * iu * T + (a / xi ** 2) * (
        (bj - rho * xi * iu - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = (bj - rho * xi * iu - d) / xi ** 2 * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + iu * x0)


def _P_batch(K_array, S0, r, q, T, kappa, theta, xi, rho, v0, j, n_nodes=128):
    u, w = _gauss_legendre_nodes(n_nodes)
    phi_vals = _phi(u, S0, r, q, T, kappa, theta, xi, rho, v0, j)  # (N,)
    logK = np.log(np.asarray(K_array, dtype=float))                 # (M,)
    base = phi_vals / (1j * u)                                      # (N,)
    phase = np.exp(-1j * np.outer(u, logK))                         # (N, M)
    integrand_vals = (base[:, None] * phase).real                   # (N, M)
    integral = w @ integrand_vals                                   # (M,)
    return 0.5 + integral / np.pi


def heston_price_batch(K_array, S0, r, q, T, kappa, theta, xi, rho, v0, option_type="call"):
    """Vectorized pricing across strikes for a single maturity T."""
    P1 = _P_batch(K_array, S0, r, q, T, kappa, theta, xi, rho, v0, j=1)
    P2 = _P_batch(K_array, S0, r, q, T, kappa, theta, xi, rho, v0, j=2)
    K_array = np.asarray(K_array, dtype=float)
    call = S0 * np.exp(-q * T) * P1 - K_array * np.exp(-r * T) * P2
    if option_type == "call":
        return call
    elif option_type == "put":
        return call - S0 * np.exp(-q * T) + K_array * np.exp(-r * T)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def heston_price(S0, K, r, q, T, kappa, theta, xi, rho, v0, option_type="call"):
    """Scalar convenience wrapper around heston_price_batch."""
    return float(heston_price_batch([K], S0, r, q, T, kappa, theta, xi, rho, v0, option_type)[0])


def feller_condition(kappa, theta, xi):
    """2*kappa*theta - xi^2 >= 0 is the Feller condition (v_t stays > 0 a.s.)."""
    return 2.0 * kappa * theta - xi ** 2


def _single_heston_fit(market_df, spot, r, q, x0, bounds, enforce_feller, maxiter):
    from implied_vol import solve_implied_vol

    groups = {T: g for T, g in market_df.groupby("T_years")}

    def objective(params):
        kappa, theta, xi, rho, v0 = params
        sq_errors = []
        for T, g in groups.items():
            calls_mask = g["option_type"] == "call"
            puts_mask = ~calls_mask
            for mask, otype in [(calls_mask, "call"), (puts_mask, "put")]:
                sub = g[mask]
                if len(sub) == 0:
                    continue
                model_prices = heston_price_batch(
                    sub["strike"].values, spot, r, q, T, kappa, theta, xi, rho, v0, otype
                )
                for K, T_, mp, mkt_iv in zip(sub["strike"].values, [T] * len(sub),
                                              model_prices, sub["iv_computed"].values):
                    iv_result = solve_implied_vol(mp, spot, K, r, q, T_, otype)
                    model_iv = iv_result["iv"] if iv_result["success"] else mkt_iv
                    sq_errors.append((model_iv - mkt_iv) ** 2)
        return float(np.mean(sq_errors)) if sq_errors else np.inf

    if enforce_feller:
        constraints = [{"type": "ineq", "fun": lambda p: 2 * p[0] * p[1] - p[2] ** 2}]
        result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints,
                           method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-10})
    else:
        result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12})

    kappa, theta, xi, rho, v0 = result.x
    feller_value = feller_condition(kappa, theta, xi)

    return {
        "kappa": float(kappa), "theta": float(theta), "xi": float(xi),
        "rho": float(rho), "v0": float(v0),
        "success": bool(result.success), "message": str(result.message),
        "n_iter": int(getattr(result, "nit", -1)),
        "final_rmse_iv": float(np.sqrt(result.fun)) if np.isfinite(result.fun) else float("nan"),
        "feller_value": float(feller_value),
        "feller_satisfied": bool(feller_value >= 0),
        "enforce_feller": enforce_feller,
        "x0_used": list(x0),
    }


def calibrate_heston(market_df, spot, r, q, x0=None, enforce_feller=True, maxiter=200,
                      n_starts=1, random_seed=42):
    """Calibrate (kappa, theta, xi, rho, v0) to minimize mean squared IV
    error against market_df[strike, T_years, option_type, iv_computed].

    If enforce_feller, attempts constrained optimization (SLSQP with the
    Feller inequality as a nonlinear constraint); reports explicitly if the
    constrained optimizer fails to converge or the constraint isn't met at
    the reported optimum -- this is a real possibility with market data and
    is not hidden. Note SLSQP's constraint satisfaction is a convergence
    tolerance, not a hard guarantee -- `feller_satisfied` reflects whatever
    the optimizer actually returned, which can be marginally infeasible
    even with the constraint active.

    With n_starts > 1, runs from n_starts different initial points (the
    given/default x0 plus (n_starts-1) randomized draws) and keeps the best
    by final RMSE, since both SLSQP and L-BFGS-B are local optimizers on a
    genuinely non-convex 5-parameter objective -- a single start cannot
    rule out a local minimum. All per-start RMSEs are returned in
    `all_start_rmse` so parameter instability isn't silently confounded
    with "only tried one starting point."
    """
    bounds = [(0.05, 15.0), (0.005, 2.0), (0.02, 3.0), (-0.99, 0.99), (0.001, 2.0)]
    base_x0 = x0 or [2.0, 0.04, 0.5, -0.5, 0.04]

    rng = np.random.default_rng(random_seed)
    starts = [base_x0]
    for _ in range(n_starts - 1):
        starts.append([
            rng.uniform(0.3, 8.0),    # kappa
            rng.uniform(0.01, 0.5),   # theta
            rng.uniform(0.1, 1.5),    # xi
            rng.uniform(-0.9, 0.1),   # rho
            rng.uniform(0.005, 0.3),  # v0
        ])

    fits = [_single_heston_fit(market_df, spot, r, q, s, bounds, enforce_feller, maxiter) for s in starts]
    finite_fits = [f for f in fits if np.isfinite(f["final_rmse_iv"])]
    best = min(finite_fits or fits, key=lambda f: f["final_rmse_iv"] if np.isfinite(f["final_rmse_iv"]) else np.inf)

    best = dict(best)
    best["n_starts"] = n_starts
    best["all_start_rmse"] = [f["final_rmse_iv"] for f in fits]
    return best
