"""Error metrics and computational-cost benchmarking, shared across notebooks."""
from __future__ import annotations

import time
import numpy as np


def rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def iv_rmse(model_ivs, market_ivs):
    """RMSE on implied vol, not price -- price errors are dominated by ITM
    contracts, IV errors are comparable in magnitude across moneyness."""
    return rmse(model_ivs, market_ivs)


def time_call(fn, *args, n_reps=5, **kwargs):
    """Wall-clock timing for a pricing call, averaged over n_reps to reduce
    noise. Returns (result_of_last_call, mean_seconds, std_seconds)."""
    times = []
    result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return result, float(times.mean()), float(times.std(ddof=1)) if n_reps > 1 else 0.0
