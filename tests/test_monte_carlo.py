import numpy as np
import pytest
from scipy import stats

from monte_carlo import mc_price, mc_price_plain, mc_price_antithetic, mc_price_control_variate
from black_scholes import bs_price


S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0


def test_plain_mc_converges_to_bs_within_confidence_interval():
    bs = bs_price(S, K, r, q, sigma, T, "call")
    result = mc_price_plain(S, K, r, q, sigma, T, n_paths=200_000, option_type="call", seed=42)
    # 4-sigma band: should basically never fail spuriously
    assert abs(result["price"] - bs) < 4 * result["se"]


def test_se_scales_like_inverse_sqrt_n():
    # H1 (MC piece): SE should shrink as N grows, with a log-log slope near -0.5
    Ns = [1_000, 4_000, 16_000, 64_000, 256_000]
    ses = [mc_price_plain(S, K, r, q, sigma, T, n, "call", seed=1)["se"] for n in Ns]
    logN = np.log(Ns)
    logSE = np.log(ses)
    slope, _ = np.polyfit(logN, logSE, 1)
    assert -0.65 < slope < -0.35


def test_antithetic_reduces_variance_vs_plain():
    n = 100_000
    plain = mc_price_plain(S, K, r, q, sigma, T, n, "call", seed=7)
    anti = mc_price_antithetic(S, K, r, q, sigma, T, n, "call", seed=7)
    assert anti["se"] < plain["se"]


def test_control_variate_reduces_variance_vs_plain():
    n = 100_000
    plain = mc_price_plain(S, K, r, q, sigma, T, n, "call", seed=7)
    cv = mc_price_control_variate(S, K, r, q, sigma, T, n, "call", seed=7)
    assert cv["se"] < plain["se"]
    # correlation between payoff and S_T should be strong and positive for a call
    assert cv["corr_Y_C"] > 0.5


def test_variance_reduction_is_unbiased_h3():
    # H3: paired comparison across seeds -- variance reduction should not
    # shift the mean relative to the closed-form benchmark. Report SE
    # reduction % and a t-test on (estimate - BS) for each method.
    bs = bs_price(S, K, r, q, sigma, T, "call")
    n = 50_000
    seeds = list(range(30))

    plain_errors = np.array([mc_price_plain(S, K, r, q, sigma, T, n, "call", s)["price"] - bs for s in seeds])
    anti_errors = np.array([mc_price_antithetic(S, K, r, q, sigma, T, n, "call", s)["price"] - bs for s in seeds])
    cv_errors = np.array([mc_price_control_variate(S, K, r, q, sigma, T, n, "call", s)["price"] - bs for s in seeds])

    plain_se_mean = np.array([mc_price_plain(S, K, r, q, sigma, T, n, "call", s)["se"] for s in seeds]).mean()
    anti_se_mean = np.array([mc_price_antithetic(S, K, r, q, sigma, T, n, "call", s)["se"] for s in seeds]).mean()
    cv_se_mean = np.array([mc_price_control_variate(S, K, r, q, sigma, T, n, "call", s)["se"] for s in seeds]).mean()

    # variance reduction should show up in the realized SE across seeds too
    assert anti_se_mean < plain_se_mean
    assert cv_se_mean < plain_se_mean

    # unbiasedness: t-test that mean error is not significantly different from 0
    for errors, name in [(plain_errors, "plain"), (anti_errors, "antithetic"), (cv_errors, "control")]:
        t_stat, p_value = stats.ttest_1samp(errors, 0.0)
        assert p_value > 0.01, f"{name} estimator appears biased: p={p_value:.4f}"


def test_put_call_agree_with_bs():
    bs_put = bs_price(S, K, r, q, sigma, T, "put")
    result = mc_price_plain(S, K, r, q, sigma, T, n_paths=200_000, option_type="put", seed=99)
    assert abs(result["price"] - bs_put) < 4 * result["se"]


def test_dispatcher_matches_direct_calls():
    r1 = mc_price(S, K, r, q, sigma, T, 10_000, "call", seed=3, method="plain")
    r2 = mc_price_plain(S, K, r, q, sigma, T, 10_000, "call", seed=3)
    assert r1["price"] == r2["price"]


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        mc_price(S, K, r, q, sigma, T, 1000, "call", seed=1, method="bogus")
