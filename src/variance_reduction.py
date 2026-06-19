"""
variance_reduction.py
-----------------------
Three classic variance-reduction techniques for European option MC pricing:
antithetic variates, control variates (S_T as control), and quasi-Monte
Carlo via Sobol low-discrepancy sequences. Includes an efficiency
comparison (variance reduction factor relative to crude MC, normalized for
equal computational cost).
"""

import numpy as np
from scipy.stats import qmc, norm
try:
    from .black_scholes import bs_price
except ImportError:
    from black_scholes import bs_price


def _payoff(S_T, K, option_type):
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def mc_price_crude(S0, K, T, r, sigma, n_paths, option_type="call", seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    disc_payoff = np.exp(-r * T) * _payoff(S_T, K, option_type)
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n_paths)


def mc_price_antithetic(S0, K, T, r, sigma, n_paths, option_type="call", seed=None):
    """
    Pairs each Z with -Z. n_paths is the total number of simulated paths
    (n_paths // 2 independent pairs).
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    Z = rng.standard_normal(half)
    drift = (r - 0.5 * sigma ** 2) * T

    S_T_pos = S0 * np.exp(drift + sigma * np.sqrt(T) * Z)
    S_T_neg = S0 * np.exp(drift - sigma * np.sqrt(T) * Z)

    payoff_pos = np.exp(-r * T) * _payoff(S_T_pos, K, option_type)
    payoff_neg = np.exp(-r * T) * _payoff(S_T_neg, K, option_type)

    pair_avg = 0.5 * (payoff_pos + payoff_neg)
    price = pair_avg.mean()
    stderr = pair_avg.std(ddof=1) / np.sqrt(half)
    return price, stderr, payoff_pos, payoff_neg


def mc_price_control_variate(S0, K, T, r, sigma, n_paths, option_type="call", seed=None):
    """
    Uses S_T itself as the control variate: E[S_T] = S0*exp(rT) is known
    exactly, and Cov(payoff, S_T) is estimated from the same sample to pick
    the optimal coefficient beta.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.exp(-r * T) * _payoff(S_T, K, option_type)
    control = S_T  # not discounted; control adjustment handles the scale

    mean_control = S0 * np.exp(r * T)
    cov = np.cov(payoff, control, ddof=1)
    beta = cov[0, 1] / cov[1, 1]

    adjusted = payoff - beta * (control - mean_control)
    price = adjusted.mean()
    stderr = adjusted.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr, beta


def mc_price_sobol(S0, K, T, r, sigma, n_paths, option_type="call", seed=None):
    """
    Quasi-Monte Carlo using a scrambled Sobol sequence mapped through the
    inverse normal CDF. n_paths is rounded up to the next power of two,
    as required by Sobol sequence construction.
    """
    m = int(np.ceil(np.log2(n_paths)))
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    U = sampler.random_base2(m=m).flatten()
    U = np.clip(U, 1e-10, 1 - 1e-10)  # avoid inf at the tails of the inverse CDF
    Z = norm.ppf(U)

    drift = (r - 0.5 * sigma ** 2) * T
    S_T = S0 * np.exp(drift + sigma * np.sqrt(T) * Z)
    disc_payoff = np.exp(-r * T) * _payoff(S_T, K, option_type)
    price = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(len(Z))
    return price, stderr, len(Z)


def efficiency_table(S0, K, T, r, sigma, n_paths, option_type="call", seed=42):
    """
    Builds a comparison table: std error and efficiency factor (crude
    variance / method variance) for each technique, plus absolute error
    against the closed-form Black-Scholes price.
    """
    bs_ref = bs_price(S0, K, T, r, sigma, option_type)

    crude_price, crude_se = mc_price_crude(S0, K, T, r, sigma, n_paths, option_type, seed)
    anti_price, anti_se, _, _ = mc_price_antithetic(S0, K, T, r, sigma, n_paths, option_type, seed)
    cv_price, cv_se, beta = mc_price_control_variate(S0, K, T, r, sigma, n_paths, option_type, seed)
    sobol_price, sobol_se, n_used = mc_price_sobol(S0, K, T, r, sigma, n_paths, option_type, seed)

    rows = {
        "Crude MC": (crude_price, crude_se, 1.0, abs(crude_price - bs_ref)),
        "Antithetic": (anti_price, anti_se, (crude_se / anti_se) ** 2, abs(anti_price - bs_ref)),
        "Control Variate": (cv_price, cv_se, (crude_se / cv_se) ** 2, abs(cv_price - bs_ref)),
        "Quasi-MC (Sobol)": (sobol_price, sobol_se, (crude_se / sobol_se) ** 2, abs(sobol_price - bs_ref)),
    }
    return rows, bs_ref


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    rows, bs_ref = efficiency_table(S, K, T, r, sigma, n_paths=50_000)
    print(f"BS reference price: {bs_ref:.4f}\n")
    for name, (price, se, eff, err) in rows.items():
        print(f"{name:18s} price={price:.4f}  se={se:.5f}  efficiency={eff:.2f}x  |err|={err:.5f}")