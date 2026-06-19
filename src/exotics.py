"""
exotics.py
-----------
Path-dependent option pricing: arithmetic & geometric Asian options, and
single knock-in/knock-out barrier options. Includes the discretely-monitored
geometric-Asian closed-form (Kemna-Vorst style) used to validate the MC
engine, and the standard in-out barrier parity identity:
    KnockOut + KnockIn = Vanilla   (same barrier, same side)
"""

import numpy as np
from scipy.stats import norm
try:
    from .monte_carlo import simulate_paths
except ImportError:
    from monte_carlo import simulate_paths


def _payoff(values, K, option_type):
    if option_type == "call":
        return np.maximum(values - K, 0.0)
    return np.maximum(K - values, 0.0)


# ---------------------------------------------------------------- Asian ---

def asian_arithmetic_mc(S0, K, T, r, sigma, n_steps, n_paths, option_type="call", seed=None):
    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, scheme="exact")
    avg = paths[:, 1:].mean(axis=1)  # average over the monitoring dates, excludes S0
    disc_payoff = np.exp(-r * T) * _payoff(avg, K, option_type)
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n_paths)


def asian_geometric_mc(S0, K, T, r, sigma, n_steps, n_paths, option_type="call", seed=None):
    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, scheme="exact")
    geo_avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    disc_payoff = np.exp(-r * T) * _payoff(geo_avg, K, option_type)
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n_paths)


def asian_geometric_closed_form(S0, K, T, r, sigma, n_steps, option_type="call"):
    """
    Discretely-monitored geometric-average Asian option, closed form
    (Kemna & Vorst, 1990 style adjusted-volatility BS formula).
    n_steps = number of equally spaced monitoring dates.
    """
    n = n_steps
    sigma_G = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6.0 * n ** 2))
    b_G = (r - 0.5 * sigma ** 2) * (n + 1) / (2.0 * n) + 0.5 * sigma_G ** 2

    d1 = (np.log(S0 / K) + (b_G + 0.5 * sigma_G ** 2) * T) / (sigma_G * np.sqrt(T))
    d2 = d1 - sigma_G * np.sqrt(T)

    if option_type == "call":
        price = S0 * np.exp((b_G - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp((b_G - r) * T) * norm.cdf(-d1)
    return price


# -------------------------------------------------------------- Barrier ---

def barrier_option_mc(S0, K, B, T, r, sigma, n_steps, n_paths,
                       option_type="call", barrier_type="up-and-out", seed=None):
    """
    barrier_type in {'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'}.
    Discrete monitoring at each simulated step (n_steps observations).
    """
    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, scheme="exact")
    S_T = paths[:, -1]
    running_max = paths.max(axis=1)
    running_min = paths.min(axis=1)

    if barrier_type == "up-and-out":
        alive = running_max < B
    elif barrier_type == "up-and-in":
        alive = running_max >= B
    elif barrier_type == "down-and-out":
        alive = running_min > B
    elif barrier_type == "down-and-in":
        alive = running_min <= B
    else:
        raise ValueError("invalid barrier_type")

    payoff = _payoff(S_T, K, option_type) * alive
    disc_payoff = np.exp(-r * T) * payoff
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n_paths)


def barrier_parity_check(S0, K, B, T, r, sigma, n_steps, n_paths, option_type="call",
                          side="up", seed=None):
    """
    Verifies KnockOut + KnockIn == Vanilla (same seed -> same underlying
    paths for a fair, low-noise comparison).
    """
    out_type = f"{side}-and-out"
    in_type = f"{side}-and-in"

    out_price, _ = barrier_option_mc(S0, K, B, T, r, sigma, n_steps, n_paths, option_type, out_type, seed)
    in_price, _ = barrier_option_mc(S0, K, B, T, r, sigma, n_steps, n_paths, option_type, in_type, seed)

    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, scheme="exact")
    S_T = paths[:, -1]
    vanilla_price = (np.exp(-r * T) * _payoff(S_T, K, option_type)).mean()

    return out_price, in_price, out_price + in_price, vanilla_price


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    n_steps, n_paths = 252, 200_000

    arith_price, arith_se = asian_arithmetic_mc(S, K, T, r, sigma, n_steps, n_paths, seed=1)
    geo_mc_price, geo_se = asian_geometric_mc(S, K, T, r, sigma, n_steps, n_paths, seed=1)
    geo_cf_price = asian_geometric_closed_form(S, K, T, r, sigma, n_steps)

    print(f"Asian arithmetic (MC):        {arith_price:.4f}  (se {arith_se:.4f})")
    print(f"Asian geometric  (MC):        {geo_mc_price:.4f}  (se {geo_se:.4f})")