"""
greeks.py
----------
Finite-difference Greeks computed on top of the Monte Carlo engine, and
the equivalent bump-and-reprice machinery to study stability vs bump size h.
Compares against the closed-form Black-Scholes Greeks.
"""

import numpy as np
try:
    from .monte_carlo import mc_price
    from .black_scholes import bs_greeks
except ImportError:  # allows running this file standalone too
    from monte_carlo import mc_price
    from black_scholes import bs_greeks


def mc_delta_gamma(S0, K, T, r, sigma, n_paths, option_type="call", h=1.0, seed=42):
    """Central-difference Delta and second-order central-difference Gamma in S."""
    p_up, _ = mc_price(S0 + h, K, T, r, sigma, n_paths, option_type, seed)
    p_mid, _ = mc_price(S0, K, T, r, sigma, n_paths, option_type, seed)
    p_down, _ = mc_price(S0 - h, K, T, r, sigma, n_paths, option_type, seed)

    delta = (p_up - p_down) / (2 * h)
    gamma = (p_up - 2 * p_mid + p_down) / (h ** 2)
    return delta, gamma


def mc_vega(S0, K, T, r, sigma, n_paths, option_type="call", h=0.01, seed=42):
    p_up, _ = mc_price(S0, K, T, r, sigma + h, n_paths, option_type, seed)
    p_down, _ = mc_price(S0, K, T, r, sigma - h, n_paths, option_type, seed)
    return (p_up - p_down) / (2 * h)


def mc_theta(S0, K, T, r, sigma, n_paths, option_type="call", h=1 / 365, seed=42):
    # Theta is conventionally reported as -dPrice/dT (time decay as T shrinks)
    p_up, _ = mc_price(S0, K, T + h, r, sigma, n_paths, option_type, seed)
    p_down, _ = mc_price(S0, K, T - h, r, sigma, n_paths, option_type, seed)
    return -(p_up - p_down) / (2 * h)


def mc_rho(S0, K, T, r, sigma, n_paths, option_type="call", h=0.0001, seed=42):
    p_up, _ = mc_price(S0, K, T, r + h, sigma, n_paths, option_type, seed)
    p_down, _ = mc_price(S0, K, T, r - h, sigma, n_paths, option_type, seed)
    return (p_up - p_down) / (2 * h)


def mc_all_greeks(S0, K, T, r, sigma, n_paths, option_type="call", seed=42):
    delta, gamma = mc_delta_gamma(S0, K, T, r, sigma, n_paths, option_type, seed=seed)
    vega = mc_vega(S0, K, T, r, sigma, n_paths, option_type, seed=seed)
    theta = mc_theta(S0, K, T, r, sigma, n_paths, option_type, seed=seed)
    rho = mc_rho(S0, K, T, r, sigma, n_paths, option_type, seed=seed)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def greeks_comparison_table(S0, K, T, r, sigma, n_paths, option_type="call", seed=42):
    """Returns dict of {greek: (bs_value, mc_value, pct_error)}."""
    bs = bs_greeks(S0, K, T, r, sigma, option_type)
    mc = mc_all_greeks(S0, K, T, r, sigma, n_paths, option_type, seed)
    table = {}
    for g in bs:
        bs_val, mc_val = bs[g], mc[g]
        pct_err = abs(mc_val - bs_val) / abs(bs_val) * 100 if bs_val != 0 else np.nan
        table[g] = (bs_val, mc_val, pct_err)
    return table


def bump_stability_study(S0, K, T, r, sigma, n_paths, option_type="call",
                          h_list=None, seed=42, greek="delta"):
    """
    Sweeps bump size h and records the resulting Delta or Gamma estimate,
    to visualize the bias/variance tradeoff (small h -> noise dominates,
    large h -> truncation error dominates).
    """
    if h_list is None:
        h_list = np.concatenate([np.linspace(0.05, 2.0, 20), np.linspace(2.5, 10, 10)])
    results = []
    for h in h_list:
        delta, gamma = mc_delta_gamma(S0, K, T, r, sigma, n_paths, option_type, h, seed)
        results.append((h, delta, gamma))
    return results


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    table = greeks_comparison_table(S, K, T, r, sigma, n_paths=200_000)
    for g, (bs_v, mc_v, err) in table.items():
        print(f"{g:6s}  BS={bs_v: .5f}  MC={mc_v: .5f}  err={err:.3f}%")