"""
monte_carlo.py
---------------
Core Monte Carlo engine: simulates terminal stock prices (and full paths)
under Geometric Brownian Motion, prices European options, and provides
standard-error / convergence diagnostics.
"""

import numpy as np


def simulate_terminal_prices(S0, r, sigma, T, n_paths, seed=None, q=0.0):
    """
    Exact (no discretization error) simulation of S_T under risk-neutral GBM:
        S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=None, q=0.0, scheme="exact"):
    """
    Full path simulation, needed for path-dependent payoffs (Asian, Barrier)
    and for Greeks bump tests. scheme='exact' uses the exact GBM solution at
    each step; scheme='euler' uses Euler-Maruyama discretization.

    Returns an array of shape (n_paths, n_steps + 1), column 0 = S0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    if scheme == "exact":
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_increments = drift + diffusion
        paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
    elif scheme == "euler":
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] + (r - q) * paths[:, t] * dt + \
                sigma * paths[:, t] * np.sqrt(dt) * Z[:, t]
    else:
        raise ValueError("scheme must be 'exact' or 'euler'")

    return paths


def _payoff(S_T, K, option_type):
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def mc_price(S0, K, T, r, sigma, n_paths, option_type="call", seed=None, q=0.0):
    """
    Crude Monte Carlo price + standard error for a European vanilla option.
    Returns (price, stderr).
    """
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_paths, seed, q)
    discounted_payoff = np.exp(-r * T) * _payoff(S_T, K, option_type)
    price = discounted_payoff.mean()
    stderr = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr


def convergence_study(S0, K, T, r, sigma, n_list, option_type="call", seed=None, q=0.0):
    """
    Runs mc_price across a list of path counts. Returns lists of
    (n_paths, price, stderr).
    """
    results = []
    for i, N in enumerate(n_list):
        s = None if seed is None else seed + i
        price, stderr = mc_price(S0, K, T, r, sigma, N, option_type, s, q)
        results.append((N, price, stderr))
    return results


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    price, se = mc_price(S, K, T, r, sigma, n_paths=200_000, seed=42)
    print(f"MC Call price: {price:.4f}  (stderr {se:.4f})")