
import numpy as np
from scipy.optimize import brentq
try:
    from .black_scholes import bs_price
except ImportError:
    from black_scholes import bs_price


def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed=None):
    """
    Returns (S_paths, v_paths), each of shape (n_paths, n_steps + 1).
    Uses full-truncation Euler-Maruyama for both SDEs with correlated
    Brownian increments.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2  # correlate the two drivers

        v_pos = np.maximum(v[:, t], 0.0)  # full truncation: use max(v,0) in the diffusion terms
        sqrt_v = np.sqrt(v_pos)

        S[:, t + 1] = S[:, t] + r * S[:, t] * dt + sqrt_v * S[:, t] * np.sqrt(dt) * W1
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * np.sqrt(dt) * W2

    return S, v


def heston_mc_price(S0, K, T, r, v0, kappa, theta, xi, rho, n_steps, n_paths,
                     option_type="call", seed=None):
    """European option price under Heston via Monte Carlo. Returns (price, stderr)."""
    S, _ = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed)
    S_T = S[:, -1]
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    disc_payoff = np.exp(-r * T) * payoff
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n_paths)


def implied_vol(price, S, K, T, r, option_type="call", lo=1e-4, hi=3.0):
    """
    Inverts the Black-Scholes formula to find the volatility that
    reproduces a given option price (Newton/Brent root-find on price - BS(sigma) = 0).
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    # guard against a price outside what's achievable in [lo, hi]
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(objective, lo, hi, xtol=1e-8)


def implied_vol_smile(S0, T, r, v0, kappa, theta, xi, rho, n_steps, n_paths,
                       strikes, option_type="call", seed=None):
    """
    Prices a European option under Heston at each strike, then backs out
    the Black-Scholes implied volatility at each strike. A flat curve
    would mean Heston reduces to GBM; a curved one is the volatility
    "smile"/"skew" that constant-volatility GBM cannot produce.
    """
    ivs, prices = [], []
    for i, K in enumerate(strikes):
        s = None if seed is None else seed + i
        price, _ = heston_mc_price(S0, K, T, r, v0, kappa, theta, xi, rho,
                                    n_steps, n_paths, option_type, s)
        prices.append(price)
        ivs.append(implied_vol(price, S0, K, T, r, option_type))
    return np.array(prices), np.array(ivs)


if __name__ == "__main__":
    # Calibrated so v0 = 0.20^2, matching the constant sigma=0.20 used
    # throughout the rest of this project.
    S0, K, T, r = 100, 110, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.20 ** 2, 2.0, 0.20 ** 2, 0.30, -0.7

    price, se = heston_mc_price(S0, K, T, r, v0, kappa, theta, xi, rho,
                                 n_steps=252, n_paths=200_000, seed=42)
    bs_ref = bs_price(S0, K, T, r, sigma=np.sqrt(v0))
    iv = implied_vol(price, S0, K, T, r)

    print(f"Heston price: {price:.4f} (stderr {se:.4f})")
    print(f"BS price (flat sigma=0.20): {bs_ref:.4f}")
    print(f"Heston implied vol @ K={K}: {iv:.4f}")