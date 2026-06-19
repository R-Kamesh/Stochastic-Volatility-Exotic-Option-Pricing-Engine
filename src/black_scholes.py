"""
black_scholes.py
-----------------
Analytical Black-Scholes-Merton pricing and closed-form Greeks for European
vanilla options. Used as the benchmark that the Monte Carlo engine is
validated against.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Standard d1, d2 terms (q = continuous dividend yield, default 0)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Closed-form Black-Scholes price.

    S : spot price
    K : strike
    T : time to maturity (years)
    r : risk-free rate (annualized, continuously compounded)
    sigma : volatility (annualized)
    option_type : 'call' or 'put'
    q : continuous dividend yield
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Closed-form Greeks: Delta, Gamma, Vega, Theta (per year), Rho.
    Gamma and Vega are identical for calls and puts.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)  # per 1.0 change in sigma

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def put_call_parity_check(S, K, T, r, sigma, q=0.0, tol=1e-8):
    """
    Verifies C - P = S*exp(-qT) - K*exp(-rT). Returns (lhs, rhs, ok).
    """
    call = bs_price(S, K, T, r, sigma, "call", q)
    put = bs_price(S, K, T, r, sigma, "put", q)
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return lhs, rhs, abs(lhs - rhs) < tol


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma, "call")
    greeks = bs_greeks(S, K, T, r, sigma, "call")
    lhs, rhs, ok = put_call_parity_check(S, K, T, r, sigma)
    print(f"BS Call price: {price:.4f}")
    print(f"Greeks: {greeks}")
    print(f"Put-call parity: LHS={lhs:.6f} RHS={rhs:.6f} OK={ok}")
