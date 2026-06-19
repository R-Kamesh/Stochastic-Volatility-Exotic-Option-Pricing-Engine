# Stochastic Volatility & Exotic Option Pricing Engine

A professional-grade quantitative derivative pricing framework built in Python. This engine compares analytical closed-form solutions against Monte Carlo simulations, featuring advanced variance reduction techniques, path-dependent exotic options pricing, and stochastic volatility modeling via the Heston model.

## Features

- **Core Pricer**: Analytical Black-Scholes-Merton pricing and core Monte Carlo simulation engine for European vanilla options.
- **Advanced Greeks Analysis**: Comparison between Closed-form Greeks and Finite-Difference (Bump-and-Reprice) Monte Carlo Greeks.
- **Variance Reduction**: Implements Antithetic Variates, Control Variates, and Quasi-Monte Carlo (Sobol low-discrepancy sequences) to optimize convergence.
- **Exotic Options**: Pricing for path-dependent derivatives including Arithmetic & Geometric Asian options, and Knock-in/Knock-out Barrier options.
- **Stochastic Volatility**: Heston model path simulation using Full-Truncation Euler-Maruyama, pricing, and implied volatility (smile/skew) extraction.
- **Robustness**: Built-in sanity checks including Put-Call Parity and Barrier Parity evaluations.

## Project Structure

```text
├── src/
│   ├── black_scholes.py      # Closed-form BS pricing and Greeks
│   ├── monte_carlo.py        # Core simulation engine (Exact and Euler schemes)
│   ├── variance_reduction.py # Optimization techniques (Antithetic, CV, Sobol)
│   ├── exotics.py            # Path-dependent options (Asian, Barrier)
│   ├── greeks.py             # Finite-difference sensitivity analysis
│   └── heston.py             # Stochastic volatility and IV extraction
├── notebooks/
│   └── full_analysis.ipynb   # End-to-end orchestration, convergence tests, and visualization
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`

### Installation

Clone the repository and install the required lightweight dependencies:

```bash
git clone https://github.com/R-Kamesh/Stochastic-Volatility-Exotic-Option-Pricing-Engine.git
cd Stochastic-Volatility-Exotic-Option-Pricing-Engine
pip install -r requirements.txt
```

### Usage

You can run the core modules independently or explore the Jupyter Notebook for a comprehensive mathematical walkthrough and visualizations.

#### Example: Pricing a Vanilla Call Option
```python
from src.black_scholes import bs_price
from src.monte_carlo import mc_price

S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

# Analytical
bs_val = bs_price(S, K, T, r, sigma, option_type="call")
print(f"BS Price: {bs_val:.4f}")

# Monte Carlo
mc_val, stderr = mc_price(S, K, T, r, sigma, n_paths=200_000, option_type="call", seed=42)
print(f"MC Price: {mc_val:.4f} (stderr: {stderr:.4f})")
```

## Future Roadmap

- [ ] Transition to strict Python Type Hinting for enhanced developer experience.
- [ ] Migrate `if __name__ == '__main__':` parity checks to a dedicated `pytest` regression suite.
- [ ] Implement custom exceptions (e.g., `ModelConvergenceError`) for robust error handling.