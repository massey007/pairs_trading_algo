# Pairs Trading Algorithm (Statistical Arbitrage)

A Python-based long/short trading strategy that exploits temporary price discrepancies between historically correlated assets using mean reversion.

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* `pandas`
* `numpy`
* `statsmodels`

### Installation
```bash
git clone https://github.com
cd pairs_trading_algo
pip install -r requirements.txt
```

### Usage
```bash
python main.py --assetA AAPL --assetB MSFT
```

## 📊 The Math Sequence

The algorithm operates on the mathematical principle of mean reversion through a two-step process:

1. **Calculate the spread:**
   $$\text{Spread}_t = \ln(P_{A, t}) - (\beta \times \ln(P_{B, t}))$$
   *(where $\beta$ is the hedge ratio calculated via linear regression).*

2. **Calculate the Z-score:**
   $$Z_t = \frac{\text{Spread}_t-\mu_{\text{Spread}}}{\sigma_{\text{Spread}}}$$
   *(where $\mu$ and $\sigma$ are the rolling historical mean and standard deviation of the spread).*

## ⚙️ Execution Rules

The strategy triggers trades when prices diverge unnaturally, betting that they will eventually snap back to their historical relationship.

* **Sell Signal (Spread too wide):** When $Z_t \geq +2.0$
  * **Action:** Short asset A and Long Asset B.
  * **Rationale:** Asset A is overvalued relative to Asset B.
* **Buy Signal (Spread too narrow):** When $Z_t \leq -2.0$
  * **Action:** Long Asset A and Short Asset B.
  * **Rationale:** Asset A is undervalued relative to Asset B.
* **Exit Signal (Converged to Mean):** When $Z_t = 0$
  * **Action:** Close both positions and lock in profits.

