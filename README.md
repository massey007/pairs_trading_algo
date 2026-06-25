# pairs_trading_algo

When it comes to long/short strategy trading, a powerful tool/signal to use is Pairs Trading (Statistical Arbitrage), which relies on the mathematical principle of mean reversion.

The idea is centered around not betting on market direction, instead you bet on the price spread between two historically correlated assets (like two gold mining stocks or two retail giants or two tech firms).

The Math Sequence:

1. Calculate the spread:
$\text{Spread}_t = ln(P_{A, t}) - (\beta \times ln(P_{B, t}))$
(where $\beta$ is the hedge ratio calculated via linear regression).
2. Calculate the Z-score:
$Z_t = \frac{\text{Spread}_t-\mu_{\text{Spread}}}{\sigma_{\text{Spread}}}$
(Where $\mu = $ The rolling historical mean of the spread, $\sigma = $ The rolling historical standard deviation of the spread.

### Execution Rulles (the Strategy)
The strategy triggers trades when prices diverge unnaturally, bettin that they will eventually snap back to their historical relationship.

- Sell Signal (Spread is too wide): When $Z_t \geq + 2.0$
  - Action: Short asset A and Long Asset B.
  - Rationale: Asset A is overvalued relative to Asset B.
- Buy Signal (Spread is too narrow): When $Z_t \leq -2.0$
  - Action: Long Asset A and Short Asset B
  - Rationale: Asset A is undervalued relative to Asset B.
- Exit Signal (Converged to Mean): When $Z_t = 0$
  - Action: Close both positions and lock in profits.

