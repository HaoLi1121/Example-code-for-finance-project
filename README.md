# Walk-Forward Pairs Trading

A cointegration-based statistical arbitrage strategy, backtested out of sample with
rolling pair re-selection and transaction costs. It extends the single-window design in
[Jialu, Zhao, Li & Guo (2024), *Heliyon*](https://doi.org/10.1016/j.heliyon.2024.e30876)
to a full walk-forward test.

## Results

<!-- TODO: replace with your real-data run -->
| Metric | Value |
|---|---|
| Sample | 2015 to 2024, 30 U.S. healthcare stocks |
| Annual return (net, 10 bps) | |
| Sharpe | |
| Max drawdown | |
| Trades / win rate | |

![equity curve](results/equity_curve.png)

## Method

1. **Formation (252 days).** Keep pairs with log-price correlation of at least 0.8, then
   Engle-Granger cointegration (p < 0.05, both orientations), ADF confirmation on the
   residual, and an AR(1) half-life between 1 and 60 days. Each stock is used in at most
   one pair; the top 10 pairs by p-value are traded.
2. **Trading (next 126 days).** Spread z-score uses formation-window parameters only.
   Enter at |z| > 1.5, exit when z crosses 0, stop out at |z| > 2.5, and force-close at the
   end of the window.
3. **Execution.** Signals at close t earn returns from t+1. Positions are long 1 unit of y
   and short beta units of x, scaled to unit gross exposure. Costs are charged on turnover.
4. **Roll** forward 126 days and repeat. Capital is split equally across selected pairs,
   and idle capital earns zero.

## Robustness
<!-- TODO: add the sensitivity heatmap (entry x stop) and cost sweep (0 / 10 / 20 bps) -->

## Limitations

- **Survivorship bias:** the universe is today's index members.
- **Multiple testing:** with N stocks there are N(N-1)/2 tests, so some selected pairs are
  spurious. The synthetic run shows this: a pure random walk passes the filters.
- **Execution:** closing-price fills, and no borrow costs or short constraints.

## Usage

```bash
pip install -r requirements.txt
python run.py --synthetic                # offline check on planted pairs
python run.py --start 2014-01-01 --end 2025-01-01 --cost-bps 10
pytest -q                                # includes a look-ahead test
```

`tests/test_backtest.py::test_no_lookahead` scrambles all prices after a cutoff date and
checks that every return before the cutoff is unchanged.
