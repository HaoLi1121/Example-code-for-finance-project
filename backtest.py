"""Walk-forward backtest: re-select pairs each period, trade them out of sample."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .selection import Pair, select_pairs
from .signals import positions, zscore


def pair_returns(prices: pd.DataFrame, pair: Pair, pos: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Daily net return of one pair, per unit of gross capital.

    Signal at close t -> traded at close t, earns returns from t+1 (pos is shifted).
    Weights: long 1 unit of y, short beta units of x, scaled so gross exposure = 1.
    Cost is charged on gross turnover each time the position changes.
    """
    rets = prices[[pair.y, pair.x]].pct_change().fillna(0.0)
    gross = 1.0 + abs(pair.beta)
    w_y, w_x = 1.0 / gross, -pair.beta / gross
    held = pos.shift(1).fillna(0.0)
    gross_ret = held * (w_y * rets[pair.y] + w_x * rets[pair.x])
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame({"ret": gross_ret - cost, "pos": pos, "held": held})


def walk_forward(
    prices: pd.DataFrame,
    formation_days: int = 252,
    trading_days: int = 126,
    cost_bps: float = 10.0,
    entry: float = 1.5,
    exit: float = 0.0,
    stop: float = 2.5,
    **select_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (daily portfolio returns, trade log).

    Capital is split equally across the pairs chosen for each period; capital
    allocated to a pair that is flat earns zero, so returns are not inflated.
    """
    logp = np.log(prices)
    port, trades = [], []
    start = formation_days
    while start < len(prices):
        form = logp.iloc[start - formation_days:start]
        trade_idx = prices.index[start:start + trading_days]
        pairs = select_pairs(form, **select_kwargs)

        # Include the last formation day so the first trading-day return is defined.
        window = prices.index[start - 1:start + trading_days]
        if pairs:
            per_pair = []
            for p in pairs:
                z = zscore(logp.loc[window], p)
                pos = positions(z, entry, exit, stop)
                pr = pair_returns(prices.loc[window], p, pos, cost_bps)
                per_pair.append(pr["ret"].loc[trade_idx])
                trades += _trade_log(pr.loc[trade_idx], p, trade_idx[0])
            port.append(pd.concat(per_pair, axis=1).mean(axis=1))
        else:
            port.append(pd.Series(0.0, index=trade_idx))
        start += trading_days

    return pd.concat(port).rename("strategy"), pd.DataFrame(trades)


def _trade_log(pr: pd.DataFrame, pair: Pair, period: pd.Timestamp) -> list[dict]:
    """One row per round trip: entry, exit, direction, compounded return."""
    out, held = [], pr["held"]
    trade_id = (held != held.shift()).cumsum()
    for _, seg in pr[held != 0].groupby(trade_id[held != 0]):
        out.append({
            "period": period, "pair": f"{pair.y}/{pair.x}",
            "direction": int(seg["held"].iloc[0]),
            "entry": seg.index[0], "exit": seg.index[-1], "days": len(seg),
            "return": (1 + seg["ret"]).prod() - 1,
        })
    return out
