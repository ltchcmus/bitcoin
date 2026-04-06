from typing import Dict

import numpy as np
import pandas as pd

from trading_pipeline.config import ANNUALIZATION_30S


def run_backtest(
    future_returns: pd.Series,
    probs: np.ndarray,
    confidence_threshold: float,
    fee_per_trade: float,
    confirmation_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    class_idx = np.argmax(probs, axis=1)
    max_prob = np.max(probs, axis=1)
    raw_signal = np.array([-1, 0, 1])[class_idx]
    signal = np.where(max_prob >= confidence_threshold, raw_signal, 0)
    if confirmation_mask is not None:
        signal = np.where(confirmation_mask, signal, 0)

    pnl = signal * future_returns.to_numpy()
    cost = (signal != 0).astype(float) * fee_per_trade
    pnl_after_fee = pnl - cost
    equity_curve = (1 + pd.Series(pnl_after_fee)).cumprod()

    pnl_series = pd.Series(pnl_after_fee)
    std = pnl_series.std()
    sharpe = 0.0 if std == 0 else (pnl_series.mean() / std) * ANNUALIZATION_30S

    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    traded = signal != 0
    if traded.any():
        win_rate = float((pnl_after_fee[traded] > 0).mean())
    else:
        win_rate = 0.0

    return {
        "profit": float(equity_curve.iloc[-1] - 1.0) if len(equity_curve) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "trade_coverage": float(traded.mean()) if len(traded) else 0.0,
        "win_rate": win_rate,
        "avg_confidence": float(max_prob.mean()) if len(max_prob) else 0.0,
    }
