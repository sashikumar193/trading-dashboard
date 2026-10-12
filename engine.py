"""
Simple but realistic vectorized + event-style backtester.
Supports long-only or long/short, commission, slippage.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal
from risk.manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str                 # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, Any]
    signals: pd.Series
    params: Dict[str, Any]


class Backtester:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk = RiskManager(
            risk_config or RiskConfig(), initial_capital
        )

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "ASSET",
        allow_short: bool = False,
    ) -> BacktestResult:
        """
        Run backtest on OHLCV dataframe.
        Assumes columns: open, high, low, close, volume
        """
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        signals = self.strategy.generate_signals(data)
        signals = signals.reindex(data.index).fillna(Signal.HOLD)

        equity = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        side = None
        trades: List[Trade] = []
        equity_curve = []

        for i, (ts, row) in enumerate(data.iterrows()):
            price = row["close"]
            signal = signals.loc[ts]

            # Update risk
            self.risk.update_equity(equity, ts.to_pydatetime())
            if self.risk.killed:
                # Force close everything
                if position != 0:
                    exit_price = price * (1 - self.slippage_pct if position > 0 else 1 + self.slippage_pct)
                    pnl = (exit_price - entry_price) * position
                    commission = abs(position) * exit_price * self.commission_pct
                    equity += pnl - commission
                    trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=ts.to_pydatetime(),
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            size=abs(position),
                            pnl=pnl - commission,
                            pnl_pct=(exit_price / entry_price - 1) * (1 if position > 0 else -1),
                            commission=commission,
                        )
                    )
                    position = 0.0
                equity_curve.append(equity)
                continue

            # Exit logic
            if position != 0 and (
                (position > 0 and signal == Signal.SELL)
                or (position < 0 and signal == Signal.BUY)
                or (not allow_short and signal == Signal.SELL)
            ):
                exit_price = price * (
                    1 - self.slippage_pct if position > 0 else 1 + self.slippage_pct
                )
                pnl = (exit_price - entry_price) * position
                commission = abs(position) * exit_price * self.commission_pct
                equity += pnl - commission
                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=ts.to_pydatetime(),
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=abs(position),
                        pnl=pnl - commission,
                        pnl_pct=(exit_price / entry_price - 1) * (1 if position > 0 else -1),
                        commission=commission,
                    )
                )
                self.risk.register_position(symbol, 0.0)
                position = 0.0
                entry_price = 0.0
                side = None

            # Entry logic
            if position == 0:
                if signal == Signal.BUY:
                    size = self.risk.position_size(equity, price)
                    if self.risk.can_open_position(symbol, equity, price, size):
                        entry_price = price * (1 + self.slippage_pct)
                        commission = size * entry_price * self.commission_pct
                        equity -= commission
                        position = size
                        entry_time = ts.to_pydatetime()
                        side = "long"
                        self.risk.register_position(symbol, position)

                elif signal == Signal.SELL and allow_short:
                    size = self.risk.position_size(equity, price)
                    if self.risk.can_open_position(symbol, equity, price, size):
                        entry_price = price * (1 - self.slippage_pct)
                        commission = size * entry_price * self.commission_pct
                        equity -= commission
                        position = -size
                        entry_time = ts.to_pydatetime()
                        side = "short"
                        self.risk.register_position(symbol, position)

            # Mark-to-market equity
            if position != 0:
                mtm = (price - entry_price) * position
                current_equity = equity + mtm
            else:
                current_equity = equity
            equity_curve.append(current_equity)

        equity_series = pd.Series(equity_curve, index=data.index, name="equity")
        metrics = self._compute_metrics(equity_series, trades)
        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
            signals=signals,
            params=self.strategy.params,
        )

    def _compute_metrics(
        self, equity: pd.Series, trades: List[Trade]
    ) -> Dict[str, Any]:
        if equity.empty:
            return {}

        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualisation assumes ~8760 hours/year for 1h data; adjust if needed
        periods_per_year = 8760
        if len(returns) > 1:
            sharpe = (
                np.sqrt(periods_per_year) * returns.mean() / returns.std()
                if returns.std() > 0
                else 0.0
            )
        else:
            sharpe = 0.0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0.0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0.0
        profit_factor = (
            abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing))
            if losing and sum(t.pnl for t in losing) != 0
            else float("inf")
        )

        return {
            "total_return_pct": total_return * 100,
            "final_equity": equity.iloc[-1],
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "num_trades": len(trades),
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_commission": sum(t.commission for t in trades),
        }