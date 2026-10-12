"""
Multi-symbol portfolio backtester.
Runs the same (or different) strategy on multiple symbols and aggregates equity.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy
from backtester.engine import Backtester, BacktestResult, Trade
from risk.manager import RiskConfig, RiskManager


@dataclass
class PortfolioResult:
    symbol_results: Dict[str, BacktestResult]
    combined_equity: pd.Series
    metrics: Dict[str, Any]
    all_trades: List[Trade]


class PortfolioBacktester:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        risk_config: Optional[RiskConfig] = None,
        capital_per_symbol: Optional[float] = None,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_config = risk_config or RiskConfig()
        self.capital_per_symbol = capital_per_symbol

    def run(
        self,
        data_map: Dict[str, pd.DataFrame],
        allow_short: bool = False,
    ) -> PortfolioResult:
        """
        data_map: { "BTC/USDT": df, "ETH/USDT": df, ... }
        """
        n = len(data_map)
        if n == 0:
            raise ValueError("No data provided")

        capital_each = self.capital_per_symbol or (self.initial_capital / n)
        symbol_results: Dict[str, BacktestResult] = {}
        all_trades: List[Trade] = []

        for symbol, df in data_map.items():
            # Fresh strategy instance per symbol to avoid state bleed
            strat = self.strategy.__class__(**self.strategy.params)
            bt = Backtester(
                strategy=strat,
                initial_capital=capital_each,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct,
                risk_config=self.risk_config,
            )
            result = bt.run(df, symbol=symbol, allow_short=allow_short)
            symbol_results[symbol] = result
            all_trades.extend(result.trades)

        # Align equity curves on common index
        equity_dfs = []
        for sym, res in symbol_results.items():
            eq = res.equity_curve.rename(sym)
            equity_dfs.append(eq)

        combined = pd.concat(equity_dfs, axis=1).ffill().fillna(capital_each)
        combined_equity = combined.sum(axis=1)
        combined_equity.name = "portfolio_equity"

        metrics = self._portfolio_metrics(combined_equity, all_trades, capital_each * n)
        return PortfolioResult(
            symbol_results=symbol_results,
            combined_equity=combined_equity,
            metrics=metrics,
            all_trades=all_trades,
        )

    def _portfolio_metrics(
        self, equity: pd.Series, trades: List[Trade], start_capital: float
    ) -> Dict[str, Any]:
        if equity.empty:
            return {}
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / start_capital) - 1
        periods_per_year = 8760
        sharpe = (
            np.sqrt(periods_per_year) * returns.mean() / returns.std()
            if len(returns) > 1 and returns.std() > 0
            else 0.0
        )
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
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
            "profit_factor": profit_factor,
            "symbols": len(set(t.symbol for t in trades)),
        }