"""
Abstract broker interface.
All brokers (crypto CCXT, OANDA, IBKR) implement this so strategies stay the same.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    status: str
    raw: Any = None


@dataclass
class Position:
    symbol: str
    size: float          # positive = long, negative = short
    entry_price: float
    unrealized_pnl: float = 0.0


class BaseBroker(ABC):
    """Unified interface for crypto & forex brokers."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection / validate credentials."""
        pass

    @abstractmethod
    def get_balance(self, currency: str = "USD") -> float:
        """Free balance in the given currency."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Signed size for one symbol (0 if flat)."""
        pass

    @abstractmethod
    def place_market_order(
        self, symbol: str, side: str, amount: float, **kwargs
    ) -> Optional[OrderResult]:
        """side = 'buy' | 'sell'"""
        pass

    @abstractmethod
    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float, **kwargs
    ) -> Optional[OrderResult]:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ):
        """Return pandas DataFrame with OHLCV, datetime index."""
        pass

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        size = self.get_position(symbol)
        if abs(size) < 1e-12:
            return None
        side = "sell" if size > 0 else "buy"
        return self.place_market_order(symbol, side, abs(size))