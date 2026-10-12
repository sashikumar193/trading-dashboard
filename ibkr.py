"""
Interactive Brokers adapter.

Two modes:
  1. ib_insync (recommended) – requires TWS or IB Gateway running
  2. Fallback stub that documents the REST Client Portal API

.env:
  IBKR_HOST=127.0.0.1
  IBKR_PORT=7497          # 7497=paper TWS, 7496=live TWS, 4002=Gateway paper
  IBKR_CLIENT_ID=1
  IBKR_ACCOUNT=           # optional
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
import pandas as pd

from .base import BaseBroker, OrderResult, Position

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Forex, MarketOrder, LimitOrder, util
    IB_INSYNC = True
except ImportError:
    IB_INSYNC = False
    IB = Forex = MarketOrder = LimitOrder = util = None


class IBKRBroker(BaseBroker):
    name = "ibkr"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        account: Optional[str] = None,
    ):
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "1"))
        self.account = account or os.getenv("IBKR_ACCOUNT", "")
        self.ib: Optional[Any] = None
        self._connected = False

    def connect(self) -> bool:
        if not IB_INSYNC:
            logger.error(
                "ib_insync not installed. Run: pip install ib_insync\n"
                "Also start TWS or IB Gateway and enable API connections."
            )
            return False
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"IBKR connected {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"IBKR connect failed: {e}")
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False

    def _fx(self, symbol: str):
        """EUR/USD or EURUSD → Forex contract."""
        s = symbol.replace("/", "").upper()
        if len(s) == 6:
            return Forex(s[:3] + s[3:])
        parts = symbol.replace("_", "/").split("/")
        if len(parts) == 2:
            return Forex(parts[0] + parts[1])
        return Forex(symbol)

    def get_balance(self, currency: str = "USD") -> float:
        if not self.ib:
            return 0.0
        for av in self.ib.accountValues():
            if av.tag == "TotalCashValue" and (av.currency == currency or not currency):
                return float(av.value)
        return 0.0

    def get_positions(self) -> List[Position]:
        if not self.ib:
            return []
        result = []
        for p in self.ib.positions():
            contract = p.contract
            symbol = f"{contract.symbol}/{getattr(contract, 'currency', 'USD')}"
            result.append(Position(
                symbol=symbol,
                size=float(p.position),
                entry_price=float(p.avgCost),
                unrealized_pnl=0.0,
            ))
        return result

    def get_position(self, symbol: str) -> float:
        target = symbol.replace("_", "/").upper()
        for p in self.get_positions():
            if p.symbol.replace("_", "/").upper() == target or \
               p.symbol.replace("/", "") == target.replace("/", ""):
                return p.size
        return 0.0

    def place_market_order(
        self, symbol: str, side: str, amount: float, **kwargs
    ) -> Optional[OrderResult]:
        if not self.ib:
            return None
        contract = self._fx(symbol)
        self.ib.qualifyContracts(contract)
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = MarketOrder(action, abs(amount))
        if self.account:
            order.account = self.account
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1.0)
        return OrderResult(
            id=str(trade.order.orderId),
            symbol=symbol,
            side=side,
            amount=abs(amount),
            price=float(trade.orderStatus.avgFillPrice or 0),
            status=trade.orderStatus.status.lower(),
            raw=trade,
        )

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float, **kwargs
    ) -> Optional[OrderResult]:
        if not self.ib:
            return None
        contract = self._fx(symbol)
        self.ib.qualifyContracts(contract)
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = LimitOrder(action, abs(amount), price)
        if self.account:
            order.account = self.account
        trade = self.ib.placeOrder(contract, order)
        return OrderResult(
            id=str(trade.order.orderId),
            symbol=symbol,
            side=side,
            amount=abs(amount),
            price=price,
            status=trade.orderStatus.status.lower(),
            raw=trade,
        )

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        if not self.ib:
            return False
        for o in self.ib.openOrders():
            if str(o.orderId) == str(order_id):
                self.ib.cancelOrder(o)
                return True
        return False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        if not self.ib:
            return pd.DataFrame()
        contract = self._fx(symbol)
        self.ib.qualifyContracts(contract)

        bar_map = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "1h": "1 hour", "4h": "4 hours", "1d": "1 day",
        }
        bar_size = bar_map.get(timeframe, "1 hour")
        duration = f"{max(limit, 30)} D" if timeframe in ("1h", "4h", "1d") else f"{limit * 5} M"

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
        )
        if not bars:
            return pd.DataFrame()
        df = util.df(bars)
        df = df.rename(columns={"date": "timestamp", "volume": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        return df.tail(limit)