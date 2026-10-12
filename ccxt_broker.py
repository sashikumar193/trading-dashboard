"""
CCXT wrapper that implements BaseBroker (crypto exchanges).
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
import pandas as pd

from .base import BaseBroker, OrderResult, Position

logger = logging.getLogger(__name__)

try:
    import ccxt
    CCXT_OK = True
except ImportError:
    CCXT_OK = False
    ccxt = None


class CCXTBroker(BaseBroker):
    name = "ccxt"

    def __init__(
        self,
        exchange_id: str = "binance",
        sandbox: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        if not CCXT_OK:
            raise ImportError("pip install ccxt")
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        key = api_key or os.getenv(f"{exchange_id.upper()}_API_KEY", "")
        secret = api_secret or os.getenv(f"{exchange_id.upper()}_API_SECRET", "")
        config = {"enableRateLimit": True}
        if key and secret:
            config["apiKey"] = key
            config["secret"] = secret
        self.exchange = getattr(ccxt, exchange_id)(config)
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)
        self._connected = False

    def connect(self) -> bool:
        try:
            self.exchange.load_markets()
            self._connected = True
            logger.info(f"CCXT {self.exchange_id} connected (sandbox={self.sandbox})")
            return True
        except Exception as e:
            logger.error(f"CCXT connect failed: {e}")
            return False

    def get_balance(self, currency: str = "USDT") -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("free", {}).get(currency, 0) or 0)

    def get_positions(self) -> List[Position]:
        # Spot: approximate from balances; futures: fetch_positions
        result = []
        try:
            if self.exchange.has.get("fetchPositions"):
                for p in self.exchange.fetch_positions():
                    size = float(p.get("contracts") or p.get("amount") or 0)
                    if abs(size) < 1e-12:
                        continue
                    result.append(Position(
                        symbol=p["symbol"],
                        size=size if p.get("side") != "short" else -size,
                        entry_price=float(p.get("entryPrice") or 0),
                        unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                    ))
        except Exception:
            pass
        return result

    def get_position(self, symbol: str) -> float:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p.size
        return 0.0

    def place_market_order(self, symbol, side, amount, **kwargs) -> Optional[OrderResult]:
        try:
            o = self.exchange.create_order(symbol, "market", side, amount)
            return OrderResult(
                id=str(o.get("id", "")),
                symbol=symbol,
                side=side,
                amount=amount,
                price=float(o.get("average") or o.get("price") or 0),
                status=o.get("status", "closed"),
                raw=o,
            )
        except Exception as e:
            logger.error(f"CCXT order failed: {e}")
            return None

    def place_limit_order(self, symbol, side, amount, price, **kwargs) -> Optional[OrderResult]:
        try:
            o = self.exchange.create_order(symbol, "limit", side, amount, price)
            return OrderResult(
                id=str(o.get("id", "")),
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                status=o.get("status", "open"),
                raw=o,
            )
        except Exception as e:
            logger.error(f"CCXT limit order failed: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=500) -> pd.DataFrame:
        since_ms = int(since.timestamp() * 1000) if since else None
        candles = self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")