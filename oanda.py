"""
OANDA Forex broker adapter (v20 REST API).

Setup:
  1. Create account at https://www.oanda.com
  2. Generate API token (Practice or Live)
  3. Put in .env:
       OANDA_API_TOKEN=your_token
       OANDA_ACCOUNT_ID=your_account_id
       OANDA_ENV=practice   # or live
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Any
import pandas as pd
import requests

from .base import BaseBroker, OrderResult, Position

logger = logging.getLogger(__name__)

# OANDA timeframe mapping
TF_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D", "1w": "W",
}


class OANDABroker(BaseBroker):
    name = "oanda"

    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: str = "practice",   # practice | live
    ):
        self.token = token or os.getenv("OANDA_API_TOKEN", "")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID", "")
        self.environment = (environment or os.getenv("OANDA_ENV", "practice")).lower()

        if self.environment == "live":
            self.base_url = "https://api-fxtrade.oanda.com"
        else:
            self.base_url = "https://api-fxpractice.oanda.com"

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self._connected = False

    def connect(self) -> bool:
        if not self.token or not self.account_id:
            logger.error("OANDA_API_TOKEN and OANDA_ACCOUNT_ID required")
            return False
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/summary",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            self._connected = True
            bal = r.json()["account"].get("balance", "?")
            logger.info(f"OANDA connected ({self.environment}) | balance={bal}")
            return True
        except Exception as e:
            logger.error(f"OANDA connect failed: {e}")
            return False

    def _get(self, path: str, params: dict = None) -> Any:
        r = requests.get(
            f"{self.base_url}{path}", headers=self.headers, params=params, timeout=20
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        r = requests.post(
            f"{self.base_url}{path}", headers=self.headers, json=body, timeout=20
        )
        r.raise_for_status()
        return r.json()

    def get_balance(self, currency: str = "USD") -> float:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return float(data["account"]["balance"])

    def get_positions(self) -> List[Position]:
        data = self._get(f"/v3/accounts/{self.account_id}/openPositions")
        positions = []
        for p in data.get("positions", []):
            instrument = p["instrument"]
            long_units = float(p.get("long", {}).get("units", 0) or 0)
            short_units = float(p.get("short", {}).get("units", 0) or 0)
            size = long_units + short_units   # short_units is already negative
            if abs(size) < 1e-12:
                continue
            avg = float(
                p.get("long", {}).get("averagePrice")
                or p.get("short", {}).get("averagePrice")
                or 0
            )
            upl = float(p.get("unrealizedPL", 0) or 0)
            positions.append(Position(instrument, size, avg, upl))
        return positions

    def get_position(self, symbol: str) -> float:
        # OANDA uses EUR_USD style
        symbol = symbol.replace("/", "_")
        for p in self.get_positions():
            if p.symbol == symbol:
                return p.size
        return 0.0

    def place_market_order(
        self, symbol: str, side: str, amount: float, **kwargs
    ) -> Optional[OrderResult]:
        instrument = symbol.replace("/", "_")
        units = amount if side.lower() == "buy" else -amount
        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)) if units == int(units) else str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        try:
            data = self._post(f"/v3/accounts/{self.account_id}/orders", body)
            fill = data.get("orderFillTransaction") or data.get("orderCreateTransaction") or {}
            return OrderResult(
                id=str(fill.get("id", "")),
                symbol=instrument,
                side=side,
                amount=abs(amount),
                price=float(fill.get("price", 0) or 0),
                status="filled" if "orderFillTransaction" in data else "pending",
                raw=data,
            )
        except Exception as e:
            logger.error(f"OANDA order failed: {e}")
            return None

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float, **kwargs
    ) -> Optional[OrderResult]:
        instrument = symbol.replace("/", "_")
        units = amount if side.lower() == "buy" else -amount
        body = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "timeInForce": "GTC",
            }
        }
        try:
            data = self._post(f"/v3/accounts/{self.account_id}/orders", body)
            tx = data.get("orderCreateTransaction", {})
            return OrderResult(
                id=str(tx.get("id", "")),
                symbol=instrument,
                side=side,
                amount=abs(amount),
                price=price,
                status="pending",
                raw=data,
            )
        except Exception as e:
            logger.error(f"OANDA limit order failed: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        try:
            self._put = getattr(self, "_put", None)
            r = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel",
                headers=self.headers,
                timeout=15,
            )
            return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        instrument = symbol.replace("/", "_")
        granularity = TF_MAP.get(timeframe, "H1")
        params = {"granularity": granularity, "count": min(limit, 5000), "price": "M"}
        if since:
            params["from"] = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        data = self._get(f"/v3/instruments/{instrument}/candles", params)
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": pd.to_datetime(c["time"], utc=True),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("timestamp")
        return df