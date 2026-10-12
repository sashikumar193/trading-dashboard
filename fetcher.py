"""
Data fetching utilities.
Supports CCXT (crypto) and simple CSV / Yahoo-style fallbacks.
"""

import os
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None


class DataFetcher:
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{exchange}_{safe_symbol}_{timeframe}.parquet"

    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV from any CCXT-supported exchange.
        Returns DataFrame with columns: open, high, low, close, volume
        Index is datetime (UTC).
        """
        if not CCXT_AVAILABLE:
            raise ImportError(
                "ccxt is required for live market data. "
                "Install with: pip install ccxt\n"
                "Or use --use-sample for offline testing."
            )

        cache_file = self._cache_path(exchange_id, symbol, timeframe)

        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            if not df.empty and (datetime.utcnow() - df.index[-1].to_pydatetime()).total_seconds() < 3600:
                return df

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        since_ms = None
        if since:
            since_ms = int(since.timestamp() * 1000)

        all_candles = []
        while True:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=limit
            )
            if not candles:
                break
            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if since_ms and last_ts <= since_ms:
                break
            since_ms = last_ts + 1
            if len(all_candles) > 20000:
                break
            time.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            raise ValueError(f"No data returned for {symbol} on {exchange_id}")

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        df.to_parquet(cache_file)
        return df

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load a standard OHLCV CSV (must have datetime + OHLCV columns)."""
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(c.lower() for c in df.columns)):
            df.columns = [c.lower() for c in df.columns]
        return df

    def generate_sample_data(
        self,
        n_bars: int = 1000,
        start_price: float = 50000.0,
        volatility: float = 0.015,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV for offline testing when no internet / keys.
        """
        import numpy as np

        np.random.seed(seed)
        dates = pd.date_range(end=datetime.utcnow(), periods=n_bars, freq="1h", tz="UTC")
        returns = np.random.normal(0, volatility, n_bars)
        close = start_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.normal(0, volatility / 2, n_bars)))
        low = close * (1 - np.abs(np.random.normal(0, volatility / 2, n_bars)))
        open_ = np.roll(close, 1)
        open_[0] = start_price
        volume = np.random.lognormal(10, 1, n_bars)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        return df
