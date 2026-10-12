"""
Trading System — Full Web Dashboard
Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetcher import DataFetcher
from strategies import (
    EMACrossover,
    RSIMeanReversion,
    Breakout,
    BollingerBands,
    MACDCrossover,
    MultiTimeframeEMA,
)
from backtester.engine import Backtester
from backtester.portfolio import PortfolioBacktester
from risk.manager import RiskConfig

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Algo Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #2d2d44;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2f;
        border-radius: 8px;
        padding: 8px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Algo Trading System</p>', unsafe_allow_html=True)
st.caption("Backtest • Portfolio • Strategies • Risk Controls  |  Crypto & Forex ready")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
STRATEGY_OPTIONS = {
    "EMA Crossover": EMACrossover,
    "RSI Mean-Reversion": RSIMeanReversion,
    "Breakout (Donchian)": Breakout,
    "Bollinger Bands": BollingerBands,
    "MACD Crossover": MACDCrossover,
    "Multi-Timeframe EMA": MultiTimeframeEMA,
}

def make_strategy(name: str, params: dict):
    cls = STRATEGY_OPTIONS[name]
    # Only pass accepted kwargs
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in params.items() if k in sig.parameters}
    return cls(**valid)

def run_single_backtest(strategy, df, symbol, capital, commission, allow_short, risk_cfg):
    bt = Backtester(
        strategy=strategy,
        initial_capital=capital,
        commission_pct=commission,
        risk_config=risk_cfg,
    )
    return bt.run(df, symbol=symbol, allow_short=allow_short)

def equity_fig(df, result, symbol, strategy_name):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=(f"{symbol} — Price", "Equity Curve"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Entry markers
    for t in result.trades:
        color = "#26a69a" if t.side == "long" else "#ef5350"
        fig.add_vline(x=t.entry_time, line_dash="dot", line_color=color, opacity=0.35, row=1, col=1)
        if t.exit_time:
            fig.add_vline(x=t.exit_time, line_dash="dot", line_color="#888", opacity=0.25, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve,
        name="Equity", line=dict(color="#7b2ff7", width=2),
        fill="tozeroy", fillcolor="rgba(123,47,247,0.12)",
    ), row=2, col=1)

    fig.update_layout(
        height=680, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig

def metrics_row(m: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Return", f"{m.get('total_return_pct', 0):.2f}%")
    c2.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
    c3.metric("Max DD", f"{m.get('max_drawdown_pct', 0):.2f}%")
    c4.metric("Trades", m.get("num_trades", 0))
    c5.metric("Win Rate", f"{m.get('win_rate_pct', 0):.1f}%")
    c6.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Mode", ["Single Backtest", "Portfolio Backtest", "Strategy Info"], index=0)

    st.divider()
    strategy_name = st.selectbox("Strategy", list(STRATEGY_OPTIONS.keys()))

    with st.expander("Strategy Parameters", expanded=True):
        params = {}
        if "EMA" in strategy_name or "Multi" in strategy_name:
            params["fast_period"] = st.slider("Fast EMA", 5, 50, 12)
            params["slow_period"] = st.slider("Slow EMA", 10, 200, 26)
            if "Multi" in strategy_name:
                params["higher_tf_factor"] = st.slider("Higher TF factor", 2, 12, 4)
        elif "RSI" in strategy_name:
            params["rsi_period"] = st.slider("RSI Period", 5, 30, 14)
            params["oversold"] = st.slider("Oversold", 10, 40, 30)
            params["overbought"] = st.slider("Overbought", 60, 90, 70)
        elif "Breakout" in strategy_name:
            params["lookback"] = st.slider("Lookback", 5, 100, 20)
        elif "Bollinger" in strategy_name:
            params["period"] = st.slider("Period", 10, 50, 20)
            params["std_dev"] = st.slider("Std Dev", 1.0, 3.5, 2.0, 0.1)
        elif "MACD" in strategy_name:
            params["fast"] = st.slider("Fast", 5, 20, 12)
            params["slow"] = st.slider("Slow", 15, 50, 26)
            params["signal"] = st.slider("Signal", 5, 20, 9)

    st.divider()
    st.subheader("Data")
    use_sample = st.checkbox("Synthetic sample data (offline)", value=True)
    symbol = st.text_input("Symbol", "BTC/USDT")
    symbols_portfolio = st.text_input("Portfolio symbols (comma)", "BTC/USDT,ETH/USDT,SOL/USDT")
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    days = st.slider("Lookback (days)", 30, 730, 180)

    st.divider()
    st.subheader("Capital & Risk")
    capital = st.number_input("Initial Capital ($)", 1000, 1_000_000, 10_000, 500)
    commission = st.number_input("Commission %", 0.0, 1.0, 0.10, 0.01) / 100
    allow_short = st.checkbox("Allow shorting", False)
    max_pos_pct = st.slider("Max position %", 1, 50, 10) / 100
    max_dd = st.slider("Max drawdown kill %", 5, 50, 15) / 100

    risk_cfg = RiskConfig(
        max_position_pct=max_pos_pct,
        max_drawdown_pct=max_dd,
        max_daily_loss_pct=0.03,
        risk_per_trade_pct=0.01,
    )

    run = st.button("🚀 Run", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
fetcher = DataFetcher()

if mode == "Strategy Info":
    st.subheader("📚 Available Strategies")
    info = [
        ("EMA Crossover", "Trend following", "Buy when fast EMA crosses above slow EMA"),
        ("RSI Mean-Reversion", "Mean reversion", "Buy oversold (<30), sell overbought (>70)"),
        ("Breakout (Donchian)", "Breakout", "Buy break of N-bar high, sell break of N-bar low"),
        ("Bollinger Bands", "Mean reversion", "Buy at lower band, sell at upper band"),
        ("MACD Crossover", "Momentum", "Buy MACD line cross above signal, sell cross below"),
        ("Multi-Timeframe EMA", "Trend + filter", "EMA cross only when higher TF trend is bullish"),
    ]
    for name, style, desc in info:
        with st.expander(f"**{name}**  ·  {style}"):
            st.write(desc)
            st.code(f'python main.py backtest --strategy {name.split()[0].lower()} --use-sample', language="bash")

    st.subheader("How to extend")
    st.markdown("""
1. Create a new file in `strategies/` inheriting from `BaseStrategy`
2. Implement `generate_signals(df) → pd.Series` with values `1 / -1 / 0`
3. Register it in `main.py` → `STRATEGY_MAP` and in this dashboard
    """)

elif run:
    strategy = make_strategy(strategy_name, params)

    if mode == "Single Backtest":
        with st.spinner("Fetching data & running backtest…"):
            if use_sample:
                df = fetcher.generate_sample_data(n_bars=min(days * 24, 3000), start_price=50000)
                symbol_used = "SAMPLE/USDT"
            else:
                try:
                    since = datetime.utcnow() - timedelta(days=days)
                    df = fetcher.fetch_ohlcv("binance", symbol, timeframe=timeframe, since=since)
                    symbol_used = symbol
                except Exception as e:
                    st.warning(f"Live data failed ({e}). Using sample data.")
                    df = fetcher.generate_sample_data(n_bars=min(days * 24, 3000))
                    symbol_used = "SAMPLE/USDT"

            result = run_single_backtest(
                strategy, df, symbol_used, capital, commission, allow_short, risk_cfg
            )

        st.success(f"Backtest complete — **{strategy.name}** on **{symbol_used}**")
        metrics_row(result.metrics)

        tab1, tab2, tab3 = st.tabs(["📊 Chart", "📋 Trades", "📈 Equity Table"])
        with tab1:
            st.plotly_chart(equity_fig(df, result, symbol_used, strategy.name), use_container_width=True)
        with tab2:
            if result.trades:
                tdf = pd.DataFrame([{
                    "Entry": t.entry_time,
                    "Exit": t.exit_time,
                    "Side": t.side,
                    "Entry $": round(t.entry_price, 4),
                    "Exit $": round(t.exit_price or 0, 4),
                    "Size": round(t.size, 6),
                    "PnL $": round(t.pnl, 2),
                    "PnL %": round(t.pnl_pct * 100, 2),
                    "Commission": round(t.commission, 2),
                } for t in result.trades])
                st.dataframe(tdf, use_container_width=True, height=400)
            else:
                st.info("No trades generated.")
        with tab3:
            eq = result.equity_curve.reset_index()
            eq.columns = ["Time", "Equity"]
            st.dataframe(eq.tail(200), use_container_width=True)

    else:  # Portfolio
        syms = [s.strip() for s in symbols_portfolio.split(",") if s.strip()]
        with st.spinner(f"Running portfolio backtest on {len(syms)} symbols…"):
            data_map: Dict[str, pd.DataFrame] = {}
            for i, sym in enumerate(syms):
                if use_sample:
                    data_map[sym] = fetcher.generate_sample_data(
                        n_bars=min(days * 24, 2000),
                        start_price=20000 + (i + 1) * 8000,
                        seed=42 + i * 17,
                    )
                else:
                    try:
                        since = datetime.utcnow() - timedelta(days=days)
                        data_map[sym] = fetcher.fetch_ohlcv(
                            "binance", sym, timeframe=timeframe, since=since
                        )
                    except Exception as e:
                        st.warning(f"{sym}: {e}")
                        data_map[sym] = fetcher.generate_sample_data(
                            n_bars=min(days * 24, 2000), seed=42 + i * 17
                        )

            pbt = PortfolioBacktester(
                strategy=strategy,
                initial_capital=capital,
                commission_pct=commission,
                risk_config=risk_cfg,
            )
            presult = pbt.run(data_map, allow_short=allow_short)

        st.success(f"Portfolio backtest complete — **{strategy.name}** · {len(syms)} symbols")
        metrics_row(presult.metrics)

        # Combined equity
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=presult.combined_equity.index,
            y=presult.combined_equity,
            name="Portfolio Equity",
            line=dict(color="#7b2ff7", width=2.5),
            fill="tozeroy", fillcolor="rgba(123,47,247,0.12)",
        ))
        for sym, res in presult.symbol_results.items():
            fig.add_trace(go.Scatter(
                x=res.equity_curve.index, y=res.equity_curve,
                name=sym, line=dict(width=1.2), opacity=0.7,
            ))
        fig.update_layout(
            height=480, template="plotly_dark",
            title="Portfolio & Per-Symbol Equity",
            margin=dict(l=40, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-symbol metrics
        st.subheader("Per-Symbol Results")
        rows = []
        for sym, res in presult.symbol_results.items():
            m = res.metrics
            rows.append({
                "Symbol": sym,
                "Return %": round(m.get("total_return_pct", 0), 2),
                "Sharpe": round(m.get("sharpe_ratio", 0), 2),
                "Max DD %": round(m.get("max_drawdown_pct", 0), 2),
                "Trades": m.get("num_trades", 0),
                "Win %": round(m.get("win_rate_pct", 0), 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    # Landing / empty state
    st.info("👈 Configure settings in the sidebar and click **Run**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Single Backtest")
        st.write("Test one strategy on one symbol with full metrics and trade list.")
    with col2:
        st.markdown("### 📁 Portfolio")
        st.write("Run the same strategy across multiple symbols and see combined equity.")
    with col3:
        st.markdown("### 📖 Strategy Info")
        st.write("Learn how each strategy works and how to add your own.")

    st.divider()
    st.markdown("""
    **CLI alternative**
    ```bash
    python main.py backtest --strategy ema --use-sample
    python main.py portfolio --strategy breakout --symbols BTC/USDT,ETH/USDT --use-sample
    ```
    """)
