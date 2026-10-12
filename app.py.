import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="TradePulse - Algo Backtester & Bot", layout="wide")

st.title("⚡ TradePulse: Algo Trading & Backtesting Terminal")
st.caption("Crypto & Forex Strategy Backtester + Automated Bot Dashboard")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
market_type = st.sidebar.selectbox("Market Type", ["Crypto (Binance / Bybit)", "Forex (MT5)"])
symbol = st.sidebar.text_input("Asset Symbol", "BTC/USDT" if market_type.startswith("Crypto") else "EURUSD")
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1D"])

initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000.0, step=500.0)
fast_ema = st.sidebar.slider("Fast EMA Period", 5, 50, 9)
slow_ema = st.sidebar.slider("Slow EMA Period", 10, 200, 21)

bot_status = st.sidebar.toggle("🟢 Activate Live Bot Execution", value=False)

# --- DUMMY / REAL DATA LOADER ---
@st.cache_data
def load_market_data(n_bars=300):
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_bars, freq="h")
    returns = np.random.normal(0.0005, 0.015, n_bars)
    price = 60000.0 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price * (1 - np.random.uniform(0, 0.005, n_bars)),
        "high": price * (1 + np.random.uniform(0, 0.008, n_bars)),
        "low": price * (1 - np.random.uniform(0, 0.008, n_bars)),
        "close": price,
        "volume": np.random.randint(100, 5000, n_bars)
    }).set_index("timestamp")
    return df

data = load_market_data()

# Strategy Calculations
data["ema_fast"] = data["close"].ewm(span=fast_ema, adjust=False).mean()
data["ema_slow"] = data["close"].ewm(span=slow_ema, adjust=False).mean()
data["signal"] = 0
data.loc[data["ema_fast"] > data["ema_slow"], "signal"] = 1
data.loc[data["ema_fast"] < data["ema_slow"], "signal"] = -1

# Performance Backtest
data["position"] = data["signal"].shift(1).fillna(0)
data["market_return"] = data["close"].pct_change().fillna(0)
data["strategy_return"] = data["position"] * data["market_return"]
data["equity"] = (1 + data["strategy_return"]).cumprod() * initial_capital

# --- METRIC CARDS ---
total_return = ((data["equity"].iloc[-1] / initial_capital) - 1) * 100
current_signal = "BUY (Long)" if data["signal"].iloc[-1] == 1 else "SELL (Short)"
peak = data["equity"].cummax()
max_dd = (((data["equity"] - peak) / peak).min()) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Capital", f"${data['equity'].iloc[-1]:,.2f}", f"{total_return:.2f}%")
col2.metric("Max Drawdown", f"{max_dd:.2f}%")
col3.metric("Current Signal", current_signal)
col4.metric("Bot Status", "ONLINE (Trading)" if bot_status else "IDLE (Simulation)")

# --- CHARTS SECTION ---
st.subheader("📈 Price Action & EMA Indicators")
st.line_chart(data[["close", "ema_fast", "ema_slow"]])

st.subheader("💰 Equity Curve Growth")
st.area_chart(data["equity"])

# --- TRADE LOG PREVIEW ---
with st.expander("📋 Recent Execution Logs"):
    trades = data[data["signal"].diff() != 0][["close", "signal"]].tail(10)
    trades["Action"] = trades["signal"].map({1: "LONG ENTRY", -1: "SHORT ENTRY", 0: "EXIT"})
    st.dataframe(trades[["Action", "close"]], use_container_width=True)
