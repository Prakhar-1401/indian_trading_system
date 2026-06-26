"""
Stock Screener Module — Custom filter builder for scanning stocks.

Features:
- Filter by RSI, MACD, Moving Averages, Volume, Sector, Market Cap
- Preset screeners (Oversold Bounce, Momentum Breakout, etc.)
- Results table with key metrics
- Save/load custom presets
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import json

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PRESETS_FILE = DATA_DIR / "screener_presets.json"

# Universe to scan
SCREEN_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "BAJFINANCE", "SUNPHARMA", "TITAN", "MARUTI", "WIPRO",
    "HCLTECH", "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
    "AXISBANK", "KOTAKBANK", "ULTRACEMCO", "ONGC", "TECHM", "ADANIENT",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "CIPLA", "DRREDDY",
    "BAJAJFINSV", "EICHERMOT", "DIVISLAB", "APOLLOHOSP", "TATACONSUM",
    "HEROMOTOCO", "SBILIFE", "BRITANNIA", "PIDILITIND", "GODREJCP",
    "HDFCLIFE", "DABUR", "HAVELLS", "INDUSINDBK", "BANKBARODA", "IOC",
    "BPCL", "VEDL",
]

BUILTIN_PRESETS = {
    "Oversold Bounce": {"rsi_max": 35, "price_above_sma50": False, "description": "RSI < 35, potential reversal"},
    "Momentum Breakout": {"rsi_min": 55, "rsi_max": 75, "price_above_sma50": True,
                          "description": "RSI 55-75, above 50-SMA, trend continuation"},
    "Volume Spike": {"vol_spike_min": 2.0, "description": "Volume > 2x 20-day average"},
    "Golden Cross": {"sma50_above_sma200": True, "description": "50-SMA crossed above 200-SMA"},
    "Undervalued Large Cap": {"rsi_max": 50, "price_above_sma200": True,
                              "description": "RSI<50 but above 200-SMA, dip-buy opportunity"},
}


def compute_screen_metrics(symbol: str) -> dict:
    """Compute screening metrics for a single stock."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="1y")
        df = df[df['Close'].notna()]
        if df.empty or len(df) < 50:
            return None

        close = df['Close'].values
        volume = df['Volume'].values
        n = len(close)

        # RSI
        deltas = pd.Series(close).diff()
        gain = deltas.where(deltas > 0, 0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-deltas.where(deltas < 0, 0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50

        # Moving averages
        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
        sma200 = pd.Series(close).rolling(200).mean().iloc[-1] if n >= 200 else np.nan

        # Volume
        avg_vol_20 = pd.Series(volume).rolling(20).mean().iloc[-1]
        vol_ratio = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26

        # Price change
        pct_1d = (close[-1] / close[-2] - 1) * 100 if n >= 2 else 0
        pct_1w = (close[-1] / close[-5] - 1) * 100 if n >= 5 else 0
        pct_1m = (close[-1] / close[-22] - 1) * 100 if n >= 22 else 0

        return {
            'symbol': symbol,
            'price': close[-1],
            'rsi': rsi,
            'sma20': sma20,
            'sma50': sma50,
            'sma200': sma200,
            'macd': macd,
            'vol_ratio': vol_ratio,
            'pct_1d': pct_1d,
            'pct_1w': pct_1w,
            'pct_1m': pct_1m,
            'price_above_sma50': close[-1] > sma50,
            'price_above_sma200': close[-1] > sma200 if not np.isnan(sma200) else False,
            'sma50_above_sma200': sma50 > sma200 if not np.isnan(sma200) else False,
        }
    except Exception:
        return None


def apply_filters(metrics: dict, filters: dict) -> bool:
    """Check if a stock passes all filters."""
    if metrics is None:
        return False

    if 'rsi_min' in filters and metrics['rsi'] < filters['rsi_min']:
        return False
    if 'rsi_max' in filters and metrics['rsi'] > filters['rsi_max']:
        return False
    if 'price_above_sma50' in filters and metrics['price_above_sma50'] != filters['price_above_sma50']:
        return False
    if 'price_above_sma200' in filters and metrics['price_above_sma200'] != filters['price_above_sma200']:
        return False
    if 'sma50_above_sma200' in filters and metrics['sma50_above_sma200'] != filters['sma50_above_sma200']:
        return False
    if 'vol_spike_min' in filters and metrics['vol_ratio'] < filters['vol_spike_min']:
        return False
    if 'macd_positive' in filters and filters['macd_positive'] and metrics['macd'] <= 0:
        return False

    return True


def render_screener():
    """Render the stock screener interface."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            🔎 STOCK SCREENER
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
            Filter 50 stocks by technicals │ Custom & preset screens
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Preset buttons
    st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;margin-bottom:4px;">
        QUICK PRESETS</div>""", unsafe_allow_html=True)

    preset_cols = st.columns(len(BUILTIN_PRESETS))
    selected_preset = None
    for i, (name, preset) in enumerate(BUILTIN_PRESETS.items()):
        with preset_cols[i]:
            if st.button(name, key=f"preset_{i}", use_container_width=True,
                        help=preset.get('description', '')):
                selected_preset = preset

    st.markdown('<hr style="margin:8px 0;border-color:#1a2332;">', unsafe_allow_html=True)

    # Custom filters
    with st.expander("🛠️ CUSTOM FILTERS", expanded=selected_preset is None):
        f_cols = st.columns(4)
        with f_cols[0]:
            rsi_range = st.slider("RSI Range", 0, 100, (20, 80), key="scr_rsi")
        with f_cols[1]:
            above_sma50 = st.checkbox("Price > 50-SMA", value=False, key="scr_sma50")
            above_sma200 = st.checkbox("Price > 200-SMA", value=False, key="scr_sma200")
        with f_cols[2]:
            golden_cross = st.checkbox("Golden Cross (50>200)", value=False, key="scr_golden")
            macd_pos = st.checkbox("MACD Positive", value=False, key="scr_macd")
        with f_cols[3]:
            vol_spike = st.number_input("Min Volume Ratio", value=1.0, min_value=0.5,
                                        max_value=10.0, step=0.5, key="scr_vol")

    # Build filters
    if selected_preset:
        filters = {k: v for k, v in selected_preset.items() if k != 'description'}
    else:
        filters = {}
        if rsi_range[0] > 0:
            filters['rsi_min'] = rsi_range[0]
        if rsi_range[1] < 100:
            filters['rsi_max'] = rsi_range[1]
        if above_sma50:
            filters['price_above_sma50'] = True
        if above_sma200:
            filters['price_above_sma200'] = True
        if golden_cross:
            filters['sma50_above_sma200'] = True
        if macd_pos:
            filters['macd_positive'] = True
        if vol_spike > 1.0:
            filters['vol_spike_min'] = vol_spike

    # Run screener
    if st.button("🚀 RUN SCREENER", key="run_screener", use_container_width=True):
        progress = st.progress(0, text="Scanning stocks...")
        results = []

        for i, sym in enumerate(SCREEN_UNIVERSE):
            progress.progress((i + 1) / len(SCREEN_UNIVERSE), text=f"Scanning {sym}...")
            metrics = compute_screen_metrics(sym)
            if apply_filters(metrics, filters):
                results.append(metrics)

        progress.empty()

        if not results:
            st.warning("No stocks matched your filters. Try relaxing criteria.")
        else:
            st.success(f"✅ {len(results)} stocks matched")

            # Results table
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('rsi', ascending=True)

            # Format for display
            display_df = results_df[['symbol', 'price', 'rsi', 'pct_1d', 'pct_1w',
                                     'pct_1m', 'vol_ratio', 'macd']].copy()
            display_df.columns = ['Symbol', 'Price', 'RSI', '1D%', '1W%', '1M%', 'Vol Ratio', 'MACD']
            display_df['Price'] = display_df['Price'].apply(lambda x: f"₹{x:,.2f}")
            display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
            display_df['1D%'] = display_df['1D%'].apply(lambda x: f"{x:+.2f}%")
            display_df['1W%'] = display_df['1W%'].apply(lambda x: f"{x:+.2f}%")
            display_df['1M%'] = display_df['1M%'].apply(lambda x: f"{x:+.2f}%")
            display_df['Vol Ratio'] = display_df['Vol Ratio'].apply(lambda x: f"{x:.1f}x")
            display_df['MACD'] = display_df['MACD'].apply(lambda x: f"{x:+.2f}")

            st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)

            # Store in session for quick access
            st.session_state['screener_results'] = results
