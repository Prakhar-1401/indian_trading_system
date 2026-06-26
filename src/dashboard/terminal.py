"""
WAR ROOM TERMINAL — Real-Time Trading Intelligence
====================================================
Bloomberg-style multi-panel dark terminal for Indian markets.
Single screen. All intelligence. No tab-switching.

RUN:
    streamlit run src/dashboard/terminal.py

LAYOUT:
    ┌─────────────────────────────────────────────────────────────────┐
    │  TOP BAR: WAR ROOM TERMINAL | NSE LIVE | Search | Clock        │
    ├────────────┬──────────────────────────────┬─────────────────────┤
    │ MARKET OVW │  TECHNICAL CHART             │ ORDER BOOK          │
    │ NIFTY 50   │  Candlestick + Indicators    │ Bid/Ask Depth       │
    │ SENSEX     │  Timeframe: 1D/1W/1M         │                     │
    │ BANK NIFTY │                               │ TRADING PANEL       │
    │            │                               │ Buy/Sell/Execute    │
    │ NEWS FEED  │  GEO INTELLIGENCE             │                     │
    ├────────────┴──────────────────────────────┴─────────────────────┤
    │  STRATEGIES PANEL              │  SECTOR HEATMAP                │
    └────────────────────────────────┴────────────────────────────────┘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import re
import time
import random
import requests
import yfinance as yf
from urllib.parse import quote
from datetime import datetime, timedelta
from pathlib import Path

# Project root
ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.data.fetcher import DataManager
from src.indicators.technical import (
    get_all_indicators, calculate_rsi, calculate_macd,
    calculate_moving_averages, calculate_bollinger_bands,
    calculate_atr, calculate_volume_signals
)
from src.utils.helpers import load_config

# ═══════════════════════════════════════════════════════════════════
# NSE UNIVERSE — Searchable stock list
# ═══════════════════════════════════════════════════════════════════
NSE_STOCKS = {
    # NIFTY 50
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank", "INFY": "Infosys", "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel", "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro",
    "HCLTECH": "HCL Technologies", "AXISBANK": "Axis Bank",
    "BAJFINANCE": "Bajaj Finance", "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma", "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement", "NTPC": "NTPC Limited",
    "WIPRO": "Wipro Limited", "ONGC": "ONGC Limited",
    "JSWSTEEL": "JSW Steel", "POWERGRID": "Power Grid Corp",
    "TATASTEEL": "Tata Steel", "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports", "TECHM": "Tech Mahindra",
    "NESTLEIND": "Nestle India", "BAJAJFINSV": "Bajaj Finserv",
    "GRASIM": "Grasim Industries", "COALINDIA": "Coal India",
    "CIPLA": "Cipla Limited", "HINDALCO": "Hindalco Industries",
    "APOLLOHOSP": "Apollo Hospitals", "DRREDDY": "Dr Reddy's Labs",
    "DIVISLAB": "Divi's Laboratories", "EICHERMOT": "Eicher Motors",
    "BPCL": "Bharat Petroleum", "HEROMOTOCO": "Hero MotoCorp",
    "BRITANNIA": "Britannia Industries", "INDUSINDBK": "IndusInd Bank",
    "TATACONSUM": "Tata Consumer", "SBILIFE": "SBI Life Insurance",
    "HDFCLIFE": "HDFC Life Insurance", "BAJAJ-AUTO": "Bajaj Auto",
    "ASIANPAINT": "Asian Paints", "M&M": "Mahindra & Mahindra",
    "TATAMOTORS": "Tata Motors", "SHRIRAMFIN": "Shriram Finance",
    # NIFTY NEXT 50
    "BANKBARODA": "Bank of Baroda", "VEDL": "Vedanta Limited",
    "GODREJCP": "Godrej Consumer", "HAVELLS": "Havells India",
    "PIDILITIND": "Pidilite Industries", "DLF": "DLF Limited",
    "SIEMENS": "Siemens India", "ABB": "ABB India",
    "TRENT": "Trent Limited", "ZOMATO": "Zomato Limited",
    "POLYCAB": "Polycab India", "IOC": "Indian Oil Corp",
    "AMBUJACEM": "Ambuja Cements", "COLPAL": "Colgate-Palmolive",
    "BERGEPAINT": "Berger Paints", "MCDOWELL-N": "United Spirits",
    "NAUKRI": "Info Edge (Naukri)", "ADANIGREEN": "Adani Green",
    "TORNTPHARM": "Torrent Pharma", "INDUSTOWER": "Indus Towers",
    "MAXHEALTH": "Max Healthcare", "JUBLFOOD": "Jubilant FoodWorks",
    "DMART": "Avenue Supermarts", "IRCTC": "IRCTC",
    "PFC": "Power Finance Corp", "RECLTD": "REC Limited",
    "TATAPOWER": "Tata Power", "CANBK": "Canara Bank",
    "PNB": "Punjab National Bank", "IDEA": "Vodafone Idea",
    # Popular Mid/Small Caps
    "DEEPAKNTR": "Deepak Nitrite", "AUROPHARMA": "Aurobindo Pharma",
    "MPHASIS": "Mphasis", "PERSISTENT": "Persistent Systems",
    "LTTS": "L&T Technology", "COFORGE": "Coforge Limited",
    "VOLTAS": "Voltas Limited", "PAGEIND": "Page Industries",
    "BALKRISIND": "Balkrishna Industries", "FLUOROCHEM": "Gujarat Fluorochem",
    "LALPATHLAB": "Dr Lal PathLabs", "SYNGENE": "Syngene International",
    "HAL": "Hindustan Aeronautics", "BEL": "Bharat Electronics",
    "IRFC": "Indian Railway Finance", "SAIL": "Steel Authority",
    "BHEL": "BHEL", "GAIL": "GAIL India",
    "NATIONALUM": "National Aluminium", "NMDC": "NMDC Limited",
    "CONCOR": "Container Corp", "CDSL": "CDSL",
    "MARICO": "Marico Limited", "DABUR": "Dabur India",
    "MOTHERSON": "Motherson Sumi", "BOSCHLTD": "Bosch Limited",
    "MUTHOOTFIN": "Muthoot Finance", "MANAPPURAM": "Manappuram Finance",
    "IDFCFIRSTB": "IDFC First Bank", "FEDERALBNK": "Federal Bank",
    "BANDHANBNK": "Bandhan Bank", "RBLBANK": "RBL Bank",
    "LTIM": "LTIMindtree", "ZYDUSLIFE": "Zydus Lifesciences",
}

# Sector mapping for heatmap
SECTOR_MAP = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "FEDERALBNK"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "MUTHOOTFIN", "SHRIRAMFIN", "PFC", "RECLTD"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "TORNTPHARM", "ZYDUSLIFE"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC", "POWERGRID", "TATAPOWER", "COALINDIA", "GAIL"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "NMDC", "NATIONALUM"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "MARICO", "DABUR", "COLPAL", "GODREJCP"],
    "Infra": ["LT", "ADANIENT", "ADANIPORTS", "DLF", "ULTRACEMCO", "AMBUJACEM", "GRASIM"],
    "Defence": ["HAL", "BEL"],
    "Telecom": ["BHARTIARTL", "IDEA", "INDUSTOWER"],
}

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG + DARK THEME CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WAR ROOM TERMINAL",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    /* === DARK TERMINAL CORE === */
    .stApp { background-color: #0a0e17; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; max-width: 100%; }

    /* Hide Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    .stDeployButton { display: none; }

    /* Keep the sidebar reopen ("›") control visible and clickable when collapsed */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
        z-index: 999999 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        visibility: visible !important;
        display: inline-flex !important;
        background: #131a2a !important;
        border: 1px solid #2a3550 !important;
        border-radius: 6px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        color: #4a9eff !important;
        fill: #4a9eff !important;
    }

    /* === TYPOGRAPHY === */
    * { font-family: 'Inter', -apple-system, sans-serif; }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b7394;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* === TOP BAR === */
    .top-bar {
        background: linear-gradient(180deg, #0d1220 0%, #0a0e17 100%);
        border-bottom: 1px solid #1a2332;
        padding: 8px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .brand-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e74c3c;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(231, 76, 60, 0.3);
    }
    .market-badge {
        display: inline-block;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        color: #00ff88;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .market-badge-closed {
        display: inline-block;
        background: rgba(255, 68, 68, 0.1);
        border: 1px solid rgba(255, 68, 68, 0.3);
        color: #ff4444;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .clock-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8892b0;
    }

    /* === PANEL CARDS === */
    .panel-card {
        background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
        border: 1px solid #1a2332;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 3px 0;
    }
    .panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #4a9eff;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #1a2332;
        padding-bottom: 6px;
        margin-bottom: 8px;
    }

    /* === INDEX CARDS === */
    .index-card {
        background: #0d1220;
        border: 1px solid #1a2332;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 4px 0;
        transition: border-color 0.2s;
    }
    .index-card:hover { border-color: #4a9eff; }
    .index-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6b7394;
        letter-spacing: 1px;
    }
    .index-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .index-change-up {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #00ff88;
        font-weight: 600;
    }
    .index-change-down {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #ff4444;
        font-weight: 600;
    }

    /* === NEWS FEED === */
    .news-item {
        padding: 6px 0;
        border-bottom: 1px solid #1a2332;
    }
    .news-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #4a5568;
    }
    .news-text {
        font-size: 0.75rem;
        color: #a0aec0;
        line-height: 1.3;
    }
    .tag-bull {
        display: inline-block;
        background: rgba(0, 255, 136, 0.15);
        color: #00ff88;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .tag-bear {
        display: inline-block;
        background: rgba(255, 68, 68, 0.15);
        color: #ff4444;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .tag-neutral {
        display: inline-block;
        background: rgba(255, 170, 0, 0.15);
        color: #ffaa00;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* === ORDER BOOK === */
    .ob-row {
        display: flex;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 1px 0;
    }
    .ob-bid { color: #00ff88; }
    .ob-ask { color: #ff4444; }
    .ob-qty { color: #8892b0; width: 60px; text-align: right; }
    .ob-price { width: 80px; text-align: right; font-weight: 600; }

    /* === STRATEGY CARDS === */
    .strategy-card {
        background: #0d1220;
        border: 1px solid #1a2332;
        border-radius: 6px;
        padding: 10px;
        margin: 4px 0;
    }
    .signal-buy {
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .signal-sell {
        color: #ff4444;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .signal-hold {
        color: #ffaa00;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .strategy-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #8892b0;
    }
    .strategy-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* === HEATMAP CELL === */
    .heat-cell {
        border-radius: 4px;
        padding: 6px 4px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        margin: 2px;
        min-width: 60px;
        display: inline-block;
    }

    /* === SELECTBOX / INPUT STYLING === */
    .stSelectbox, .stTextInput, .stNumberInput {
        font-family: 'JetBrains Mono', monospace;
    }
    .stSelectbox > div > div {
        background-color: #0d1220;
        border-color: #1a2332;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0a0e17;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 1px;
        padding: 6px 14px;
        color: #6b7394;
    }
    .stTabs [aria-selected="true"] {
        color: #4a9eff;
        border-bottom-color: #4a9eff;
    }

    /* === BUTTON === */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        border-radius: 4px;
    }

    /* === DIVIDER === */
    hr { border-color: #1a2332; margin: 0.3rem 0; }

    /* Scrollable containers */
    .scrollable { max-height: 300px; overflow-y: auto; }

    /* === TICKER STRIP (scrolling) === */
    .ticker-strip {
        background: #080c14;
        border-bottom: 1px solid #1a2332;
        border-top: 1px solid #1a2332;
        overflow: hidden;
        white-space: nowrap;
        padding: 5px 0;
        margin-bottom: 4px;
    }
    .ticker-scroll {
        display: inline-block;
        animation: tickerScroll 40s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8892b0;
    }
    @keyframes tickerScroll {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .ticker-item {
        display: inline-block;
        margin: 0 18px;
    }
    .ticker-sym { color: #e2e8f0; font-weight: 600; }
    .ticker-up { color: #00ff88; }
    .ticker-down { color: #ff4444; }

    /* === GLOW + ANIMATIONS === */
    .glow-green { text-shadow: 0 0 8px rgba(0,255,136,0.4); }
    .glow-red { text-shadow: 0 0 8px rgba(255,68,68,0.4); }
    .glow-blue { text-shadow: 0 0 8px rgba(74,158,255,0.4); }
    .index-card { transition: border-color 0.3s, box-shadow 0.3s; }
    .index-card:hover { box-shadow: 0 0 12px rgba(74,158,255,0.15); }
    .strategy-card { transition: border-color 0.3s, box-shadow 0.3s; }
    .strategy-card:hover { border-color: #4a9eff; box-shadow: 0 0 10px rgba(74,158,255,0.12); }
    .heat-cell { transition: transform 0.15s, box-shadow 0.2s; }
    .heat-cell:hover { transform: scale(1.08); box-shadow: 0 0 8px rgba(255,255,255,0.1); }
    .panel-card { transition: box-shadow 0.3s; }
    .panel-card:hover { box-shadow: 0 0 15px rgba(74,158,255,0.1); }

    /* Pulse for live badge */
    .market-badge { animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* === DEPTH BARS (Order Book) === */
    .ob-depth-row {
        display: flex;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 1px 0;
        position: relative;
    }
    .ob-depth-bar-bid {
        position: absolute;
        right: 50%;
        top: 0;
        height: 100%;
        background: rgba(0, 255, 136, 0.08);
        border-radius: 2px 0 0 2px;
        z-index: 0;
    }
    .ob-depth-bar-ask {
        position: absolute;
        left: 50%;
        top: 0;
        height: 100%;
        background: rgba(255, 68, 68, 0.08);
        border-radius: 0 2px 2px 0;
        z-index: 0;
    }
    .ob-depth-row > span { position: relative; z-index: 1; }

    /* === TOP BAR BUTTONS === */
    .topbar-btn {
        display: inline-block;
        background: rgba(74,158,255,0.08);
        border: 1px solid rgba(74,158,255,0.25);
        color: #4a9eff;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        cursor: pointer;
        transition: background 0.2s;
        margin: 0 3px;
    }
    .topbar-btn:hover { background: rgba(74,158,255,0.18); }
    .region-badge {
        display: inline-block;
        background: rgba(255,170,0,0.1);
        border: 1px solid rgba(255,170,0,0.3);
        color: #ffaa00;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LAYER — Cached, efficient, on-demand
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def fetch_index_data(symbol: str, period: str = "5d"):
    """Fetch index/stock price with 30-second cache."""
    try:
        dm = DataManager()
        df = dm.get_stock_data(symbol, period=period)
        if df is not None and len(df) >= 2:
            # Normalize column names
            df.columns = [c.capitalize() for c in df.columns]
            current = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = current - prev
            pct = (change / prev) * 100
            return {"price": current, "change": change, "pct": pct, "ok": True}
    except:
        pass
    return {"price": 0, "change": 0, "pct": 0, "ok": False}


def _fast_info_get(fi, *keys):
    """Read a value from yfinance fast_info trying multiple key/attr names."""
    for k in keys:
        try:
            v = fi[k]
            if v is not None:
                return v
        except Exception:
            pass
        v = getattr(fi, k, None)
        if v is not None:
            return v
    return None


@st.cache_data(ttl=3, show_spinner=False)
def fetch_live_quote(symbol: str):
    """
    Near-real-time quote (3-second cache) for live-ticking the selected stock.
    Uses yfinance fast_info which reflects the latest available price.
    """
    try:
        sym = symbol
        if not sym.endswith(".NS") and not sym.startswith("^"):
            sym = sym + ".NS"
        fi = yf.Ticker(sym).fast_info
        price = _fast_info_get(fi, "last_price", "lastPrice")
        prev = _fast_info_get(fi, "previous_close", "previousClose")
        if price and prev and np.isfinite(price) and np.isfinite(prev) and prev != 0:
            change = price - prev
            return {"price": float(price), "change": float(change),
                    "pct": float(change / prev * 100), "ok": True}
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_stock_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d"):
    """Fetch OHLCV data for charting."""
    try:
        dm = DataManager()
        df = dm.get_stock_data(symbol, period=period, interval=interval)
        if df is not None and len(df) > 10:
            df.columns = [c.capitalize() for c in df.columns]
            return df
    except:
        pass
    return None


@st.cache_data(ttl=300)
def fetch_news_feed():
    """Fetch latest market news with sentiment tags."""
    try:
        from src.sentiment.news_analyzer import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        articles = analyzer.fetch_rss_articles()
        results = []
        for art in articles[:20]:
            sent = analyzer.analyze_sentiment(art.get('title', ''))
            compound = sent.get('compound', 0)
            tag = 'BULL' if compound > 0.15 else ('BEAR' if compound < -0.15 else 'NEUTRAL')
            results.append({
                'time': art.get('published', '')[:16],
                'title': art.get('title', 'N/A'),
                'source': art.get('source', ''),
                'tag': tag,
                'score': compound,
            })
        return results
    except:
        return []


@st.cache_data(ttl=120)
def fetch_regime():
    """Get current market regime."""
    try:
        from src.strategy.regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        return detector.detect_regime()
    except:
        return None


@st.cache_data(ttl=120)
def fetch_sector_returns():
    """Fetch 1-day returns for sector heatmap stocks."""
    dm = DataManager()
    returns = {}
    for sector, stocks in SECTOR_MAP.items():
        for sym in stocks[:5]:  # Top 5 per sector for speed
            try:
                df = dm.get_stock_data(sym, period="5d")
                if df is not None and len(df) >= 2:
                    df.columns = [c.capitalize() for c in df.columns]
                    prev = df['Close'].iloc[-2]
                    last = df['Close'].iloc[-1]
                    if not np.isfinite(prev) or not np.isfinite(last) or prev == 0:
                        continue
                    ret = ((last / prev) - 1) * 100
                    if not np.isfinite(ret):
                        continue
                    returns[sym] = {"sector": sector, "return": ret}
            except:
                pass
    return returns


def run_decision_engine(symbol: str):
    """Run the full Decision Engine on ONE stock (on-demand)."""
    try:
        from src.strategy.decision_engine import DecisionEngine
        engine = DecisionEngine()
        return engine.analyze_stock(symbol)
    except Exception as e:
        st.error(f"Decision Engine error: {e}")
        return None


def run_single_stock_backtest(symbol: str) -> dict:
    """
    Full-strategy backtest on ONE stock (2 years).
    
    Replicates the ACTUAL project logic:
    1. Computes momentum score (0-10) weekly — same logic as compute_momentum_score()
    2. Adds quality score as constant (from current fundamentals)
    3. Entry when composite score > 6.0 (would rank in top 15 of universe)
    4. Exit when score drops below 4.0 OR ATR stop-loss hit
    5. Position sizing via Kelly/ATR (risk 2% per trade)
    
    This is how the project ACTUALLY trades — weekly scoring + threshold-based signals.
    Sentiment (20%) and Smart Money (15%) are set to neutral (5/10) since historical
    data isn't available free — same approach real quant firms use for alt-data backtests.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="2y")
        if df.empty or len(df) < 200:
            st.warning(f"Insufficient data for {symbol} (need 200+ days, got {len(df)})")
            return None

        df.columns = [c.lower() for c in df.columns]
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        dates = df.index
        n = len(close)

        # === PRE-COMPUTE ALL INDICATORS ===
        # RSI-14
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean().values
        loss_arr = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(loss_arr != 0, gain / loss_arr, 0)
        rsi = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        macd_hist = macd_line - signal_line

        # Moving Averages
        sma50 = pd.Series(close).rolling(50).mean().values
        sma200 = pd.Series(close).rolling(200).mean().values

        # ATR-14
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(14).mean().values

        # Volume ratio (20-day avg)
        vol_sma20 = pd.Series(volume, dtype=float).rolling(20).mean().values

        # === MOMENTUM SCORE FUNCTION (same as src/indicators/technical.py) ===
        def calc_momentum_score(i: int) -> float:
            """Compute momentum score at day i. Returns 0-10."""
            score = 0
            max_raw = 12

            # RSI Score (0-3)
            if 40 <= rsi[i] <= 65:
                score += 3
            elif 30 <= rsi[i] < 40 or 65 < rsi[i] <= 70:
                score += 1

            # MACD Score (0-2)
            if macd_line[i] > signal_line[i]:
                score += 1
            if i > 0 and macd_hist[i] > macd_hist[i-1]:
                score += 1

            # Moving Average Score (0-3)
            if not np.isnan(sma200[i]) and close[i] > sma200[i]:
                score += 1
            if not np.isnan(sma50[i]) and close[i] > sma50[i]:
                score += 1
            if not np.isnan(sma50[i]) and not np.isnan(sma200[i]) and sma50[i] > sma200[i]:
                score += 1  # Golden cross

            # Return Score (0-3) — 3-month and 6-month returns
            if i >= 63:
                ret_3m = (close[i] / close[i-63] - 1) * 100
                if ret_3m > 15:
                    score += 2
                elif ret_3m > 5:
                    score += 1
            if i >= 126:
                ret_6m = (close[i] / close[i-126] - 1) * 100
                if ret_6m > 20:
                    score += 1

            # Volume confirmation (0-1)
            if not np.isnan(vol_sma20[i]) and vol_sma20[i] > 0:
                # Check if recent up-days had above-average volume
                if i >= 5:
                    up_vol_count = 0
                    for j in range(i-4, i+1):
                        if close[j] > close[j-1] and volume[j] > vol_sma20[j] * 1.5:
                            up_vol_count += 1
                    if up_vol_count >= 2:
                        score += 1

            return round((score / max_raw) * 10, 2)

        # === QUALITY SCORE (constant — fundamentals don't change weekly) ===
        quality_score = 5.0  # Neutral default
        try:
            info = ticker.info
            roe = info.get('returnOnEquity', 0)
            de = info.get('debtToEquity', 0)
            margins = info.get('profitMargins', 0)
            if roe:
                roe_pct = roe * 100 if roe < 1 else roe
                if roe_pct > 20: quality_score += 1.5
                elif roe_pct > 15: quality_score += 1.0
            if de is not None:
                de_val = de / 100 if de > 10 else de
                if de_val < 0.3: quality_score += 1.0
                elif de_val < 0.8: quality_score += 0.5
            if margins:
                margin_pct = margins * 100 if margins < 1 else margins
                if margin_pct > 20: quality_score += 1.0
                elif margin_pct > 10: quality_score += 0.5
            quality_score = min(quality_score, 10.0)
        except Exception:
            quality_score = 5.0

        # === FACTOR WEIGHTS (from config/strategy.yaml) ===
        W_MOMENTUM = 0.40
        W_QUALITY = 0.25
        W_SENTIMENT = 0.20  # Neutral (5.0) — no historical data
        W_SMART_MONEY = 0.15  # Neutral (5.0) — no historical data
        SENTIMENT_NEUTRAL = 5.0
        SMART_MONEY_NEUTRAL = 5.0

        # === WEEKLY REBALANCE SIMULATION ===
        capital = 1000000.0
        equity = capital
        position = 0
        entry_price = 0.0
        entry_score = 0.0
        stop_loss_price = 0.0
        trades = []
        equity_curve = []
        trade_dates = []
        score_history = []

        ENTRY_THRESHOLD = 5.5   # Composite > 5.5 = strong momentum (adjusted for single-stock mode)
        EXIT_THRESHOLD = 3.8    # Composite drops below 3.8 = momentum clearly fading

        for i in range(200, n):  # Start after 200-SMA warms up
            current_equity = equity + (close[i] - entry_price) * position if position > 0 else equity
            equity_curve.append(current_equity)
            trade_dates.append(dates[i])

            # Compute composite score weekly (every Monday) or on first day
            is_rebalance_day = (dates[i].weekday() == 0) or (i == 200)

            if is_rebalance_day or position > 0:
                momentum = calc_momentum_score(i)
                composite = (W_MOMENTUM * momentum +
                            W_QUALITY * quality_score +
                            W_SENTIMENT * SENTIMENT_NEUTRAL +
                            W_SMART_MONEY * SMART_MONEY_NEUTRAL)
                score_history.append({'date': dates[i].strftime('%Y-%m-%d'), 'score': composite})

            if position == 0 and is_rebalance_day:
                # === ENTRY: Composite score above threshold ===
                if composite >= ENTRY_THRESHOLD and not np.isnan(atr[i]) and atr[i] > 0:
                    risk_amount = equity * 0.02  # Risk 2% per trade (Kelly-inspired)
                    stop_dist = 2 * atr[i]
                    shares = int(risk_amount / stop_dist)
                    cost = shares * close[i]
                    if shares > 0 and cost <= equity * 0.90:
                        position = shares
                        entry_price = close[i]
                        entry_score = composite
                        stop_loss_price = entry_price - stop_dist
                        entry_date = dates[i].strftime('%Y-%m-%d')

            elif position > 0:
                # === EXIT CONDITIONS ===
                exit_reason = None

                # 1. ATR stop-loss (hard risk limit)
                if close[i] <= stop_loss_price:
                    exit_reason = "ATR STOP-LOSS (2×ATR)"

                # 2. Score dropped below exit threshold (momentum fading)
                elif is_rebalance_day and composite < EXIT_THRESHOLD:
                    exit_reason = f"SCORE DROP ({composite:.1f} < {EXIT_THRESHOLD})"

                # 3. Trailing stop: if price rose > 10% then fell back 5% from peak
                elif position > 0:
                    unrealized_pct = (close[i] / entry_price - 1) * 100
                    peak_since_entry = max(close[max(200, i-60):i+1])
                    drop_from_peak = (close[i] / peak_since_entry - 1) * 100
                    if unrealized_pct > 10 and drop_from_peak < -5:
                        exit_reason = "TRAILING STOP (5% from peak)"

                if exit_reason:
                    pnl = (close[i] - entry_price) * position
                    pnl_pct = (close[i] / entry_price - 1) * 100
                    equity += pnl
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': dates[i].strftime('%Y-%m-%d'),
                        'Entry ₹': round(entry_price, 2),
                        'Exit ₹': round(close[i], 2),
                        'Shares': position,
                        'P&L ₹': round(pnl, 0),
                        'Return %': round(pnl_pct, 2),
                        'Entry Score': round(entry_score, 2),
                        'Exit Reason': exit_reason,
                    })
                    position = 0
                    entry_price = 0.0

        # Close open position
        if position > 0:
            pnl = (close[-1] - entry_price) * position
            equity += pnl
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': dates[-1].strftime('%Y-%m-%d'),
                'Entry ₹': round(entry_price, 2),
                'Exit ₹': round(close[-1], 2),
                'Shares': position,
                'P&L ₹': round(pnl, 0),
                'Return %': round((close[-1] / entry_price - 1) * 100, 2),
                'Entry Score': round(entry_score, 2),
                'Exit Reason': 'OPEN (end of period)',
            })
            equity_curve[-1] = equity

        # === CALCULATE METRICS ===
        if not equity_curve:
            equity_curve = [capital]

        total_return = (equity / capital - 1) * 100
        winning = [t for t in trades if t['P&L ₹'] > 0]
        losing = [t for t in trades if t['P&L ₹'] <= 0]
        win_rate = (len(winning) / len(trades) * 100) if trades else 0
        total_wins = sum(t['P&L ₹'] for t in winning) if winning else 0
        total_losses = abs(sum(t['P&L ₹'] for t in losing)) if losing else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Max drawdown
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (annualized)
        daily_returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0

        # Avg holding period
        if trades:
            hold_days = []
            for t in trades:
                try:
                    d1 = pd.Timestamp(t['Entry Date'])
                    d2 = pd.Timestamp(t['Exit Date'])
                    hold_days.append((d2 - d1).days)
                except:
                    pass
            avg_hold = np.mean(hold_days) if hold_days else 0
        else:
            avg_hold = 0

        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'equity_curve': equity_curve,
            'trades': trades,
            'dates': [d.strftime('%Y-%m-%d') for d in trade_dates],
            'strategy_name': 'Multi-Factor Composite (Momentum 40% + Quality 25% + Neutral Sent/SM)',
            'capital': capital,
            'avg_hold_days': avg_hold,
            'quality_score': quality_score,
            'score_history': score_history,
        }
    except Exception as e:
        st.error(f"Backtest error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def generate_order_book(current_price: float, live: bool = True, symbol: str = ""):
    """
    Generate SIMULATED order book depth around current price.
    Used only as a fallback when real NSE depth is unavailable.

    live=True  → time-varying (animates while market is open)
    live=False → stable snapshot seeded by price+symbol (frozen when market closed)
    """
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        return [], []

    if live:
        seed = (int(current_price * 100) + int(time.time())) % 100000
    else:
        seed = (int(current_price * 100) + (hash(symbol) % 100000)) % 100000
    np.random.seed(seed)
    tick = round(current_price * 0.0005, 2)
    if tick < 0.05:
        tick = 0.05

    bids, asks = [], []
    for i in range(10):
        bid_price = round(current_price - (i + 1) * tick, 2)
        ask_price = round(current_price + (i + 1) * tick, 2)
        bid_qty = int(np.random.exponential(500) + 50)
        ask_qty = int(np.random.exponential(500) + 50)
        bid_orders = int(np.random.exponential(5) + 1)
        ask_orders = int(np.random.exponential(5) + 1)
        bids.append({"price": bid_price, "qty": bid_qty, "orders": bid_orders})
        asks.append({"price": ask_price, "qty": ask_qty, "orders": ask_orders})

    return bids, asks


def is_market_open() -> bool:
    """
    Real NSE market status when available (free marketStatus API),
    else fall back to IST clock: Mon–Fri, 09:15–15:30.
    """
    real = fetch_market_status()
    if real is not None:
        return real
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)


_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_status():
    """Real NSE Capital Market open/closed state (free marketStatus API). None if unknown."""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE", timeout=8)
        r = s.get("https://www.nseindia.com/api/marketStatus",
                  headers={"Accept": "application/json", "Referer": "https://www.nseindia.com/"},
                  timeout=8)
        if r.status_code == 200:
            for m in r.json().get("marketState", []):
                if m.get("market") == "Capital Market":
                    return str(m.get("marketStatus", "")).strip().lower() == "open"
    except Exception:
        pass
    return None


@st.cache_data(ttl=3, show_spinner=False)
def fetch_quote_groww(symbol: str):
    """
    Real near-real-time quote from Groww's free public API.
    Returns LTP, previous close, day change, REAL total bid/ask quantities and volume.
    """
    try:
        url = (f"https://groww.in/v1/api/stocks_data/v1/tr_live_prices/"
               f"exchange/NSE/segment/CASH/{quote(symbol)}/latest")
        r = requests.get(url, headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"}, timeout=8)
        if r.status_code == 200:
            d = r.json()
            ltp = d.get("ltp")
            if ltp:
                return {
                    "ltp": float(ltp),
                    "prev_close": d.get("close"),
                    "day_change": d.get("dayChange"),
                    "pct": d.get("dayChangePerc"),
                    "total_buy": d.get("totalBuyQty"),
                    "total_sell": d.get("totalSellQty"),
                    "volume": d.get("volume"),
                    "high": d.get("high"),
                    "low": d.get("low"),
                }
    except Exception:
        pass
    return None


def _try_nse_ladder(symbol: str):
    """
    Attempt the REAL NSE 5-level order book ladder (quote-equity API).
    Works on networks where NSE's Akamai protection allows it (uses Chrome TLS
    impersonation). If blocked once, it is skipped for the rest of the session.
    """
    if st.session_state.get("_nse_depth_blocked"):
        return None
    try:
        from curl_cffi import requests as creq
        s = creq.Session(impersonate="chrome120")
        s.get("https://www.nseindia.com", timeout=8)
        s.get(f"https://www.nseindia.com/get-quotes/equity?symbol={quote(symbol)}", timeout=8)
        r = s.get(f"https://www.nseindia.com/api/quote-equity?symbol={quote(symbol)}",
                  headers={"Accept": "application/json, text/plain, */*",
                           "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={quote(symbol)}"},
                  timeout=8)
        if r.status_code == 200:
            d = r.json()
            dep = d.get("marketDeptOrderBook", {}) or {}
            bids = [{"price": float(b["price"]), "qty": int(b.get("quantity", 0) or 0), "orders": None}
                    for b in dep.get("bid", []) if b.get("price")]
            asks = [{"price": float(a["price"]), "qty": int(a.get("quantity", 0) or 0), "orders": None}
                    for a in dep.get("ask", []) if a.get("price")]
            if bids and asks:
                return {
                    "bids": bids, "asks": asks, "real_ladder": True,
                    "ltp": d.get("priceInfo", {}).get("lastPrice"),
                    "total_buy": dep.get("totalBuyQuantity"),
                    "total_sell": dep.get("totalSellQuantity"),
                }
        if r.status_code in (401, 403):
            st.session_state["_nse_depth_blocked"] = True
    except Exception:
        st.session_state["_nse_depth_blocked"] = True
    return None


def build_ladder_from_aggregates(price: float, total_buy, total_sell,
                                 symbol: str = "", live: bool = True):
    """
    Build a 5-level depth ladder whose TOTAL bid/ask quantities equal the REAL
    aggregate buy/sell quantities (from Groww), distributed across price levels
    with a realistic front-loaded profile. Per-level prices step from the real LTP.
    """
    if price is None or not np.isfinite(price) or price <= 0:
        return [], []
    tick = round(price * 0.0005, 2)
    if tick < 0.05:
        tick = 0.05
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

    seed = (int(price * 100) + (hash(symbol) % 100000) + (int(time.time()) // 2 if live else 0)) % 100000
    rng = np.random.default_rng(seed)
    jitter = lambda: 1.0 + rng.uniform(-0.12, 0.12)

    tb = int(total_buy) if total_buy and total_buy > 0 else 0
    ts = int(total_sell) if total_sell and total_sell > 0 else 0

    bids, asks = [], []
    for i in range(5):
        bq = int(tb * weights[i] * jitter()) if tb else 0
        aq = int(ts * weights[i] * jitter()) if ts else 0
        bids.append({"price": round(price - (i + 1) * tick, 2), "qty": bq, "orders": None})
        asks.append({"price": round(price + (i + 1) * tick, 2), "qty": aq, "orders": None})
    return bids, asks


# ═══════════════════════════════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def render_top_bar():
    """Top navigation bar with branding, market status, controls, and clock."""
    now = datetime.now()
    market_open = now.weekday() < 5 and 9 <= now.hour < 16
    status = "LIVE" if market_open else "CLOSED"
    badge_class = "market-badge" if market_open else "market-badge-closed"

    st.markdown(f"""
    <div class="top-bar">
        <div>
            <span class="brand-title">◉ WAR ROOM TERMINAL</span>
            &nbsp;&nbsp;
            <span class="{badge_class}">NSE {status}</span>
            &nbsp;&nbsp;
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
                NIFTY 50 DERIVATIVES
            </span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span class="region-badge">🇮🇳 INDIA</span>
            <span class="clock-display">
                {now.strftime('%a %d %b %Y')} &nbsp;│&nbsp; {now.strftime('%H:%M:%S')} IST
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_index_card(name: str, data: dict):
    """Render a single index card."""
    if not data["ok"]:
        st.markdown(f"""
        <div class="index-card">
            <div class="index-name">{name}</div>
            <div class="index-price" style="color:#4a5568;">--</div>
        </div>""", unsafe_allow_html=True)
        return

    change_class = "index-change-up" if data["change"] >= 0 else "index-change-down"
    arrow = "▲" if data["change"] >= 0 else "▼"
    st.markdown(f"""
    <div class="index-card">
        <div class="index-name">{name}</div>
        <div class="index-price">{data['price']:,.2f}</div>
        <div class="{change_class}">
            {arrow} {abs(data['change']):,.2f} ({data['pct']:+.2f}%)
        </div>
    </div>""", unsafe_allow_html=True)


def render_news_feed(news_items: list):
    """Render scrollable news feed with sentiment tags."""
    st.markdown('<div class="panel-header">📰 LIVE NEWS FEED</div>', unsafe_allow_html=True)

    if not news_items:
        st.markdown('<p style="color:#4a5568;font-size:0.75rem;">No news available</p>',
                    unsafe_allow_html=True)
        return

    html = '<div class="scrollable">'
    for item in news_items[:12]:
        tag_class = f"tag-{item['tag'].lower()}"
        html += f"""
        <div class="news-item">
            <span class="news-time">{item['time']}</span>
            &nbsp;<span class="{tag_class}">{item['tag']}</span>
            <div class="news-text">{item['title'][:120]}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_candlestick_chart(df: pd.DataFrame, symbol: str, indicators: list):
    """Render interactive candlestick chart with indicators."""
    if df is None or len(df) < 5:
        st.warning("No chart data available")
        return

    # Compute indicators
    df = df.copy()
    # Ensure lowercase for get_all_indicators, then capitalize OHLCV for charting
    df.columns = [c.lower() for c in df.columns]
    df = get_all_indicators(df)
    ohlcv_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df.rename(columns={k: v for k, v in ohlcv_map.items() if k in df.columns}, inplace=True)

    # Build figure with volume subplot
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=None
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88', decreasing_fillcolor='#ff4444',
        name='Price', whiskerwidth=0.5
    ), row=1, col=1)

    # Moving Averages
    if 'EMA 20' in indicators and 'ema_21' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], name='EMA 21',
                                 line=dict(color='#4a9eff', width=1)), row=1, col=1)
    if 'SMA 50' in indicators and 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50',
                                 line=dict(color='#ff9f43', width=1)), row=1, col=1)
    if 'SMA 200' in indicators and 'sma_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], name='SMA 200',
                                 line=dict(color='#a855f7', width=1)), row=1, col=1)

    # Bollinger Bands
    if 'Bollinger' in indicators and 'bb_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                                 line=dict(color='rgba(74,158,255,0.3)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                                 line=dict(color='rgba(74,158,255,0.3)', width=1, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(74,158,255,0.03)'), row=1, col=1)

    # Volume bars
    vol_colors = ['#00ff88' if c >= o else '#ff4444'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                         marker_color=vol_colors, opacity=0.5), row=2, col=1)

    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                                 line=dict(color='#ffaa00', width=1.2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,68,68,0.4)",
                      row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,136,0.4)",
                      row=3, col=1)

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0d1220',
        height=520,
        margin=dict(l=50, r=20, t=10, b=20),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=9, color='#6b7394')
        ),
        xaxis_rangeslider_visible=False,
        font=dict(family='JetBrains Mono', size=10, color='#8892b0'),
    )

    # Axes styling
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor='#1a2332', zeroline=False,
            showgrid=True, gridwidth=0.5, row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='#1a2332', zeroline=False,
            showgrid=True, gridwidth=0.5, row=i, col=1
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath'],
        'displaylogo': False,
        'scrollZoom': True,
    })


def render_order_book(bids: list, asks: list, current_price: float,
                      source: str = None, source_color: str = "#00ff88",
                      total_buy=None, total_sell=None):
    """Render order book depth ladder with a data-source badge."""
    header = '📊 ORDER BOOK DEPTH'
    if source:
        header += (f'<span style="float:right;font-size:0.6rem;color:{source_color};'
                   f'font-weight:700;letter-spacing:0.5px;">● {source}</span>')
    st.markdown(f'<div class="panel-header">{header}</div>', unsafe_allow_html=True)

    if not bids or not asks:
        st.markdown(
            '<div style="text-align:center;color:#6b7394;font-family:\'JetBrains Mono\',monospace;'
            'font-size:0.72rem;padding:18px 0;">Depth unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    max_qty = max(
        max(b['qty'] for b in bids) if bids else 1,
        max(a['qty'] for a in asks) if asks else 1
    ) or 1

    # Spread info
    if bids and asks:
        spread = asks[0]['price'] - bids[0]['price']
        spread_pct = (spread / current_price) * 100 if current_price else 0
        st.markdown(f"""
        <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.7rem;margin:4px 0;">
            <span style="color:#8892b0;">LTP</span>
            <span style="color:#e2e8f0;font-weight:700;font-size:0.9rem;"> ₹{current_price:,.2f}</span>
            &nbsp;│&nbsp;
            <span style="color:#8892b0;">Spread</span>
            <span style="color:#ffaa00;"> ₹{spread:.2f} ({spread_pct:.3f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="ob-row" style="color:#4a5568;font-size:0.6rem;border-bottom:1px solid #1a2332;padding-bottom:3px;">
        <span class="ob-qty">ORDERS</span>
        <span class="ob-qty">QTY</span>
        <span class="ob-price" style="color:#00ff88;">BID</span>
        <span style="width:10px;">&nbsp;</span>
        <span class="ob-price" style="color:#ff4444;">ASK</span>
        <span class="ob-qty">QTY</span>
        <span class="ob-qty">ORDERS</span>
    </div>
    """, unsafe_allow_html=True)

    for i in range(min(len(bids), len(asks), 10)):
        b, a = bids[i], asks[i]
        bid_pct = int((b['qty'] / max_qty) * 100)
        ask_pct = int((a['qty'] / max_qty) * 100)
        b_ord = b['orders'] if b.get('orders') is not None else '—'
        a_ord = a['orders'] if a.get('orders') is not None else '—'
        st.markdown(f"""
        <div class="ob-depth-row">
            <div class="ob-depth-bar-bid" style="width:{bid_pct//2}%;"></div>
            <div class="ob-depth-bar-ask" style="width:{ask_pct//2}%;"></div>
            <span class="ob-qty">{b_ord}</span>
            <span class="ob-qty ob-bid">{b['qty']:,}</span>
            <span class="ob-price ob-bid">{b['price']:,.2f}</span>
            <span style="width:10px;color:#1a2332;">│</span>
            <span class="ob-price ob-ask">{a['price']:,.2f}</span>
            <span class="ob-qty ob-ask">{a['qty']:,}</span>
            <span class="ob-qty">{a_ord}</span>
        </div>
        """, unsafe_allow_html=True)

    # Total buy/sell quantity (like real brokerage apps)
    if total_buy is not None or total_sell is not None:
        tb = f"{int(total_buy):,}" if total_buy else "—"
        ts = f"{int(total_sell):,}" if total_sell else "—"
        st.markdown(f"""
        <div class="ob-row" style="margin-top:4px;border-top:1px solid #1a2332;padding-top:4px;
             font-family:'JetBrains Mono',monospace;font-size:0.62rem;">
            <span style="color:#8892b0;">TOTAL BID QTY</span>
            <span class="ob-bid" style="font-weight:700;">{tb}</span>
            <span style="flex:1;"></span>
            <span class="ob-ask" style="font-weight:700;">{ts}</span>
            <span style="color:#8892b0;">TOTAL ASK QTY</span>
        </div>
        """, unsafe_allow_html=True)


def render_trading_panel(symbol: str, current_price: float):
    """Render the broker execution panel."""
    st.markdown('<div class="panel-header">⚡ TRADING PANEL</div>', unsafe_allow_html=True)

    # Broker connection
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;margin-bottom:6px;">
        BROKER: <span style="color:#4a9eff;">ICICI Direct</span>
        &nbsp;<span style="color:#ffaa00;">●</span> Connect via Breeze API
    </div>
    """, unsafe_allow_html=True)

    # Buy/Sell toggle
    action = st.radio("Action", ["BUY", "SELL"], horizontal=True, key="trade_action",
                      label_visibility="collapsed")

    c1, c2 = st.columns(2)
    with c1:
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"],
                                  key="order_type")
    with c2:
        product = st.selectbox("Product", ["MIS", "CNC", "NRML"], key="product")

    c1, c2 = st.columns(2)
    with c1:
        qty = st.number_input("Quantity", min_value=1, value=1, step=1, key="qty")
    with c2:
        price = st.number_input("Price ₹", value=round(current_price, 2),
                                step=0.05, key="price", format="%.2f")

    if order_type in ["SL", "SL-M"]:
        trigger = st.number_input("Trigger ₹", value=round(current_price * 0.99, 2),
                                  step=0.05, key="trigger", format="%.2f")

    total_val = qty * price
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#8892b0;
                border-top:1px solid #1a2332;padding-top:6px;margin-top:6px;">
        Total: <span style="color:#e2e8f0;font-weight:700;">₹{total_val:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    btn_color = "🟢" if action == "BUY" else "🔴"
    if st.button(f"{btn_color} PLACE {action} ORDER", use_container_width=True, key="place_order"):
        st.warning("⚠️ Connect Breeze API to execute live orders. This is a paper order.")
        st.info(f"Paper {action}: {symbol} × {qty} @ ₹{price:.2f} ({order_type}/{product})")


def render_strategy_signals(symbol: str, df: pd.DataFrame):
    """Render strategy signals panel for selected stock."""
    st.markdown('<div class="panel-header">🎯 STRATEGY SIGNALS</div>', unsafe_allow_html=True)

    if df is None or len(df) < 20:
        st.markdown('<p style="color:#4a5568;font-size:0.75rem;">Insufficient data</p>',
                    unsafe_allow_html=True)
        return

    df = df.copy()
    # Ensure lowercase for get_all_indicators
    df.columns = [c.lower() for c in df.columns]
    df = get_all_indicators(df)
    ohlcv_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df.rename(columns={k: v for k, v in ohlcv_map.items() if k in df.columns}, inplace=True)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    current_price = last['Close']

    strategies = []

    # 1. EMA Crossover
    if 'ema_21' in df.columns and 'sma_50' in df.columns:
        ema_above = last['ema_21'] > last['sma_50']
        prev_ema_above = prev['ema_21'] > prev['sma_50']
        if ema_above and not prev_ema_above:
            signal, signal_cls = "BUY", "signal-buy"
        elif not ema_above and prev_ema_above:
            signal, signal_cls = "SELL", "signal-sell"
        elif ema_above:
            signal, signal_cls = "HOLD ↑", "signal-hold"
        else:
            signal, signal_cls = "HOLD ↓", "signal-hold"

        rsi = last.get('rsi', 50)
        entry = round(current_price, 2)
        sl = round(current_price * 0.97, 2)
        target = round(current_price * 1.05, 2)
        rr = round(abs(target - entry) / max(abs(entry - sl), 0.01), 1)

        strategies.append({
            'name': 'EMA CROSSOVER',
            'signal': signal, 'signal_cls': signal_cls,
            'entry': entry, 'sl': sl, 'target': target,
            'rsi': rsi, 'rr': rr,
            'tags': [f"EMA21{'>' if ema_above else '<'}SMA50", f"RSI {rsi:.0f}"],
            'win_prob': 58 if signal in ['BUY', 'HOLD ↑'] else 42,
        })

    # 2. Bollinger Squeeze
    if 'bb_width' in df.columns and 'bb_upper' in df.columns:
        bb_width = last['bb_width']
        avg_width = df['bb_width'].rolling(20).mean().iloc[-1]
        squeeze = bb_width < avg_width * 0.75

        if current_price > last['bb_upper']:
            signal, signal_cls = "BUY", "signal-buy"
        elif current_price < last['bb_lower']:
            signal, signal_cls = "SELL", "signal-sell"
        else:
            signal, signal_cls = "NEUTRAL", "signal-hold"

        entry = round(current_price, 2)
        sl = round(last['bb_lower'], 2)
        target = round(last['bb_upper'], 2)
        rr = round(abs(target - entry) / max(abs(entry - sl), 0.01), 1)

        tags = []
        if squeeze:
            tags.append("SQUEEZE ⚡")
        tags.append(f"BB Width {bb_width:.4f}")
        if 'vol_price_confirm' in df.columns and last.get('vol_price_confirm', False):
            tags.append("Vol Confirm ✓")

        strategies.append({
            'name': 'BOLLINGER SQUEEZE',
            'signal': signal, 'signal_cls': signal_cls,
            'entry': entry, 'sl': sl, 'target': target,
            'rsi': last.get('rsi', 50), 'rr': rr,
            'tags': tags,
            'win_prob': 55 if squeeze else 48,
        })

    # 3. RSI Reversal
    if 'rsi' in df.columns:
        rsi = last['rsi']
        if rsi < 30:
            signal, signal_cls = "BUY", "signal-buy"
            tags = ["OVERSOLD", f"RSI {rsi:.0f}"]
            win_prob = 62
        elif rsi > 70:
            signal, signal_cls = "SELL", "signal-sell"
            tags = ["OVERBOUGHT", f"RSI {rsi:.0f}"]
            win_prob = 60
        else:
            signal, signal_cls = "NEUTRAL", "signal-hold"
            tags = [f"RSI {rsi:.0f}"]
            win_prob = 50

        entry = round(current_price, 2)
        sl = round(current_price * (0.97 if rsi < 50 else 1.03), 2)
        target = round(current_price * (1.06 if rsi < 50 else 0.94), 2)
        rr = round(abs(target - entry) / max(abs(entry - sl), 0.01), 1)

        strategies.append({
            'name': 'RSI REVERSAL',
            'signal': signal, 'signal_cls': signal_cls,
            'entry': entry, 'sl': sl, 'target': target,
            'rsi': rsi, 'rr': rr,
            'tags': tags,
            'win_prob': win_prob,
        })

    # 4. MACD Crossover
    if 'macd' in df.columns and 'signal' in df.columns:
        macd_val = last['macd']
        signal_val = last['signal']
        prev_macd = prev['macd']
        prev_signal = prev['signal']

        if macd_val > signal_val and prev_macd <= prev_signal:
            signal, signal_cls = "BUY", "signal-buy"
            win_prob = 56
        elif macd_val < signal_val and prev_macd >= prev_signal:
            signal, signal_cls = "SELL", "signal-sell"
            win_prob = 54
        elif macd_val > signal_val:
            signal, signal_cls = "HOLD ↑", "signal-hold"
            win_prob = 52
        else:
            signal, signal_cls = "HOLD ↓", "signal-hold"
            win_prob = 48

        entry = round(current_price, 2)
        sl = round(current_price * 0.97, 2)
        target = round(current_price * 1.05, 2)
        rr = round(abs(target - entry) / max(abs(entry - sl), 0.01), 1)

        strategies.append({
            'name': 'MACD CROSSOVER',
            'signal': signal, 'signal_cls': signal_cls,
            'entry': entry, 'sl': sl, 'target': target,
            'rsi': last.get('rsi', 50), 'rr': rr,
            'tags': [f"MACD {macd_val:.2f}", f"Hist {last.get('histogram', 0):.2f}"],
            'win_prob': win_prob,
        })

    # Render strategy cards
    if not strategies:
        st.markdown('<p style="color:#4a5568;font-size:0.75rem;">No strategies computed</p>',
                    unsafe_allow_html=True)
        return

    cols = st.columns(min(len(strategies), 4))
    for i, strat in enumerate(strategies):
        with cols[i % len(cols)]:
            tags_html = ' '.join(
                f'<span class="tag-{"bull" if "BUY" in strat["signal"] else ("bear" if "SELL" in strat["signal"] else "neutral")}">{t}</span>'
                for t in strat['tags']
            )
            expected_ret = ((strat['target'] - strat['entry']) / strat['entry']) * 100

            st.markdown(f"""
            <div class="strategy-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a9eff;
                                 font-weight:600;letter-spacing:1px;">{strat['name']}</span>
                    <span class="{strat['signal_cls']}">{strat['signal']}</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;">
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">Entry</span>
                        <span class="strategy-value">₹{strat['entry']:,.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">Target</span>
                        <span class="strategy-value" style="color:#00ff88;">₹{strat['target']:,.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">Stop Loss</span>
                        <span class="strategy-value" style="color:#ff4444;">₹{strat['sl']:,.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">RSI</span>
                        <span class="strategy-value">{strat['rsi']:.1f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">R:R</span>
                        <span class="strategy-value">{strat['rr']:.1f}x</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">Win Prob</span>
                        <span class="strategy-value">{strat['win_prob']}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span class="strategy-metric">Exp Return</span>
                        <span class="strategy-value" style="color:{'#00ff88' if expected_ret>0 else '#ff4444'};">
                            {expected_ret:+.1f}%
                        </span>
                    </div>
                </div>
                <div style="margin-top:6px;">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)


def render_sector_heatmap(returns_data: dict):
    """Render clickable sector heatmap grid — click a stock to open it on the Trade tab."""
    st.markdown('<div class="panel-header">🔥 SECTOR HEATMAP — click a stock to open</div>',
                unsafe_allow_html=True)

    # Keep only valid, finite returns
    valid = {s: d for s, d in returns_data.items() if np.isfinite(d.get('return', np.nan))}
    if not valid:
        st.markdown('<p style="color:#4a5568;font-size:0.75rem;">Sector data unavailable (data source returned no prices).</p>',
                    unsafe_allow_html=True)
        return

    items = sorted(valid.items(), key=lambda x: x[1]['return'], reverse=True)

    def colors_for(ret):
        if ret > 2:
            return 'rgba(0,200,100,0.35)', '#00ff88'
        if ret > 0.5:
            return 'rgba(0,200,100,0.18)', '#00cc66'
        if ret > -0.5:
            return 'rgba(128,128,128,0.15)', '#cbd5e0'
        if ret > -2:
            return 'rgba(255,68,68,0.18)', '#ff6666'
        return 'rgba(255,68,68,0.35)', '#ff4444'

    # Inject CSS to color each button by its return (targets Streamlit's st-key-<key> class)
    css = "<style>"
    for sym, data in items:
        safe = re.sub(r'[^A-Za-z0-9]', '', sym)
        bg, col = colors_for(data['return'])
        css += (f".st-key-heat_{safe} button{{background:{bg}!important;color:{col}!important;"
                f"border:1px solid #1a2332!important;border-radius:4px!important;"
                f"font-family:'JetBrains Mono',monospace!important;font-size:0.68rem!important;"
                f"font-weight:600!important;padding:3px 4px!important;min-height:0!important;}}")
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)

    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        row = items[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for c, (sym, data) in zip(cols, row):
            safe = re.sub(r'[^A-Za-z0-9]', '', sym)
            ret = data['return']
            with c:
                if st.button(f"{sym}  {ret:+.1f}%", key=f"heat_{safe}", use_container_width=True):
                    if sym in NSE_STOCKS:
                        st.session_state['pending_symbol'] = sym
                        st.rerun()


def render_decision_engine_results(decision):
    """Render the AI Decision Engine verdict."""
    if decision is None:
        return

    verdict_colors = {'TAKE': '#00ff88', 'WATCH': '#ffaa00', 'SKIP': '#ff4444'}
    color = verdict_colors.get(decision.verdict, '#8892b0')

    st.markdown(f"""
    <div class="panel-card" style="border-color:{color};border-width:2px;">
        <div class="panel-header">🧠 AI DECISION ENGINE</div>
        <div style="text-align:center;margin:8px 0;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:{color};
                        text-shadow:0 0 20px {color}40;">
                {decision.verdict}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#8892b0;">
                Score: {decision.score:.1f}/100 &nbsp;│&nbsp; Conviction: {decision.conviction}/5
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#8892b0;">
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>Signal</span><span style="color:#e2e8f0;">{decision.signal_score:.1f}/25</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>ML</span><span style="color:#e2e8f0;">{decision.ml_score:.1f}/20</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>Kelly</span><span style="color:#e2e8f0;">{decision.kelly_score:.1f}/20</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>Regime</span><span style="color:#e2e8f0;">{decision.regime_score:.1f}/15</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>Events</span><span style="color:#e2e8f0;">{decision.event_score:.1f}/10</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>Geo</span><span style="color:#e2e8f0;">{decision.geo_score:.1f}/10</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Trade plan if TAKE
    if decision.verdict == 'TAKE' and decision.entry_price > 0:
        st.markdown(f"""
        <div style="border-top:1px solid #1a2332;margin-top:8px;padding-top:8px;
                     font-family:'JetBrains Mono',monospace;font-size:0.7rem;">
            <div style="color:#4a9eff;font-weight:600;margin-bottom:4px;">TRADE PLAN</div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">Entry</span>
                <span style="color:#e2e8f0;">₹{decision.entry_price:,.2f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">Stop Loss</span>
                <span style="color:#ff4444;">₹{decision.stop_loss:,.2f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">Target</span>
                <span style="color:#00ff88;">₹{decision.target:,.2f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">Position</span>
                <span style="color:#e2e8f0;">{decision.position_size} shares (₹{decision.position_value:,.0f})</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">R:R</span>
                <span style="color:#e2e8f0;">{decision.risk_reward:.1f}x</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;">Max Loss</span>
                <span style="color:#ff4444;">₹{decision.max_loss:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Reasons
    if decision.reasons_for:
        reasons_html = '<br>'.join(f'<span style="color:#00ff88;">+ {r}</span>' for r in decision.reasons_for[:4])
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;margin-top:6px;">{reasons_html}</div>',
                    unsafe_allow_html=True)
    if decision.reasons_against:
        reasons_html = '<br>'.join(f'<span style="color:#ff4444;">- {r}</span>' for r in decision.reasons_against[:3])
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;margin-top:3px;">{reasons_html}</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# LIVE FRAGMENTS — auto-refreshing pieces (price, order book, ticker)
# ═══════════════════════════════════════════════════════════════════

# Liquid pool for the rotating ticker strip (bounded to avoid API rate limits)
TICKER_POOL = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "BAJFINANCE", "LT", "HCLTECH", "AXISBANK", "SUNPHARMA", "TITAN",
    "MARUTI", "WIPRO", "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA",
    "KOTAKBANK", "ULTRACEMCO", "ONGC", "TECHM", "ADANIENT", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "CIPLA", "DRREDDY", "BAJAJFINSV", "EICHERMOT",
    "NESTLEIND", "HEROMOTOCO", "BRITANNIA", "DABUR", "HAVELLS", "IOC",
    "BPCL", "VEDL",
]


@st.fragment(run_every=3)
def live_price_header(symbol: str):
    """Continuously-updating price line for the selected stock."""
    q = fetch_live_quote(symbol)
    if not q or not q.get("ok"):
        q = fetch_index_data(symbol + ".NS")
    if not q or not q.get("ok"):
        return
    change_color = "#00ff88" if q["change"] >= 0 else "#ff4444"
    arrow = "▲" if q["change"] >= 0 else "▼"
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#e2e8f0;">
            {symbol}
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600;color:#e2e8f0;">
            ₹{q['price']:,.2f}
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:{change_color};font-weight:600;">
            {arrow} {abs(q['change']):,.2f} ({q['pct']:+.2f}%)
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#00ff88;">
            ● LIVE {datetime.now().strftime('%H:%M:%S')}
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#4a5568;">
            {NSE_STOCKS.get(symbol, '')}
        </span>
    </div>
    """, unsafe_allow_html=True)


@st.fragment(run_every=4)
def live_order_book(symbol: str):
    """
    Order book depth for the selected stock, from FREE live sources.

    Priority of data:
      1. REAL NSE 5-level ladder (quote-equity API) when the network/Akamai allows it.
      2. Otherwise a 5-level ladder whose TOTAL bid/ask quantities are the REAL
         aggregate buy/sell qty from Groww's free API, anchored to the REAL LTP.
      3. Market open/closed is the REAL NSE status; when closed the book is frozen.
    """
    market_open = is_market_open()

    # 1) Try the genuine NSE ladder (best case)
    nse = _try_nse_ladder(symbol) if market_open else None

    # 2) Real aggregates + LTP from Groww (works even when NSE ladder is blocked)
    gq = fetch_quote_groww(symbol)

    # Reference price: prefer NSE LTP → Groww LTP → yfinance
    price = None
    if nse and nse.get("ltp"):
        price = float(nse["ltp"])
    elif gq and gq.get("ltp"):
        price = float(gq["ltp"])
    else:
        q = fetch_live_quote(symbol)
        if q and q.get("ok"):
            price = q["price"]
        else:
            fb = fetch_index_data(symbol + ".NS")
            price = fb["price"] if fb["ok"] else 1000

    if nse and nse.get("real_ladder"):
        # Genuine NSE depth
        src, color = ("NSE LIVE", "#00ff88") if market_open else ("NSE · SNAPSHOT", "#ffaa00")
        render_order_book(nse["bids"], nse["asks"], price, source=src, source_color=color,
                          total_buy=nse.get("total_buy"), total_sell=nse.get("total_sell"))
        return

    total_buy = gq.get("total_buy") if gq else None
    total_sell = gq.get("total_sell") if gq else None

    if market_open and gq and (total_buy or total_sell):
        # Ladder built from REAL total bid/ask quantities + REAL LTP
        bids, asks = build_ladder_from_aggregates(price, total_buy, total_sell, symbol, live=True)
        render_order_book(bids, asks, price, source="LIVE · DEPTH EST", source_color="#00d4ff",
                          total_buy=total_buy, total_sell=total_sell)
    elif not market_open:
        # Frozen, balanced snapshot when market is closed (no synthetic churn)
        bids, asks = generate_order_book(price, live=False, symbol=symbol)
        render_order_book(bids, asks, price, source="MARKET CLOSED · SNAPSHOT", source_color="#6b7394")
    else:
        # Last resort: modeled ladder around the real price
        bids, asks = generate_order_book(price, live=True)
        render_order_book(bids, asks, price, source="LIVE · MODELED", source_color="#ff6b35")


@st.fragment(run_every=5)
def live_ticker_strip():
    """Scrolling ticker showing a rotating, random set of stocks."""
    picks = random.sample(TICKER_POOL, min(14, len(TICKER_POOL)))
    ticker_html = '<div class="ticker-strip"><div class="ticker-scroll">'
    for sym in picks:
        tdata = fetch_index_data(sym + '.NS')
        if tdata['ok']:
            cls = 'ticker-up' if tdata['change'] >= 0 else 'ticker-down'
            arrow = '▲' if tdata['change'] >= 0 else '▼'
            ticker_html += f'''<span class="ticker-item">
                <span class="ticker-sym">{sym}</span>
                <span class="{cls}"> ₹{tdata["price"]:,.1f} {arrow}{abs(tdata["pct"]):.1f}%</span>
            </span>'''
    ticker_html += '</div></div>'
    st.markdown(ticker_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════

def main():
    # ─── MODULE NAVIGATION (Sidebar) ───
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;
                    color:#4a9eff;padding:8px 0;border-bottom:1px solid #1a2332;margin-bottom:8px;">
            🔴 WAR ROOM
        </div>
        """, unsafe_allow_html=True)
        nav_module = st.radio(
            "Module",
            ["📊 TRADE", "🎯 PICK", "🔍 ANALYZE", "💰 PAPER TRADE", "⭐ WATCHLIST", "🔎 SCREENER", "📈 PERFORMANCE"],
            index=0,
            key="nav_module",
            label_visibility="collapsed",
        )

    # Route to selected module
    if nav_module == "🎯 PICK":
        from src.dashboard.stock_picker import render_stock_picker
        render_stock_picker()
        return
    elif nav_module == "🔍 ANALYZE":
        from src.dashboard.analyzer import render_stock_analyzer
        # Show analyzer with symbol search
        st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
        stock_options = [f"{sym} — {name}" for sym, name in sorted(NSE_STOCKS.items())]
        default_idx = stock_options.index("RELIANCE — Reliance Industries") if "RELIANCE — Reliance Industries" in stock_options else 0
        selected = st.selectbox("🔍 Select Stock to Analyze", stock_options, index=default_idx, key="analyze_stock")
        symbol = selected.split(" — ")[0] if selected else "RELIANCE"
        render_stock_analyzer(symbol)
        return
    elif nav_module == "💰 PAPER TRADE":
        from src.dashboard.paper_trading import render_paper_trading
        st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
        render_paper_trading()
        return
    elif nav_module == "⭐ WATCHLIST":
        from src.dashboard.watchlist import render_watchlist
        st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
        render_watchlist()
        return
    elif nav_module == "🔎 SCREENER":
        from src.dashboard.screener import render_screener
        st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
        render_screener()
        return
    elif nav_module == "📈 PERFORMANCE":
        from src.dashboard.performance import render_performance
        st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
        render_performance()
        return

    # === DEFAULT: TRADE (original terminal) ===
    # ─── TOP BAR ───
    render_top_bar()

    # ─── LIVE TICKER STRIP (rotating random stocks) ───
    live_ticker_strip()

    # ─── STOCK SEARCH BAR ───
    # Apply a pending symbol selection (e.g. from a heatmap click) BEFORE the
    # selectbox widget is instantiated — session_state for a widget key cannot
    # be changed after the widget is created.
    if 'pending_symbol' in st.session_state:
        psym = st.session_state.pop('pending_symbol')
        if psym in NSE_STOCKS:
            st.session_state['stock_search'] = f"{psym} — {NSE_STOCKS[psym]}"

    search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns([3, 1, 1, 1, 1])

    with search_col1:
        stock_options = [f"{sym} — {name}" for sym, name in sorted(NSE_STOCKS.items())]
        default_idx = stock_options.index("RELIANCE — Reliance Industries") if "RELIANCE — Reliance Industries" in stock_options else 0
        selected = st.selectbox(
            "🔍 Search Stock",
            stock_options,
            index=default_idx,
            key="stock_search",
            label_visibility="collapsed",
            placeholder="Search symbol or company name..."
        )
        selected_symbol = selected.split(" — ")[0] if selected else "RELIANCE"

    with search_col2:
        timeframe = st.selectbox("Timeframe",
                                 ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                                 index=4, key="timeframe", label_visibility="collapsed")

    with search_col3:
        indicators = st.multiselect("Indicators", ["EMA 20", "SMA 50", "SMA 200", "Bollinger"],
                                    default=["EMA 20", "Bollinger"], key="indicators",
                                    label_visibility="collapsed")

    with search_col4:
        run_ai = st.button("🧠 RUN AI ENGINE", key="run_ai", use_container_width=True)

    with search_col5:
        run_backtest = st.button("⚡ BACKTEST", key="run_backtest", use_container_width=True)

    st.markdown('<hr style="margin:2px 0;">', unsafe_allow_html=True)

    # ─── MAIN 3-COLUMN LAYOUT ───
    left_col, center_col, right_col = st.columns([1.2, 3, 1.3])

    # ═══ LEFT COLUMN: Market Overview + News ═══
    with left_col:
        st.markdown('<div class="panel-header">📈 MARKET OVERVIEW</div>', unsafe_allow_html=True)

        nifty = fetch_index_data("^NSEI")
        sensex = fetch_index_data("^BSESN")
        banknifty = fetch_index_data("^NSEBANK")

        render_index_card("NIFTY 50", nifty)
        render_index_card("SENSEX", sensex)
        render_index_card("BANK NIFTY", banknifty)

        # VIX
        vix = fetch_index_data("^INDIAVIX")
        if vix["ok"]:
            vix_color = "#ff4444" if vix['price'] > 20 else ("#ffaa00" if vix['price'] > 15 else "#00ff88")
            st.markdown(f"""
            <div class="index-card">
                <div class="index-name">INDIA VIX</div>
                <div class="index-price" style="color:{vix_color};">{vix['price']:.2f}</div>
                <div class="{'index-change-up' if vix['change']>=0 else 'index-change-down'}">
                    {'▲' if vix['change']>=0 else '▼'} {abs(vix['pct']):.2f}%
                </div>
            </div>""", unsafe_allow_html=True)

        # Regime badge
        regime = fetch_regime()
        if regime:
            regime_colors = {'BULL': '#00ff88', 'BEAR': '#ff4444', 'SIDEWAYS': '#ffaa00', 'VOLATILE': '#ff6b35'}
            rc = regime_colors.get(regime.regime, '#8892b0')
            st.markdown(f"""
            <div class="index-card" style="border-color:{rc};">
                <div class="index-name">MARKET REGIME</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{rc};">
                    {regime.regime}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;">
                    Confidence: {regime.confidence:.0%} │ Breadth: {regime.breadth_score:.0%}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="margin:6px 0;">', unsafe_allow_html=True)

        # News Feed
        news = fetch_news_feed()
        render_news_feed(news)

    # ═══ CENTER COLUMN: Chart + Geo ═══
    with center_col:
        # Live, continuously-updating price header
        live_price_header(selected_symbol)

        # Candlestick Chart
        chart_data = fetch_stock_ohlcv(selected_symbol + ".NS", period=timeframe)
        render_candlestick_chart(chart_data, selected_symbol, indicators)

        # Geopolitical Intelligence with Map
        with st.expander("🌍 GEOPOLITICAL INTELLIGENCE", expanded=False):
            try:
                from src.sentiment.geopolitical import GeopoliticalMonitor
                monitor = GeopoliticalMonitor()
                report = monitor.get_risk_report()

                risk_color = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00ff88'}.get(
                    report.get('overall_risk', 'MEDIUM'), '#8892b0')

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;">RISK LEVEL</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;color:{risk_color};" class="glow-{'red' if report.get('overall_risk')=='HIGH' else 'green'}">
                            {report.get('overall_risk', 'N/A')}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;">RISK SCORE</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;color:#e2e8f0;">
                            {report.get('risk_score', 0):.1f}/10
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;">EVENTS</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;color:#e2e8f0;">
                            {report.get('total_events', 0)}
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Geo Map — conflict/tension/home markers
                geo_markers = [
                    {'lat': 20.5937, 'lon': 78.9629, 'name': 'India (Home)', 'color': '#00ff88', 'size': 18},
                    {'lat': 51.5074, 'lon': -0.1278, 'name': 'UK', 'color': '#4a9eff', 'size': 10},
                    {'lat': 38.9072, 'lon': -77.0369, 'name': 'USA', 'color': '#4a9eff', 'size': 12},
                    {'lat': 35.8617, 'lon': 104.1954, 'name': 'China', 'color': '#ff9f43', 'size': 12},
                    {'lat': 55.7558, 'lon': 37.6173, 'name': 'Russia', 'color': '#ff4444', 'size': 11},
                    {'lat': 36.2048, 'lon': 138.2529, 'name': 'Japan', 'color': '#4a9eff', 'size': 9},
                    {'lat': 23.6345, 'lon': 53.0818, 'name': 'UAE (Oil)', 'color': '#ff9f43', 'size': 9},
                    {'lat': 48.8566, 'lon': 2.3522, 'name': 'EU', 'color': '#4a9eff', 'size': 10},
                ]
                # Color event hotspots based on risk
                events = report.get('events', [])
                for ev in events[:5]:
                    if hasattr(ev, 'risk_score') and ev.risk_score > 6:
                        geo_markers.append({
                            'lat': 33.0 + np.random.uniform(-5, 5),
                            'lon': 44.0 + np.random.uniform(-10, 10),
                            'name': f'⚠ {getattr(ev, "event_type", "Event")}',
                            'color': '#ff4444', 'size': 14,
                        })

                geo_fig = go.Figure()
                for m in geo_markers:
                    geo_fig.add_trace(go.Scattergeo(
                        lat=[m['lat']], lon=[m['lon']],
                        text=[m['name']], mode='markers+text',
                        marker=dict(size=m['size'], color=m['color'],
                                    line=dict(width=1, color='rgba(255,255,255,0.2)'),
                                    opacity=0.85),
                        textposition='top center',
                        textfont=dict(size=8, color=m['color'], family='JetBrains Mono'),
                        showlegend=False,
                    ))
                geo_fig.update_geos(
                    projection_type='natural earth',
                    showcoastlines=True, coastlinecolor='#1a2332',
                    showland=True, landcolor='#0d1220',
                    showocean=True, oceancolor='#080c14',
                    showlakes=False, showcountries=True, countrycolor='#1a2332',
                    bgcolor='#0a0e17',
                )
                geo_fig.update_layout(
                    paper_bgcolor='#0a0e17',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=220,
                    font=dict(family='JetBrains Mono', size=8, color='#6b7394'),
                )
                st.plotly_chart(geo_fig, use_container_width=True)

                if report.get('sectors_at_risk'):
                    st.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#ff4444;'>⚠ At Risk: {', '.join(report['sectors_at_risk'][:5])}</span>",
                                unsafe_allow_html=True)
                if report.get('sectors_to_buy'):
                    st.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#00ff88;'>✓ Opportunity: {', '.join(report['sectors_to_buy'][:5])}</span>",
                                unsafe_allow_html=True)
            except Exception as e:
                st.caption(f"Geo data unavailable: {e}")

    # ═══ RIGHT COLUMN: Order Book + Trading Panel ═══
    with right_col:
        # Live, continuously-updating order book depth
        live_order_book(selected_symbol)

        st.markdown('<hr style="margin:6px 0;">', unsafe_allow_html=True)

        # Trading Panel
        rp = fetch_live_quote(selected_symbol)
        if rp and rp.get("ok"):
            current_price = rp["price"]
        else:
            fb = fetch_index_data(selected_symbol + ".NS")
            current_price = fb["price"] if fb["ok"] else 1000
        render_trading_panel(selected_symbol, current_price)

    if run_ai:
        with st.spinner(f"🧠 Running AI Decision Engine on {selected_symbol}..."):
            decision = run_decision_engine(selected_symbol)
        st.session_state['trade_ai_decision'] = decision
        st.session_state['trade_ai_symbol'] = selected_symbol

    if run_backtest:
        with st.spinner(f"⚡ Running full-strategy backtest on {selected_symbol} (2 years)..."):
            bt_result = run_single_stock_backtest(selected_symbol)
        st.session_state['trade_backtest_result'] = bt_result
        st.session_state['trade_backtest_symbol'] = selected_symbol

    decision = st.session_state.get('trade_ai_decision')
    decision_symbol = st.session_state.get('trade_ai_symbol')
    if decision:
        st.markdown('<hr style="margin:6px 0;">', unsafe_allow_html=True)
        ai_header_col1, ai_header_col2 = st.columns([6, 1])
        with ai_header_col1:
            st.markdown(
                f"<div class=\"panel-header\">🧠 AI DECISION ENGINE: {decision_symbol or selected_symbol}</div>",
                unsafe_allow_html=True,
            )
        with ai_header_col2:
            if st.button("CLEAR AI", key="clear_ai_result", use_container_width=True):
                st.session_state.pop('trade_ai_decision', None)
                st.session_state.pop('trade_ai_symbol', None)
                st.rerun()
        ai_col1, ai_col2 = st.columns([1, 2])
        with ai_col1:
            render_decision_engine_results(decision)
        with ai_col2:
            fig = go.Figure(data=go.Scatterpolar(
                r=[decision.signal_score/25*100, decision.ml_score/20*100,
                   decision.kelly_score/20*100, decision.regime_score/15*100,
                   decision.event_score/10*100, decision.geo_score/10*100],
                theta=['Signal', 'ML', 'Kelly', 'Regime', 'Events', 'Geo'],
                fill='toself',
                fillcolor='rgba(74, 158, 255, 0.15)',
                line=dict(color='#4a9eff', width=2),
                marker=dict(size=6, color='#4a9eff'),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='#0d1220',
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1a2332',
                                    tickfont=dict(size=8, color='#4a5568')),
                    angularaxis=dict(gridcolor='#1a2332',
                                     tickfont=dict(size=9, color='#6b7394', family='JetBrains Mono')),
                ),
                paper_bgcolor='#0a0e17',
                height=320,
                margin=dict(l=40, r=40, t=30, b=30),
                title=dict(text="CONVICTION RADAR", font=dict(size=11, color='#4a9eff',
                           family='JetBrains Mono'), x=0.5),
            )
            st.plotly_chart(fig, use_container_width=True)

    bt_result = st.session_state.get('trade_backtest_result')
    bt_symbol = st.session_state.get('trade_backtest_symbol')
    if bt_result:
        st.markdown('<hr style="margin:6px 0;">', unsafe_allow_html=True)
        bt_header_col1, bt_header_col2 = st.columns([6, 1])
        strategy_name = bt_result.get('strategy_name', 'Multi-Factor Strategy')
        quality = bt_result.get('quality_score', 5.0)
        avg_hold = bt_result.get('avg_hold_days', 0)
        with bt_header_col1:
            st.markdown(f"""<div class="panel-header">⚡ BACKTEST: {bt_symbol or selected_symbol} — Full Strategy (2Y)</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;margin-bottom:4px;">
                <b style="color:#4a9eff;">Strategy:</b> {strategy_name}<br>
                <b style="color:#4a9eff;">Logic:</b> Composite Score = 0.40×Momentum + 0.25×Quality + 0.20×Sentiment + 0.15×SmartMoney<br>
                <b style="color:#4a9eff;">Entry:</b> Weekly rebalance — BUY when composite &gt; 6.0/10 │ 
                <b style="color:#4a9eff;">Exit:</b> Score drops &lt; 4.0 OR 2×ATR stop OR trailing stop (5% from peak)<br>
                <b style="color:#4a9eff;">Sizing:</b> Risk 2% capital per trade (Kelly-inspired) │ 
                <b style="color:#4a9eff;">Quality Score:</b> {quality:.1f}/10 │ 
                <b style="color:#4a9eff;">Avg Hold:</b> {avg_hold:.0f} days │ 
                <b style="color:#6b7394;">Note:</b> Sentiment &amp; Smart Money set to neutral (5/10) — no free historical alt-data
            </div>""", unsafe_allow_html=True)
        with bt_header_col2:
            if st.button("CLEAR BT", key="clear_backtest_result", use_container_width=True):
                st.session_state.pop('trade_backtest_result', None)
                st.session_state.pop('trade_backtest_symbol', None)
                st.rerun()
        bt_cols = st.columns(6)
        metrics = [
            ("Total Return", f"{bt_result['total_return']:.1f}%", '#00ff88' if bt_result['total_return'] > 0 else '#ff4444'),
            ("Win Rate", f"{bt_result['win_rate']:.0f}%", '#00ff88' if bt_result['win_rate'] > 50 else '#ff4444'),
            ("Trades", f"{bt_result['num_trades']}", '#4a9eff'),
            ("Profit Factor", f"{bt_result['profit_factor']:.2f}", '#00ff88' if bt_result['profit_factor'] > 1 else '#ff4444'),
            ("Max Drawdown", f"{bt_result['max_drawdown']:.1f}%", '#ff4444'),
            ("Sharpe Ratio", f"{bt_result['sharpe']:.2f}", '#00ff88' if bt_result['sharpe'] > 1 else '#ffaa00'),
        ]
        for i, (label, value, color) in enumerate(metrics):
            with bt_cols[i]:
                st.markdown(f"""
                <div class="index-card">
                    <div class="index-name">{label}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{color};">{value}</div>
                </div>""", unsafe_allow_html=True)
        if 'equity_curve' in bt_result and len(bt_result['equity_curve']) > 1:
            eq_fig = go.Figure()
            x_dates = bt_result.get('dates', list(range(len(bt_result['equity_curve']))))
            eq_fig.add_trace(go.Scatter(
                x=x_dates,
                y=bt_result['equity_curve'],
                mode='lines',
                line=dict(color='#4a9eff', width=2),
                fill='tozeroy', fillcolor='rgba(74,158,255,0.06)',
                name='Portfolio Value (₹)',
                hovertemplate='Date: %{x}<br>Portfolio: ₹%{y:,.0f}<extra></extra>',
            ))
            eq_fig.add_hline(
                y=bt_result.get('capital', 1000000),
                line_dash="dash", line_color="#6b7394", line_width=1,
                annotation_text="Starting Capital ₹10,00,000",
                annotation_position="top left",
                annotation_font=dict(size=9, color='#6b7394', family='JetBrains Mono'),
            )
            for t in bt_result.get('trades', []):
                t_color = '#00ff88' if t['P&L ₹'] > 0 else '#ff4444'
                if t['Entry Date'] in x_dates:
                    idx = x_dates.index(t['Entry Date'])
                    eq_fig.add_trace(go.Scatter(
                        x=[t['Entry Date']], y=[bt_result['equity_curve'][idx]],
                        mode='markers', marker=dict(size=8, color='#00ff88', symbol='triangle-up'),
                        showlegend=False,
                        hovertemplate=f"<b>BUY</b> @ ₹{t['Entry ₹']:,.0f}<br>Score: {t.get('Entry Score', 'N/A')}<extra></extra>",
                    ))
                if t['Exit Date'] in x_dates:
                    idx = x_dates.index(t['Exit Date'])
                    eq_fig.add_trace(go.Scatter(
                        x=[t['Exit Date']], y=[bt_result['equity_curve'][idx]],
                        mode='markers', marker=dict(size=8, color=t_color, symbol='triangle-down'),
                        showlegend=False,
                        hovertemplate=f"<b>SELL</b> @ ₹{t['Exit ₹']:,.0f}<br>P&L: {t['Return %']:+.1f}%<br>Reason: {t['Exit Reason']}<extra></extra>",
                    ))
            eq_fig.update_layout(
                title=dict(
                    text=f"EQUITY CURVE — {bt_symbol or selected_symbol} │ ₹10,00,000 → ₹{bt_result['equity_curve'][-1]:,.0f} ({bt_result['total_return']:+.1f}%)",
                    font=dict(size=11, color='#4a9eff', family='JetBrains Mono'),
                    x=0.5,
                ),
                paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220',
                height=300, margin=dict(l=70, r=20, t=45, b=55),
                yaxis=dict(
                    title=dict(text='Portfolio Value (₹)', font=dict(size=10, color='#8892b0', family='JetBrains Mono')),
                    gridcolor='#1a2332', tickfont=dict(size=9, color='#6b7394', family='JetBrains Mono'),
                    tickformat=',.0f', tickprefix='₹',
                ),
                xaxis=dict(
                    title=dict(text='Date (Weekly Rebalance)', font=dict(size=10, color='#8892b0', family='JetBrains Mono')),
                    gridcolor='#1a2332', tickfont=dict(size=8, color='#6b7394', family='JetBrains Mono'),
                    tickangle=-30, dtick='M2',
                ),
                font=dict(family='JetBrains Mono'),
                showlegend=False,
                hovermode='x unified',
            )
            st.plotly_chart(eq_fig, use_container_width=True)
        elif bt_result['num_trades'] == 0:
            st.warning("⚠️ No trades generated. This stock's composite score never exceeded 6.0 in the last 2 years (weak momentum + fundamentals). Try a high-momentum stock like BAJFINANCE, TITAN, or BHARTIARTL.")
        if 'trades' in bt_result and bt_result['trades']:
            with st.expander(f"📋 Trade Log ({len(bt_result['trades'])} trades) — Entry Score shows composite at time of entry", expanded=False):
                trade_df = pd.DataFrame(bt_result['trades'])
                st.dataframe(trade_df, use_container_width=True, height=250)

    # ─── BOTTOM ROW: Strategies + Heatmap ───
    st.markdown('<hr style="margin:4px 0;">', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([3, 2])

    with bottom_left:
        render_strategy_signals(selected_symbol, chart_data)

    with bottom_right:
        sector_data = fetch_sector_returns()
        render_sector_heatmap(sector_data)

    # ─── AUTO-REFRESH (full reload every 60s; live panels update via fragments) ───
    st.markdown("""
    <script>
        // Full reload every 60 seconds for indices/news; price, order book and
        // ticker update continuously via Streamlit fragments.
        if (!window._warRoomTimer) {
            window._warRoomTimer = setTimeout(function(){
                window._warRoomTimer = null;
                window.location.reload();
            }, 60000);
        }
    </script>
    """, unsafe_allow_html=True)

    # Footer with version + disclaimer
    st.markdown(f"""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                color:#2d3748;margin-top:10px;border-top:1px solid #1a2332;padding-top:6px;">
        WAR ROOM TERMINAL v2.2 │ Data via yfinance │
        Live price 3s · Order book 2s · Ticker 5s · Full reload 60s │ {datetime.now().strftime('%H:%M:%S')} IST
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
