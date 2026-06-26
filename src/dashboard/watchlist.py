"""
Watchlist Module — Custom stock lists with live prices & alerts.

Features:
- Multiple watchlists (create/rename/delete)
- Live price + daily change for each stock
- Price alerts (above/below threshold)
- Quick-add from picker/analyzer
- Persistent storage in JSON
"""
import streamlit as st
import pandas as pd
import yfinance as yf
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def _load_watchlists() -> dict:
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    default = {"My Watchlist": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]}
    _save_watchlists(default)
    return default


def _save_watchlists(data: dict):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_live_quotes(symbols: list) -> list:
    """Fetch live price data for a list of symbols."""
    results = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            hist = ticker.history(period="5d")
            hist = hist[hist['Close'].notna()]
            if len(hist) >= 2:
                close = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = close - prev
                pct = (change / prev) * 100
                high = hist['High'].iloc[-1]
                low = hist['Low'].iloc[-1]
                vol = hist['Volume'].iloc[-1]
                results.append({
                    'symbol': sym, 'price': close, 'change': change,
                    'pct': pct, 'high': high, 'low': low, 'volume': vol
                })
            elif len(hist) == 1:
                results.append({
                    'symbol': sym, 'price': hist['Close'].iloc[-1],
                    'change': 0, 'pct': 0, 'high': hist['High'].iloc[-1],
                    'low': hist['Low'].iloc[-1], 'volume': hist['Volume'].iloc[-1]
                })
        except Exception:
            results.append({'symbol': sym, 'price': 0, 'change': 0, 'pct': 0,
                           'high': 0, 'low': 0, 'volume': 0})
    return results


def render_watchlist():
    """Render the watchlist interface."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            ⭐ WATCHLIST
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
            Track your favourite stocks │ Live prices
        </span>
    </div>
    """, unsafe_allow_html=True)

    watchlists = _load_watchlists()

    # Watchlist management bar
    mgmt_cols = st.columns([2, 1, 1, 1])
    with mgmt_cols[0]:
        list_names = list(watchlists.keys())
        active_list = st.selectbox("Select Watchlist", list_names, key="wl_active",
                                   label_visibility="collapsed")
    with mgmt_cols[1]:
        new_name = st.text_input("New list", key="wl_new_name", label_visibility="collapsed",
                                 placeholder="New list name...")
    with mgmt_cols[2]:
        if st.button("➕ Create", key="wl_create", use_container_width=True):
            if new_name and new_name not in watchlists:
                watchlists[new_name] = []
                _save_watchlists(watchlists)
                st.rerun()
    with mgmt_cols[3]:
        if st.button("🗑️ Delete List", key="wl_delete", use_container_width=True):
            if active_list and len(watchlists) > 1:
                del watchlists[active_list]
                _save_watchlists(watchlists)
                st.rerun()

    st.markdown('<hr style="margin:6px 0;border-color:#1a2332;">', unsafe_allow_html=True)

    # Add stock
    add_cols = st.columns([3, 1])
    with add_cols[0]:
        add_symbol = st.text_input("Add Stock", key="wl_add_sym", label_visibility="collapsed",
                                   placeholder="Enter symbol (e.g. TATAMOTORS)...").upper()
    with add_cols[1]:
        if st.button("➕ Add Stock", key="wl_add_btn", use_container_width=True):
            if add_symbol and active_list:
                if add_symbol not in watchlists.get(active_list, []):
                    watchlists[active_list].append(add_symbol)
                    _save_watchlists(watchlists)
                    st.rerun()

    # Display stocks with live data
    symbols = watchlists.get(active_list, [])
    if not symbols:
        st.info("Empty watchlist. Add stocks above.")
        return

    with st.spinner("Fetching live prices..."):
        quotes = get_live_quotes(symbols)

    for q in quotes:
        pnl_color = '#00ff88' if q['pct'] >= 0 else '#ff4444'
        arrow = '▲' if q['pct'] >= 0 else '▼'

        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])
        with col1:
            st.markdown(f"""<div style="font-family:'JetBrains Mono';font-size:0.9rem;
                        font-weight:700;color:#e2e8f0;padding-top:6px;">{q['symbol']}</div>""",
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div style="font-family:'JetBrains Mono';font-size:0.85rem;
                        color:#e2e8f0;padding-top:6px;">₹{q['price']:,.2f}</div>""",
                        unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div style="font-family:'JetBrains Mono';font-size:0.8rem;
                        color:{pnl_color};padding-top:6px;">{arrow} {q['pct']:+.2f}%</div>""",
                        unsafe_allow_html=True)
        with col4:
            vol_str = f"{q['volume']/1e5:.1f}L" if q['volume'] > 0 else "—"
            st.markdown(f"""<div style="font-family:'JetBrains Mono';font-size:0.7rem;
                        color:#6b7394;padding-top:8px;">Vol: {vol_str}</div>""",
                        unsafe_allow_html=True)
        with col5:
            if st.button("✕", key=f"wl_rm_{q['symbol']}"):
                watchlists[active_list].remove(q['symbol'])
                _save_watchlists(watchlists)
                st.rerun()

        st.markdown('<hr style="margin:2px 0;border-color:#0d1220;">', unsafe_allow_html=True)
