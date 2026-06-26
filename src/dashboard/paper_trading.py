"""
Paper Trading Module — Full brokerage-like trading simulator.

Features:
- BUY/SELL with Market/Limit/Stop-Loss orders
- Open positions with live P&L
- Order book (pending, executed, cancelled)
- Trade history with full log
- Portfolio analytics (win rate, Sharpe, equity curve)
- Persistent storage in JSON files
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import requests
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from loguru import logger

# Persistence paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"
ORDERS_FILE = DATA_DIR / "paper_orders.json"
TRADES_FILE = DATA_DIR / "paper_trades_history.json"

# Tradeable universe (mirrors the TRADE tab) for suggestion-based search
NSE_STOCKS = {
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


def _load_json(filepath: Path) -> dict:
    """Load JSON file or return default."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json(filepath: Path, data):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _normalize_portfolio(data: dict) -> dict:
    """Normalize legacy portfolio schemas into the current format."""
    if not isinstance(data, dict):
        return {}

    normalized = {
        'capital': float(data.get('capital', data.get('initial_capital', 1000000.0))),
        'available_cash': float(data.get('available_cash', data.get('cash', 1000000.0))),
        'positions': [],
        'created_at': data.get('created_at', datetime.now().isoformat()),
    }

    positions = data.get('positions', [])
    if isinstance(positions, dict):
        # Legacy schema: positions keyed by symbol
        for symbol, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            normalized['positions'].append({
                'symbol': symbol,
                'qty': int(pos.get('quantity', pos.get('qty', 0)) or 0),
                'avg_price': float(pos.get('entry_price', pos.get('avg_price', 0.0)) or 0.0),
                'side': 'LONG',
                'stop_loss': float(pos.get('stop_loss', 0.0) or 0.0),
                'target': float(pos.get('target', 0.0) or 0.0),
                'product': pos.get('product', 'CNC'),
                'entry_date': pos.get('entry_date', data.get('start_date', datetime.now().isoformat())),
                'commission': float(pos.get('commission', 0.0) or 0.0),
            })
    elif isinstance(positions, list):
        # Current schema: list of position dicts
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            normalized['positions'].append(pos)

    return normalized


def get_portfolio() -> dict:
    """Get current paper portfolio state."""
    default = {
        'capital': 1000000.0,
        'available_cash': 1000000.0,
        'positions': [],  # List of open positions
        'created_at': datetime.now().isoformat(),
    }
    data = _load_json(PORTFOLIO_FILE)
    if not data:
        _save_json(PORTFOLIO_FILE, default)
        return default

    normalized = _normalize_portfolio(data)
    if normalized and normalized != data:
        _save_json(PORTFOLIO_FILE, normalized)
    return normalized or default


def get_orders() -> list:
    """Get all orders."""
    data = _load_json(ORDERS_FILE)
    return data.get('orders', [])


def get_trade_history() -> list:
    """Get completed trades."""
    data = _load_json(TRADES_FILE)
    return data.get('trades', [])


def save_portfolio(portfolio: dict):
    """Save portfolio state."""
    _save_json(PORTFOLIO_FILE, portfolio)


def save_order(order: dict):
    """Add an order to order book."""
    data = _load_json(ORDERS_FILE)
    if 'orders' not in data:
        data['orders'] = []
    data['orders'].append(order)
    _save_json(ORDERS_FILE, data)


def save_trade(trade: dict):
    """Add a completed trade to history."""
    data = _load_json(TRADES_FILE)
    if 'trades' not in data:
        data['trades'] = []
    data['trades'].append(trade)
    _save_json(TRADES_FILE, data)


@st.cache_data(ttl=3, show_spinner=False)
def _groww_ltp(symbol: str) -> float:
    """Real near-real-time LTP from Groww's free public API (no 15-min delay)."""
    try:
        url = (f"https://groww.in/v1/api/stocks_data/v1/tr_live_prices/"
               f"exchange/NSE/segment/CASH/{quote(symbol)}/latest")
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "application/json",
        }, timeout=6)
        if r.status_code == 200:
            ltp = r.json().get("ltp")
            if ltp:
                return float(ltp)
    except Exception:
        pass
    return 0.0


def get_live_price(symbol: str) -> float:
    """Current price for a symbol: real Groww LTP first, yfinance as fallback."""
    p = _groww_ltp(symbol)
    if p and p > 0:
        return p
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="5d")
        hist = hist[hist['Close'].notna()]
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return 0.0


def _add_position(portfolio: dict, symbol: str, side: str, qty: int, price: float,
                  sl: float, target: float, product: str, commission: float):
    """Open a new position or average into an existing same-side position."""
    for p in portfolio['positions']:
        if p['symbol'] == symbol and p.get('side', 'LONG') == side:
            total_qty = p['qty'] + qty
            p['avg_price'] = (p['avg_price'] * p['qty'] + price * qty) / total_qty
            p['qty'] = total_qty
            p['commission'] = p.get('commission', 0.0) + commission
            if sl > 0:
                p['stop_loss'] = sl
            if target > 0:
                p['target'] = target
            return
    portfolio['positions'].append({
        'symbol': symbol, 'qty': qty, 'avg_price': price, 'side': side,
        'stop_loss': sl, 'target': target, 'product': product,
        'entry_date': datetime.now().isoformat(), 'commission': commission,
    })


def _record_trade(symbol: str, side: str, qty: int, entry: float, exit_price: float,
                  pnl: float, pos: dict, commission: float):
    """Persist a closed trade (LONG sell or SHORT cover) to history."""
    if side == 'SHORT':
        pnl_pct = (entry - exit_price) / entry * 100 if entry else 0.0
    else:
        pnl_pct = (exit_price / entry - 1) * 100 if entry else 0.0
    try:
        hold_days = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
    except Exception:
        hold_days = 0
    save_trade({
        'symbol': symbol, 'side': side, 'qty': qty,
        'entry_price': round(entry, 2), 'exit_price': round(exit_price, 2),
        'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2),
        'entry_date': pos.get('entry_date'),
        'exit_date': datetime.now().isoformat(),
        'hold_days': hold_days,
        'commission': round(pos.get('commission', 0.0) + commission, 2),
    })


def place_order(symbol: str, side: str, qty: int, order_type: str,
                price: float, stop_loss: float, target: float, product: str):
    """
    Place a paper trade order with full LONG + SHORT support.

      BUY  → covers an existing SHORT, otherwise opens/adds a LONG.
      SELL → closes an existing LONG, otherwise opens/adds a SHORT.

    MARKET orders execute immediately at the live LTP.
    LIMIT orders that aren't yet marketable are stored as PENDING.
    Shorts reserve the entry notional as margin (released on cover).
    """
    portfolio = get_portfolio()
    current_price = get_live_price(symbol)
    if current_price <= 0:
        return False, "Could not fetch price. Check symbol."

    # --- Determine execution price (resting LIMIT orders go to the order book) ---
    if order_type == "LIMIT":
        marketable = (side == "BUY" and price >= current_price) or \
                     (side == "SELL" and price <= current_price)
        if not marketable and price > 0:
            save_order({
                'id': f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'symbol': symbol, 'side': side, 'qty': qty, 'order_type': order_type,
                'price': price, 'stop_loss': stop_loss, 'target': target,
                'product': product, 'status': 'PENDING',
                'placed_at': datetime.now().isoformat(),
            })
            return True, f"Limit {side} order placed. Waiting for ₹{price:.2f}"
        exec_price = current_price
    else:
        exec_price = current_price

    existing = next((p for p in portfolio['positions'] if p['symbol'] == symbol), None)
    msg_parts = []
    remaining = qty

    if side == "BUY":
        # 1) Cover an existing short
        if existing and existing.get('side') == 'SHORT':
            close_qty = min(remaining, existing['qty'])
            comm = exec_price * close_qty * 0.0005
            pnl = (existing['avg_price'] - exec_price) * close_qty - comm
            _record_trade(symbol, 'SHORT', close_qty, existing['avg_price'], exec_price, pnl, existing, comm)
            # Release reserved margin + realised P&L
            portfolio['available_cash'] += existing['avg_price'] * close_qty + \
                (existing['avg_price'] - exec_price) * close_qty - comm
            if existing['qty'] == close_qty:
                portfolio['positions'].remove(existing)
            else:
                existing['qty'] -= close_qty
            msg_parts.append(f"Covered {close_qty} {symbol} @ ₹{exec_price:.2f} (P&L ₹{pnl:+,.0f})")
            remaining -= close_qty
            existing = None
        # 2) Open / add a long with whatever is left
        if remaining > 0:
            comm = exec_price * remaining * 0.0005
            total_cost = exec_price * remaining + comm
            if total_cost > portfolio['available_cash']:
                if msg_parts:
                    save_portfolio(portfolio)
                    return True, " | ".join(msg_parts) + " | (long not opened: insufficient funds)"
                return False, f"Insufficient funds. Need ₹{total_cost:,.0f}, have ₹{portfolio['available_cash']:,.0f}"
            _add_position(portfolio, symbol, 'LONG', remaining, exec_price, stop_loss, target, product, comm)
            portfolio['available_cash'] -= total_cost
            msg_parts.append(f"BUY {remaining} {symbol} @ ₹{exec_price:.2f}")

    else:  # SELL
        # 1) Close an existing long
        if existing and existing.get('side') == 'LONG':
            close_qty = min(remaining, existing['qty'])
            comm = exec_price * close_qty * 0.0005
            pnl = (exec_price - existing['avg_price']) * close_qty - comm
            _record_trade(symbol, 'LONG', close_qty, existing['avg_price'], exec_price, pnl, existing, comm)
            portfolio['available_cash'] += exec_price * close_qty - comm
            if existing['qty'] == close_qty:
                portfolio['positions'].remove(existing)
            else:
                existing['qty'] -= close_qty
            msg_parts.append(f"SOLD {close_qty} {symbol} @ ₹{exec_price:.2f} (P&L ₹{pnl:+,.0f})")
            remaining -= close_qty
            existing = None
        # 2) Open / add a short with whatever is left (reserve margin)
        if remaining > 0:
            comm = exec_price * remaining * 0.0005
            margin = exec_price * remaining + comm
            if margin > portfolio['available_cash']:
                if msg_parts:
                    save_portfolio(portfolio)
                    return True, " | ".join(msg_parts) + " | (short not opened: insufficient margin)"
                return False, f"Insufficient margin for short. Need ₹{margin:,.0f}, have ₹{portfolio['available_cash']:,.0f}"
            _add_position(portfolio, symbol, 'SHORT', remaining, exec_price, stop_loss, target, product, comm)
            portfolio['available_cash'] -= margin
            msg_parts.append(f"SHORT {remaining} {symbol} @ ₹{exec_price:.2f}")

    save_portfolio(portfolio)

    # Save the executed order to the order book
    save_order({
        'id': f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'symbol': symbol, 'side': side, 'qty': qty, 'order_type': order_type,
        'price': exec_price, 'stop_loss': stop_loss, 'target': target,
        'product': product, 'status': 'EXECUTED',
        'placed_at': datetime.now().isoformat(),
        'executed_at': datetime.now().isoformat(),
        'exec_price': exec_price,
    })
    return True, " | ".join(msg_parts) if msg_parts else f"{side} order processed."


def _render_analytics(trades: list, portfolio: dict):
    """Render performance analytics from completed trades."""
    import plotly.graph_objects as go

    trades_df = pd.DataFrame(trades)
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] <= 0]
    total_pnl = trades_df['pnl'].sum()

    # Row 1 metrics
    a_cols = st.columns(4)
    with a_cols[0]:
        wr = len(winning) / len(trades_df) * 100
        st.metric("Win Rate", f"{wr:.0f}%", f"{len(winning)}W / {len(losing)}L")
    with a_cols[1]:
        loss_sum = abs(losing['pnl'].sum())
        pf = winning['pnl'].sum() / loss_sum if loss_sum != 0 else float('inf')
        st.metric("Profit Factor", f"{pf:.2f}" if np.isfinite(pf) else "∞")
    with a_cols[2]:
        st.metric("Total Trades", len(trades_df))
    with a_cols[3]:
        st.metric("Net P&L", f"₹{total_pnl:+,.0f}",
                  f"{total_pnl / portfolio['capital'] * 100:+.2f}%")

    # Row 2 metrics
    b_cols = st.columns(4)
    avg_win = winning['pnl'].mean() if len(winning) else 0
    avg_loss = losing['pnl'].mean() if len(losing) else 0
    expectancy = trades_df['pnl'].mean()
    with b_cols[0]:
        st.metric("Avg Win", f"₹{avg_win:,.0f}")
    with b_cols[1]:
        st.metric("Avg Loss", f"₹{avg_loss:,.0f}")
    with b_cols[2]:
        st.metric("Expectancy / Trade", f"₹{expectancy:+,.0f}")
    with b_cols[3]:
        st.metric("Best / Worst", f"₹{trades_df['pnl'].max():+,.0f}",
                  f"₹{trades_df['pnl'].min():+,.0f}", delta_color="off")

    # --- Equity curve (starts at capital, auto-scaled so changes are visible) ---
    equity = [portfolio['capital']] + (trades_df['pnl'].cumsum() + portfolio['capital']).tolist()
    x_vals = list(range(len(equity)))
    line_color = '#00ff88' if equity[-1] >= portfolio['capital'] else '#ff4444'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=equity, mode='lines+markers',
        line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
        hovertemplate='Trade %{x}<br>Equity ₹%{y:,.0f}<extra></extra>',
    ))
    fig.add_hline(y=portfolio['capital'], line_dash="dash", line_color="#6b7394",
                  annotation_text="Start capital", annotation_position="bottom right",
                  annotation_font_color="#6b7394")
    lo, hi = min(equity), max(equity)
    pad = max((hi - lo) * 0.15, portfolio['capital'] * 0.001)
    fig.update_layout(
        title="Equity Curve (after each closed trade)",
        paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220',
        height=260, margin=dict(l=50, r=10, t=40, b=20),
        yaxis=dict(gridcolor='#1a2332', tickformat=',', range=[lo - pad, hi + pad]),
        xaxis=dict(gridcolor='#1a2332', title='Closed trade #'),
        font=dict(family='JetBrains Mono', size=10, color='#6b7394'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Per-trade P&L bars ---
    bar_colors = ['#00ff88' if v >= 0 else '#ff4444' for v in trades_df['pnl']]
    labels = [f"{r['symbol']} {r.get('side', '')}".strip() for _, r in trades_df.iterrows()]
    fig2 = go.Figure(go.Bar(
        x=list(range(len(trades_df))), y=trades_df['pnl'], marker_color=bar_colors,
        text=labels, hovertemplate='%{text}<br>P&L ₹%{y:,.0f}<extra></extra>',
    ))
    fig2.update_layout(
        title="P&L per Trade", paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220',
        height=220, margin=dict(l=50, r=10, t=40, b=20),
        yaxis=dict(gridcolor='#1a2332', tickformat=',', zerolinecolor='#2a3550'),
        xaxis=dict(gridcolor='#1a2332', title='Closed trade #'),
        font=dict(family='JetBrains Mono', size=10, color='#6b7394'),
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_paper_trading():
    """Render the full paper trading interface."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            📊 PAPER TRADING
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
            Simulate trades with ₹10L virtual capital │ Full order management
        </span>
    </div>
    """, unsafe_allow_html=True)

    portfolio = get_portfolio()

    # Portfolio summary bar (LONG = market value; SHORT = reserved margin + P&L)
    total_invested = 0.0
    total_current = 0.0
    for p in portfolio['positions']:
        price = get_live_price(p['symbol'])
        if price <= 0:
            price = p['avg_price']
        notional = p['avg_price'] * p['qty']
        if p.get('side') == 'SHORT':
            total_invested += notional
            total_current += notional + (p['avg_price'] - price) * p['qty']
        else:
            total_invested += notional
            total_current += price * p['qty']

    unrealized_pnl = total_current - total_invested
    total_value = portfolio['available_cash'] + total_current
    total_pnl = total_value - portfolio['capital']

    # Top metrics
    m_cols = st.columns(5)
    metrics = [
        ("Total Value", f"₹{total_value:,.0f}", '#e2e8f0'),
        ("Available Cash", f"₹{portfolio['available_cash']:,.0f}", '#4a9eff'),
        ("Invested", f"₹{total_invested:,.0f}", '#ffaa00'),
        ("Unrealized P&L", f"₹{unrealized_pnl:+,.0f}", '#00ff88' if unrealized_pnl >= 0 else '#ff4444'),
        ("Total P&L", f"₹{total_pnl:+,.0f} ({total_pnl/portfolio['capital']*100:+.1f}%)",
         '#00ff88' if total_pnl >= 0 else '#ff4444'),
    ]
    for i, (label, value, color) in enumerate(metrics):
        with m_cols[i]:
            st.markdown(f"""<div style="text-align:center;padding:6px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">{label}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.85rem;font-weight:700;color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="margin:8px 0;border-color:#1a2332;">', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Place Order", "📂 Open Positions", "📋 Order Book", "📈 Trade History", "📊 Analytics"
    ])

    # === TAB 1: PLACE ORDER ===
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            stock_options = [f"{sym} — {name}" for sym, name in sorted(NSE_STOCKS.items())]
            default_idx = stock_options.index("RELIANCE — Reliance Industries") \
                if "RELIANCE — Reliance Industries" in stock_options else 0
            selected = st.selectbox(
                "🔍 Search Stock", stock_options, index=default_idx, key="pt_symbol_sel",
                placeholder="Search symbol or company name...",
                help="Type a symbol or company name to filter",
            )
            symbol = selected.split(" — ")[0] if selected else "RELIANCE"
            side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="pt_side",
                            help="BUY opens a LONG (or covers a short). "
                                 "SELL opens a SHORT (or closes a long).")
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP-LOSS"], key="pt_otype")
            qty = st.number_input("Quantity", min_value=1, value=10, key="pt_qty")

        with col2:
            limit_price = st.number_input("Limit Price ₹", value=0.0, key="pt_limit",
                                          help="For LIMIT orders only")
            stop_loss = st.number_input("Stop Loss ₹", value=0.0, key="pt_sl")
            target = st.number_input("Target ₹", value=0.0, key="pt_target")
            product = st.selectbox("Product", ["CNC (Delivery)", "MIS (Intraday)"], key="pt_product")

        # Intent hint (long vs short)
        existing_pos = next((p for p in portfolio['positions'] if p['symbol'] == symbol), None)
        if side == "BUY":
            intent = "Cover SHORT" if existing_pos and existing_pos.get('side') == 'SHORT' else "Open / add LONG"
            intent_color = "#00ff88"
        else:
            intent = "Close LONG" if existing_pos and existing_pos.get('side') == 'LONG' else "Open SHORT (sell high → buy back lower)"
            intent_color = "#ff6b9d"
        st.markdown(
            f"<div style=\"font-family:'JetBrains Mono';font-size:0.65rem;color:{intent_color};"
            f"margin:2px 0 4px;\">▸ {intent}</div>", unsafe_allow_html=True)

        # Live price display
        live_price = get_live_price(symbol)
        if live_price > 0:
            order_value = live_price * qty
            risk_amt = (live_price - stop_loss) * qty if stop_loss > 0 else 0
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#6b7394;
                        padding:6px;border:1px solid #1a2332;border-radius:4px;margin:6px 0;">
                Live Price: <span style="color:#e2e8f0;font-weight:700;">₹{live_price:,.2f}</span> │
                Order Value: <span style="color:#4a9eff;">₹{order_value:,.0f}</span> │
                Risk: <span style="color:#ff4444;">₹{risk_amt:,.0f}</span>
            </div>""", unsafe_allow_html=True)

        if st.button("🚀 PLACE ORDER", key="place_order_btn", use_container_width=True):
            price = limit_price if order_type == "LIMIT" else live_price
            prod = "CNC" if "CNC" in product else "MIS"
            success, msg = place_order(symbol, side, qty, order_type, price, stop_loss, target, prod)
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
            st.rerun()

    # === TAB 2: OPEN POSITIONS ===
    with tab2:
        positions = portfolio.get('positions', [])
        if not positions:
            st.info("No open positions. Place a trade to get started.")
        else:
            for pos in positions:
                pside = pos.get('side', 'LONG')
                live = get_live_price(pos['symbol'])
                if live <= 0:
                    live = pos['avg_price']
                if pside == 'SHORT':
                    pnl = (pos['avg_price'] - live) * pos['qty']
                    pnl_pct = (pos['avg_price'] / live - 1) * 100 if live else 0
                else:
                    pnl = (live - pos['avg_price']) * pos['qty']
                    pnl_pct = (live / pos['avg_price'] - 1) * 100 if pos['avg_price'] else 0
                pnl_color = '#00ff88' if pnl >= 0 else '#ff4444'
                side_color = '#4a9eff' if pside == 'LONG' else '#ff6b9d'
                try:
                    days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
                except Exception:
                    days_held = 0

                st.markdown(f"""
                <div style="background:#0d1220;border:1px solid #1a2332;border-left:3px solid {pnl_color};
                            border-radius:4px;padding:8px 12px;margin-bottom:6px;
                            font-family:'JetBrains Mono',monospace;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:#e2e8f0;font-size:0.95rem;font-weight:700;">{pos['symbol']}</span>
                            <span style="color:{side_color};font-size:0.6rem;font-weight:700;margin-left:8px;
                                  border:1px solid {side_color};border-radius:3px;padding:1px 5px;">{pside}</span>
                            <span style="color:#4a9eff;font-size:0.7rem;margin-left:8px;">{pos['qty']} shares</span>
                            <span style="color:#4a5568;font-size:0.6rem;margin-left:8px;">{days_held}d held</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:{pnl_color};font-size:0.9rem;font-weight:700;">
                                ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)
                            </div>
                        </div>
                    </div>
                    <div style="display:flex;gap:20px;margin-top:4px;font-size:0.65rem;color:#6b7394;">
                        <span>Avg: ₹{pos['avg_price']:,.2f}</span>
                        <span>LTP: ₹{live:,.2f}</span>
                        <span>SL: ₹{pos.get('stop_loss', 0):,.2f}</span>
                        <span>Target: ₹{pos.get('target', 0):,.2f}</span>
                        <span>Value: ₹{live * pos['qty']:,.0f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Exit button — fires the opposite side (SELL to close long, BUY to cover short)
                exit_side = "SELL" if pside == 'LONG' else "BUY"
                exit_label = f"EXIT {pos['symbol']}" + (" (cover)" if pside == 'SHORT' else "")
                exit_col1, exit_col2, exit_col3 = st.columns([1, 1, 3])
                with exit_col1:
                    if st.button(exit_label, key=f"exit_{pos['symbol']}_{pos['entry_date']}"):
                        success, msg = place_order(pos['symbol'], exit_side, pos['qty'],
                                                   "MARKET", 0, 0, 0, pos.get('product', 'CNC'))
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    # === TAB 3: ORDER BOOK ===
    with tab3:
        orders = get_orders()
        if not orders:
            st.info("No orders placed yet.")
        else:
            orders_df = pd.DataFrame(orders[-20:])  # Last 20 orders
            display_cols = ['placed_at', 'symbol', 'side', 'qty', 'order_type', 'price', 'status']
            available_cols = [c for c in display_cols if c in orders_df.columns]
            st.dataframe(orders_df[available_cols].iloc[::-1], use_container_width=True, height=300)

    # === TAB 4: TRADE HISTORY ===
    with tab4:
        trades = get_trade_history()
        if not trades:
            st.info("No completed trades yet. Exit a position to see history.")
        else:
            trades_df = pd.DataFrame(trades)
            # Color P&L
            st.dataframe(trades_df.iloc[::-1], use_container_width=True, height=400)

    # === TAB 5: ANALYTICS ===
    with tab5:
        trades = get_trade_history()
        if len(trades) < 1:
            st.info("No completed trades yet. Close/exit a position to build analytics.")
        else:
            _render_analytics(trades, portfolio)

    # Reset button
    st.markdown('<hr style="margin:12px 0;border-color:#1a2332;">', unsafe_allow_html=True)
    if st.button("🗑️ Reset Portfolio (Start Fresh)", key="reset_portfolio"):
        PORTFOLIO_FILE.unlink(missing_ok=True)
        ORDERS_FILE.unlink(missing_ok=True)
        TRADES_FILE.unlink(missing_ok=True)
        st.session_state.clear()
        st.rerun()
