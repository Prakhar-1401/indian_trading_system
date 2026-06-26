"""
Bloomberg-Style Trading Terminal
=================================
Professional multi-panel terminal dashboard.
One screen. All decisions. No switching.

RUN:
    streamlit run src/dashboard/terminal.py

LAYOUT:
    ┌─────────────────────────────────────────────────────────────┐
    │  TOP BAR: NIFTY | BANKNIFTY | VIX | REGIME | GEO | BREADTH │
    ├──────────────────────┬──────────────────────────────────────┤
    │  LEFT: WATCHLIST     │  CENTER: CHART + INDICATORS          │
    │  Decision verdicts   │  Selected stock candlestick          │
    │  TAKE/WATCH/SKIP     │  + Bollinger + Volume                │
    ├──────────────────────┼──────────────────────────────────────┤
    │  TRADE ARCHITECT     │  RIGHT: DECISION PANEL               │
    │  Entry/Stop/Target   │  Score breakdown + Conviction        │
    │  Position sizing     │  Risk Radar                          │
    ├──────────────────────┴──────────────────────────────────────┤
    │  BOTTOM: NEWS FEED | SECTOR HEATMAP | PAPER PORTFOLIO       │
    └─────────────────────────────────────────────────────────────┘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import json
from datetime import datetime, date
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.fetcher import DataManager
from src.indicators.technical import get_all_indicators, compute_momentum_score
from src.utils.helpers import load_config

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG — Dark terminal aesthetic
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for terminal look
st.markdown("""
<style>
    /* Dark terminal background */
    .stApp {
        background-color: #0a0e17;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Terminal-style metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 1.1rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8892b0;
    }
    
    /* Compact dataframes */
    .dataframe {
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2a3a;
    }
    
    /* Card-style containers */
    .terminal-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #1e2a3a;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }
    
    /* Status indicators */
    .status-take { color: #00ff88; font-weight: bold; }
    .status-watch { color: #ffaa00; font-weight: bold; }
    .status-skip { color: #ff4444; }
    
    /* Ticker strip */
    .ticker-strip {
        background: #0d1117;
        border-bottom: 1px solid #1e2a3a;
        padding: 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider styling */
    hr {
        border-color: #1e2a3a;
        margin: 0.5rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0d1117;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 6px 12px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=120)
def load_market_overview():
    """Load NIFTY, BANKNIFTY and market breadth data."""
    dm = DataManager()
    nifty = dm.get_stock_data("^NSEI", period="5d")
    banknifty = dm.get_stock_data("^NSEBANK", period="5d")
    return nifty, banknifty


@st.cache_data(ttl=120)
def load_stock_data(symbol, period="1y"):
    dm = DataManager()
    return dm.get_stock_data(symbol, period=period)


@st.cache_data(ttl=300)
def load_decisions():
    """Run decision engine for all stocks."""
    try:
        from src.strategy.decision_engine import DecisionEngine
        engine = DecisionEngine()
        return engine.analyze_watchlist()
    except Exception as e:
        st.error(f"Decision engine error: {e}")
        return []


@st.cache_data(ttl=300)
def load_regime():
    """Get current market regime."""
    try:
        from src.strategy.regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        return detector.detect_regime()
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_sector_data():
    """Load sector performance."""
    try:
        from src.backtest.sector_analysis import SectorAnalyzer
        analyzer = SectorAnalyzer()
        return analyzer.get_sector_performance()
    except Exception:
        return None


def load_paper_portfolio():
    """Load paper trading portfolio."""
    portfolio_file = os.path.join("data", "paper_portfolio.json")
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# TOP BAR — Market Status Strip
# ═══════════════════════════════════════════════════════════════════
def render_top_bar():
    """Render the market status ticker strip."""
    nifty, banknifty = load_market_overview()
    regime = load_regime()
    
    cols = st.columns([2, 2, 2, 2, 2, 2])
    
    # NIFTY
    if not nifty.empty:
        nifty_price = nifty['close'].iloc[-1]
        nifty_chg = ((nifty['close'].iloc[-1] / nifty['close'].iloc[-2]) - 1) * 100 if len(nifty) > 1 else 0
        cols[0].metric("NIFTY 50", f"₹{nifty_price:,.0f}", f"{nifty_chg:+.2f}%")
    
    # BANKNIFTY
    if not banknifty.empty:
        bn_price = banknifty['close'].iloc[-1]
        bn_chg = ((banknifty['close'].iloc[-1] / banknifty['close'].iloc[-2]) - 1) * 100 if len(banknifty) > 1 else 0
        cols[1].metric("BANK NIFTY", f"₹{bn_price:,.0f}", f"{bn_chg:+.2f}%")
    
    # REGIME
    if regime:
        regime_map = {'BULL': '🟢 BULL', 'BEAR': '🔴 BEAR', 'SIDEWAYS': '🟡 SIDEWAYS', 'VOLATILE': '⚡ VOLATILE'}
        cols[2].metric("REGIME", regime_map.get(regime.regime, regime.regime), f"{regime.confidence}% conf")
    
    # VOLATILITY
    if regime:
        cols[3].metric("VOL %ILE", f"{regime.volatility_percentile:.0f}th",
                      "HIGH" if regime.volatility_percentile > 70 else "NORMAL")
    
    # TREND
    if regime:
        trend = regime.trend_score
        cols[4].metric("TREND", f"{trend:+.2f}", "Bullish" if trend > 0.2 else "Bearish" if trend < -0.2 else "Neutral")
    
    # BREADTH
    if regime:
        cols[5].metric("BREADTH", f"{regime.breadth_score:.0f}%", "Above 50-DMA")
    
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# DECISION PANEL — Watchlist with verdicts
# ═══════════════════════════════════════════════════════════════════
def render_decision_panel():
    """Render the TAKE/WATCH/SKIP decision panel."""
    st.markdown("### 🧠 Decision Verdicts")
    
    decisions = load_decisions()
    
    if not decisions:
        st.info("Loading decisions... This may take a moment on first run.")
        return None
    
    takes = [d for d in decisions if d.verdict == "TAKE"]
    watches = [d for d in decisions if d.verdict == "WATCH"]
    skips = [d for d in decisions if d.verdict == "SKIP"]
    
    # TAKE section
    if takes:
        for d in takes:
            st.markdown(f"""
            <div style="background:#0a2e1a; border:1px solid #00ff88; border-radius:6px; padding:8px; margin:4px 0;">
                <span style="color:#00ff88; font-weight:bold; font-family:monospace;">✅ TAKE</span>
                <span style="color:#ffffff; font-weight:bold; margin-left:10px;">{d.symbol}</span>
                <span style="color:#8892b0; margin-left:10px;">Score: {d.score:.0f} | Conv: {d.conviction}/5</span>
            </div>
            """, unsafe_allow_html=True)
    
    # WATCH section
    if watches:
        for d in watches[:5]:
            st.markdown(f"""
            <div style="background:#2e2a0a; border:1px solid #ffaa00; border-radius:6px; padding:8px; margin:4px 0;">
                <span style="color:#ffaa00; font-weight:bold; font-family:monospace;">👁 WATCH</span>
                <span style="color:#ffffff; margin-left:10px;">{d.symbol}</span>
                <span style="color:#8892b0; margin-left:10px;">Score: {d.score:.0f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # SKIP section (collapsed)
    if skips:
        with st.expander(f"🔴 SKIP ({len(skips)} stocks)"):
            for d in skips:
                reason = d.reasons_against[0] if d.reasons_against else "Low score"
                st.markdown(f"<span style='color:#ff4444; font-family:monospace;'>{d.symbol}</span> — {reason}",
                           unsafe_allow_html=True)
    
    # Return selected stock for detail view
    all_symbols = [d.symbol for d in decisions]
    selected_idx = st.selectbox("Select stock for detail", range(len(all_symbols)),
                               format_func=lambda i: f"{decisions[i].verdict} | {all_symbols[i]} ({decisions[i].score:.0f})")
    return decisions[selected_idx] if decisions else None


# ═══════════════════════════════════════════════════════════════════
# CHART PANEL — Price + Indicators
# ═══════════════════════════════════════════════════════════════════
def render_chart(symbol: str):
    """Render professional candlestick chart with indicators."""
    st.markdown(f"### 📈 {symbol}")
    
    data = load_stock_data(symbol, period="6mo")
    if data.empty:
        st.warning(f"No data for {symbol}")
        return
    
    indicators = get_all_indicators(data)
    
    # Create subplot with shared x-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=("", "Volume", "RSI"))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=indicators.index,
        open=indicators['open'], high=indicators['high'],
        low=indicators['low'], close=indicators['close'],
        name="Price",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Moving averages
    if 'sma_20' in indicators.columns:
        fig.add_trace(go.Scatter(x=indicators.index, y=indicators['sma_20'],
                                name="SMA 20", line=dict(color='#ffaa00', width=1)), row=1, col=1)
    if 'sma_50' in indicators.columns:
        fig.add_trace(go.Scatter(x=indicators.index, y=indicators['sma_50'],
                                name="SMA 50", line=dict(color='#00aaff', width=1)), row=1, col=1)
    
    # Bollinger Bands
    sma20 = indicators['close'].rolling(20).mean()
    std20 = indicators['close'].rolling(20).std()
    fig.add_trace(go.Scatter(x=indicators.index, y=sma20 + 2*std20,
                            name="BB Upper", line=dict(color='rgba(255,255,255,0.2)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=indicators.index, y=sma20 - 2*std20,
                            name="BB Lower", line=dict(color='rgba(255,255,255,0.2)', width=1),
                            fill='tonexty', fillcolor='rgba(100,150,255,0.05)'), row=1, col=1)
    
    # Volume
    colors = ['#00ff88' if c >= o else '#ff4444' for c, o in zip(indicators['close'], indicators['open'])]
    fig.add_trace(go.Bar(x=indicators.index, y=indicators['volume'],
                        name="Volume", marker_color=colors, opacity=0.6), row=2, col=1)
    
    # RSI
    if 'rsi' in indicators.columns:
        fig.add_trace(go.Scatter(x=indicators.index, y=indicators['rsi'],
                                name="RSI", line=dict(color='#aa88ff', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0d1117',
        height=500,
        margin=dict(l=50, r=20, t=30, b=30),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        font=dict(family="JetBrains Mono, monospace", size=10)
    )
    fig.update_xaxes(gridcolor='#1e2a3a', zeroline=False)
    fig.update_yaxes(gridcolor='#1e2a3a', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TRADE ARCHITECT — Full trade plan
# ═══════════════════════════════════════════════════════════════════
def render_trade_architect(decision):
    """Render the trade plan panel."""
    st.markdown("### 🎯 Trade Architect")
    
    if not decision or decision.entry_price == 0:
        st.info("Select a TAKE/WATCH stock to see trade plan")
        return
    
    # Score breakdown radar
    categories = ['Signal', 'ML', 'Kelly', 'Regime', 'Events', 'Geo']
    values = [
        decision.signal_score / 25 * 100,
        decision.ml_score / 20 * 100,
        decision.kelly_score / 20 * 100,
        decision.regime_score / 15 * 100,
        decision.event_score / 10 * 100,
        decision.geo_score / 10 * 100,
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 255, 136, 0.1)',
        line_color='#00ff88'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#0d1117',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e2a3a'),
            angularaxis=dict(gridcolor='#1e2a3a'),
        ),
        template="plotly_dark",
        paper_bgcolor='#0a0e17',
        height=250,
        margin=dict(l=40, r=40, t=20, b=20),
        font=dict(family="JetBrains Mono", size=10),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade details
    if decision.entry_price > 0:
        col1, col2 = st.columns(2)
        col1.metric("Entry", f"₹{decision.entry_price:,.0f}")
        col2.metric("Stop Loss", f"₹{decision.stop_loss:,.0f}",
                   f"-{((decision.entry_price - decision.stop_loss) / decision.entry_price * 100):.1f}%")
        
        col1, col2 = st.columns(2)
        col1.metric("Target", f"₹{decision.target:,.0f}",
                   f"+{((decision.target - decision.entry_price) / decision.entry_price * 100):.1f}%")
        col2.metric("R:R", f"1:{decision.risk_reward:.1f}")
        
        col1, col2 = st.columns(2)
        col1.metric("Qty", f"{decision.position_size} shares")
        col2.metric("Max Loss", f"₹{decision.max_loss:,.0f}")
        
        st.markdown(f"**Position Value**: ₹{decision.position_value:,.0f}")
    
    # Conviction meter
    st.markdown("#### Conviction")
    conviction_bar = "🟢" * decision.conviction + "⚫" * (5 - decision.conviction)
    st.markdown(f"<h3 style='letter-spacing:5px;'>{conviction_bar}</h3>", unsafe_allow_html=True)
    
    # Reasons
    if decision.reasons_for:
        st.markdown("**✅ For:**")
        for r in decision.reasons_for:
            st.markdown(f"- {r}")
    if decision.reasons_against:
        st.markdown("**⚠️ Against:**")
        for r in decision.reasons_against:
            st.markdown(f"- {r}")


# ═══════════════════════════════════════════════════════════════════
# SECTOR HEATMAP
# ═══════════════════════════════════════════════════════════════════
def render_sector_heatmap():
    """Render sector performance heatmap."""
    performance = load_sector_data()
    if performance is None:
        st.info("Loading sector data...")
        return
    
    perf_df = pd.DataFrame(performance).T
    if '1m_return' in perf_df.columns:
        fig = px.bar(
            perf_df.sort_values('1m_return', ascending=True).reset_index(),
            x='1m_return', y='index', orientation='h',
            color='1m_return', color_continuous_scale='RdYlGn',
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e17',
            plot_bgcolor='#0d1117',
            height=250,
            margin=dict(l=5, r=5, t=5, b=5),
            showlegend=False,
            yaxis_title="", xaxis_title="1M Return %",
            font=dict(family="JetBrains Mono", size=9),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAPER PORTFOLIO
# ═══════════════════════════════════════════════════════════════════
def render_portfolio():
    """Render paper trading portfolio."""
    portfolio = load_paper_portfolio()
    if not portfolio:
        st.info("No paper trades yet. Use `python main.py paper buy SYMBOL`")
        return
    
    initial = portfolio.get('initial_capital', 1000000)
    cash = portfolio.get('cash', initial)
    positions = portfolio.get('positions', {})
    total_invested = sum(p.get('quantity', 0) * p.get('entry_price', 0) for p in positions.values())
    total_value = cash + total_invested
    pnl = total_value - initial
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Value", f"₹{total_value:,.0f}", f"{(pnl/initial)*100:+.2f}%")
    col2.metric("Cash", f"₹{cash:,.0f}")
    col3.metric("Positions", len(positions))
    
    if positions:
        pos_data = []
        for sym, p in positions.items():
            pos_data.append({
                'Symbol': sym,
                'Qty': p['quantity'],
                'Entry': f"₹{p['entry_price']:,.0f}",
                'Stop': f"₹{p.get('stop_loss', 0):,.0f}",
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# ICICI DIRECT EXECUTION PANEL
# ═══════════════════════════════════════════════════════════════════
def render_execution_panel(decision):
    """Render semi-manual execution panel for ICICI Direct."""
    st.markdown("### 🏦 ICICI Direct Order")
    
    if not decision or decision.verdict == "SKIP":
        st.info("Select a TAKE stock to generate order ticket")
        return
    
    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #1e2a3a; border-radius:8px; padding:16px;">
        <table style="width:100%; font-family:monospace; color:#e6e6e6;">
            <tr><td style="color:#8892b0;">Symbol</td><td style="font-weight:bold; color:#00ff88;">{decision.symbol}</td></tr>
            <tr><td style="color:#8892b0;">Action</td><td style="color:#00ff88;">BUY</td></tr>
            <tr><td style="color:#8892b0;">Qty</td><td>{decision.position_size}</td></tr>
            <tr><td style="color:#8892b0;">Order Type</td><td>LIMIT</td></tr>
            <tr><td style="color:#8892b0;">Price</td><td>₹{decision.entry_price:,.2f}</td></tr>
            <tr><td style="color:#8892b0;">Stop Loss</td><td style="color:#ff4444;">₹{decision.stop_loss:,.2f}</td></tr>
            <tr><td style="color:#8892b0;">Target</td><td style="color:#00ff88;">₹{decision.target:,.2f}</td></tr>
            <tr><td style="color:#8892b0;">Max Risk</td><td style="color:#ffaa00;">₹{decision.max_loss:,.0f}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("**Pre-Trade Checklist:**")
    c1 = st.checkbox("✅ Regime allows this trade")
    c2 = st.checkbox("✅ No earnings in 5 days")
    c3 = st.checkbox("✅ Position size within limits")
    c4 = st.checkbox("✅ Stop-loss order will be placed")
    
    if c1 and c2 and c3 and c4:
        st.success("✅ All checks passed. Place order on ICICI Direct app.")
        st.markdown(f"[Open ICICI Direct](https://secure.icicidirect.com/)")
    else:
        st.warning("Complete all checks before placing order")


# ═══════════════════════════════════════════════════════════════════
# NEWS FEED
# ═══════════════════════════════════════════════════════════════════
def render_news_feed():
    """Render live news with sentiment tags."""
    try:
        from src.sentiment.geopolitical import GeopoliticalMonitor
        monitor = GeopoliticalMonitor()
        articles = monitor.fetch_articles()
        
        if articles:
            for article in articles[:8]:
                title = article.get('title', 'No title')
                sentiment = article.get('sentiment', 0)
                
                if sentiment > 0.2:
                    tag = '<span style="color:#00ff88; font-size:0.7rem;">BULLISH</span>'
                elif sentiment < -0.2:
                    tag = '<span style="color:#ff4444; font-size:0.7rem;">BEARISH</span>'
                else:
                    tag = '<span style="color:#8892b0; font-size:0.7rem;">NEUTRAL</span>'
                
                st.markdown(f"{tag} {title[:80]}", unsafe_allow_html=True)
        else:
            st.info("No recent news")
    except Exception:
        st.info("News feed loading...")


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
def main():
    # Title
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0;">
        <span style="font-size:1.5rem;">⚡</span>
        <span style="font-family:'Inter',sans-serif; font-size:1.3rem; font-weight:700; color:#e6e6e6;">
            TRADING TERMINAL
        </span>
        <span style="font-family:monospace; font-size:0.75rem; color:#8892b0; margin-left:auto;">
            """ + datetime.now().strftime("%a %d %b %Y • %H:%M IST") + """
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # TOP BAR
    render_top_bar()
    
    # MAIN LAYOUT: Left (decisions) | Center (chart) | Right (trade plan)
    col_left, col_center, col_right = st.columns([2, 4, 2.5])
    
    with col_left:
        selected_decision = render_decision_panel()
    
    with col_center:
        if selected_decision:
            render_chart(selected_decision.symbol)
        else:
            render_chart("RELIANCE")
    
    with col_right:
        if selected_decision:
            render_trade_architect(selected_decision)
    
    # BOTTOM PANELS
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📰 News Feed", "🔥 Sectors", "📋 Portfolio", "🏦 Execute"])
    
    with tab1:
        render_news_feed()
    
    with tab2:
        render_sector_heatmap()
    
    with tab3:
        render_portfolio()
    
    with tab4:
        render_execution_panel(selected_decision)


if __name__ == "__main__":
    main()
