"""
Stock Picker Module — Scan universe, rank top 20, show trade plans.

Runs the full multi-factor strategy on 50 liquid NSE stocks and produces
actionable trade recommendations with Entry, Stop-Loss, and Target prices.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from loguru import logger


# Universe: Top 50 liquid NSE stocks (subset of NIFTY 50 + Next 50)
PICKER_UNIVERSE = [
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

SECTOR_MAP = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking", "INFY": "IT",
    "ICICIBANK": "Banking", "SBIN": "Banking", "BHARTIARTL": "Telecom",
    "ITC": "FMCG", "LT": "Infra", "BAJFINANCE": "Finance",
    "SUNPHARMA": "Pharma", "TITAN": "Consumer", "MARUTI": "Auto",
    "WIPRO": "IT", "HCLTECH": "IT", "TATAMOTORS": "Auto",
    "NTPC": "Power", "POWERGRID": "Power", "COALINDIA": "Mining",
    "NESTLEIND": "FMCG", "AXISBANK": "Banking", "KOTAKBANK": "Banking",
    "ULTRACEMCO": "Cement", "ONGC": "Energy", "TECHM": "IT",
    "ADANIENT": "Infra", "TATASTEEL": "Metals", "JSWSTEEL": "Metals",
    "HINDALCO": "Metals", "GRASIM": "Cement", "CIPLA": "Pharma",
    "DRREDDY": "Pharma", "BAJAJFINSV": "Finance", "EICHERMOT": "Auto",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Healthcare", "TATACONSUM": "FMCG",
    "HEROMOTOCO": "Auto", "SBILIFE": "Insurance", "BRITANNIA": "FMCG",
    "PIDILITIND": "Chemicals", "GODREJCP": "FMCG", "HDFCLIFE": "Insurance",
    "DABUR": "FMCG", "HAVELLS": "Consumer", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "IOC": "Energy", "BPCL": "Energy", "VEDL": "Mining",
}


def compute_stock_score(symbol: str) -> dict:
    """
    Compute full multi-factor score for one stock.
    Returns dict with all scores, prices, and trade plan.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="1y")
        if df.empty or len(df) < 200:
            return None

        df.columns = [c.lower() for c in df.columns]
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        n = len(close)

        # --- INDICATORS ---
        # RSI-14
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean().values
        loss_arr = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(loss_arr != 0, gain / loss_arr, 0)
        rsi = 100 - (100 / (1 + rs))

        # MACD
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

        # Volume
        vol_sma20 = pd.Series(volume, dtype=float).rolling(20).mean().values

        # --- MOMENTUM SCORE (0-10) ---
        i = n - 1  # Latest day
        m_score = 0
        max_raw = 12

        if 40 <= rsi[i] <= 65:
            m_score += 3
        elif 30 <= rsi[i] < 40 or 65 < rsi[i] <= 70:
            m_score += 1

        if macd_line[i] > signal_line[i]:
            m_score += 1
        if macd_hist[i] > macd_hist[i-1]:
            m_score += 1

        if not np.isnan(sma200[i]) and close[i] > sma200[i]:
            m_score += 1
        if not np.isnan(sma50[i]) and close[i] > sma50[i]:
            m_score += 1
        if not np.isnan(sma50[i]) and not np.isnan(sma200[i]) and sma50[i] > sma200[i]:
            m_score += 1

        if i >= 63:
            ret_3m = (close[i] / close[i-63] - 1) * 100
            if ret_3m > 15:
                m_score += 2
            elif ret_3m > 5:
                m_score += 1

        if i >= 126:
            ret_6m = (close[i] / close[i-126] - 1) * 100
            if ret_6m > 20:
                m_score += 1

        # Volume confirmation
        if not np.isnan(vol_sma20[i]) and vol_sma20[i] > 0:
            up_vol = sum(1 for j in range(i-4, i+1)
                        if close[j] > close[j-1] and volume[j] > vol_sma20[j] * 1.3)
            if up_vol >= 2:
                m_score += 1

        momentum_score = round((m_score / max_raw) * 10, 2)

        # --- QUALITY SCORE (0-10) from fundamentals ---
        quality_score = 5.0
        try:
            info = ticker.info
            roe = info.get('returnOnEquity', 0)
            de = info.get('debtToEquity', 0)
            margins = info.get('profitMargins', 0)
            pe = info.get('trailingPE', 0)
            rev_growth = info.get('revenueGrowth', 0)

            if roe and roe > 0:
                roe_pct = roe * 100 if roe < 1 else roe
                if roe_pct > 20: quality_score += 1.5
                elif roe_pct > 15: quality_score += 1.0
                elif roe_pct > 10: quality_score += 0.5
            if de is not None and de > 0:
                de_val = de / 100 if de > 10 else de
                if de_val < 0.3: quality_score += 1.0
                elif de_val < 0.8: quality_score += 0.5
            if margins and margins > 0:
                m_pct = margins * 100 if margins < 1 else margins
                if m_pct > 20: quality_score += 1.0
                elif m_pct > 10: quality_score += 0.5
            if pe and 10 < pe < 25:
                quality_score += 0.5
            if rev_growth and rev_growth > 0.15:
                quality_score += 0.5
            quality_score = min(quality_score, 10.0)
        except Exception:
            pass

        # --- SENTIMENT SCORE (live from RSS) ---
        sentiment_score = 5.0
        try:
            from src.sentiment.news_analyzer import NewsSentimentAnalyzer
            analyzer = NewsSentimentAnalyzer()
            result = analyzer.get_stock_sentiment(symbol)
            # Convert from -10..+10 to 0..10
            raw = result.get('sentiment_score', 0)
            sentiment_score = round((raw + 10) / 2, 2)
            sentiment_score = max(0, min(10, sentiment_score))
        except Exception:
            sentiment_score = 5.0

        # --- SMART MONEY SCORE ---
        smart_money_score = 5.0
        try:
            from src.smart_money.tracker import SmartMoneyTracker
            tracker = SmartMoneyTracker()
            smart_money_score = tracker.compute_smart_money_score(symbol)
            smart_money_score = max(0, min(10, smart_money_score))
        except Exception:
            smart_money_score = 5.0

        # --- COMPOSITE SCORE ---
        composite = round(
            0.40 * momentum_score +
            0.25 * quality_score +
            0.20 * sentiment_score +
            0.15 * smart_money_score, 2
        )

        # --- TRADE PLAN ---
        current_price = close[i]
        current_atr = atr[i] if not np.isnan(atr[i]) else current_price * 0.02

        entry_price = current_price
        stop_loss = round(entry_price - 2 * current_atr, 2)
        target_1 = round(entry_price + 1.5 * current_atr, 2)  # Conservative (1.5R)
        target_2 = round(entry_price + 3 * current_atr, 2)    # Aggressive (3R)
        risk_per_share = entry_price - stop_loss
        reward_1 = target_1 - entry_price
        rr_ratio = round(reward_1 / risk_per_share, 1) if risk_per_share > 0 else 0

        # Position sizing: risk 2% of ₹10L capital
        capital = 1000000
        risk_amount = capital * 0.02
        position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = position_size * entry_price

        # Signal strength
        if composite >= 7.0:
            signal = "STRONG BUY"
        elif composite >= 6.0:
            signal = "BUY"
        elif composite >= 5.0:
            signal = "MODERATE"
        else:
            signal = "WEAK"

        # Day change
        day_change = (close[i] / close[i-1] - 1) * 100 if i > 0 else 0

        return {
            'symbol': symbol,
            'sector': SECTOR_MAP.get(symbol, 'Other'),
            'price': round(current_price, 2),
            'day_change': round(day_change, 2),
            'composite': composite,
            'momentum': momentum_score,
            'quality': round(quality_score, 2),
            'sentiment': round(sentiment_score, 2),
            'smart_money': round(smart_money_score, 2),
            'entry': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target_1': target_1,
            'target_2': target_2,
            'risk_reward': rr_ratio,
            'position_size': position_size,
            'position_value': round(position_value, 0),
            'signal': signal,
            'rsi': round(rsi[i], 1),
            'atr': round(current_atr, 2),
            'atr_pct': round(current_atr / current_price * 100, 2),
        }
    except Exception as e:
        logger.debug(f"Score failed for {symbol}: {e}")
        return None


def run_stock_picker(use_sentiment: bool = False) -> list:
    """
    Run the full stock picker on the universe.
    Returns sorted list of stock scores.
    
    Args:
        use_sentiment: If True, fetches live sentiment (slower but more accurate)
    """
    results = []
    progress_bar = st.progress(0, text="Scanning stocks...")
    total = len(PICKER_UNIVERSE)

    for idx, symbol in enumerate(PICKER_UNIVERSE):
        progress_bar.progress((idx + 1) / total, text=f"Analyzing {symbol}... ({idx+1}/{total})")
        score = compute_stock_score(symbol)
        if score:
            results.append(score)

    progress_bar.empty()

    # Sort by composite score (highest first)
    results.sort(key=lambda x: x['composite'], reverse=True)
    return results[:20]  # Top 20


def render_stock_picker():
    """Render the Stock Picker UI."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            🎯 STOCK PICKER
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
            Scans 50 NSE stocks │ Multi-factor ranking │ Auto trade plans
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        run_picker = st.button("🚀 RUN STOCK PICKER", key="run_picker_btn", use_container_width=True)
    with ctrl_col2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#4a5568;padding-top:8px;">
            Scans 50 stocks × 4 factors │ ~2 min │ Shows Top 20 with trade plans
        </div>""", unsafe_allow_html=True)
    with ctrl_col3:
        capital_input = st.number_input("Capital ₹", value=1000000, step=100000,
                                        key="picker_capital", label_visibility="collapsed")

    if run_picker:
        results = run_stock_picker()

        if not results:
            st.error("No stocks scored. Check internet connection.")
            return

        # Store in session state
        st.session_state['picker_results'] = results
        st.session_state['picker_time'] = datetime.now().strftime('%H:%M:%S')

    # Display results (from session state if available)
    if 'picker_results' in st.session_state:
        results = st.session_state['picker_results']
        scan_time = st.session_state.get('picker_time', '')

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#6b7394;margin:6px 0;">
            Last scan: {scan_time} IST │ Universe: 50 stocks │ Showing Top {len(results)} │
            Strategy: Momentum(40%) + Quality(25%) + Sentiment(20%) + SmartMoney(15%)
        </div>
        """, unsafe_allow_html=True)

        # Summary bar
        strong_buys = sum(1 for r in results if r['signal'] == 'STRONG BUY')
        buys = sum(1 for r in results if r['signal'] == 'BUY')
        moderate = sum(1 for r in results if r['signal'] == 'MODERATE')

        sum_cols = st.columns(4)
        with sum_cols[0]:
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">STRONG BUY</div>
                <div style="font-family:'JetBrains Mono';font-size:1.2rem;font-weight:700;color:#00ff88;">{strong_buys}</div>
            </div>""", unsafe_allow_html=True)
        with sum_cols[1]:
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">BUY</div>
                <div style="font-family:'JetBrains Mono';font-size:1.2rem;font-weight:700;color:#4a9eff;">{buys}</div>
            </div>""", unsafe_allow_html=True)
        with sum_cols[2]:
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">MODERATE</div>
                <div style="font-family:'JetBrains Mono';font-size:1.2rem;font-weight:700;color:#ffaa00;">{moderate}</div>
            </div>""", unsafe_allow_html=True)
        with sum_cols[3]:
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">AVG SCORE</div>
                <div style="font-family:'JetBrains Mono';font-size:1.2rem;font-weight:700;color:#e2e8f0;">
                    {np.mean([r['composite'] for r in results]):.1f}/10
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="margin:8px 0;border-color:#1a2332;">', unsafe_allow_html=True)

        # Results table — render each stock as a card
        for rank, stock in enumerate(results, 1):
            signal_colors = {
                'STRONG BUY': '#00ff88',
                'BUY': '#4a9eff',
                'MODERATE': '#ffaa00',
                'WEAK': '#6b7394',
            }
            sig_color = signal_colors.get(stock['signal'], '#6b7394')
            day_color = '#00ff88' if stock['day_change'] >= 0 else '#ff4444'
            day_arrow = '▲' if stock['day_change'] >= 0 else '▼'

            # Risk amount
            risk_amt = stock['position_size'] * (stock['entry'] - stock['stop_loss'])

            st.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1a2332;border-left:3px solid {sig_color};
                        border-radius:4px;padding:8px 12px;margin-bottom:6px;
                        font-family:'JetBrains Mono',monospace;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <span style="color:#4a5568;font-size:0.8rem;font-weight:700;">#{rank}</span>
                        <span style="color:#e2e8f0;font-size:0.95rem;font-weight:700;">{stock['symbol']}</span>
                        <span style="color:#4a5568;font-size:0.65rem;">{stock['sector']}</span>
                        <span style="color:{sig_color};font-size:0.7rem;font-weight:600;
                                     background:{sig_color}15;padding:2px 8px;border-radius:3px;">
                            {stock['signal']}
                        </span>
                    </div>
                    <div style="display:flex;align-items:center;gap:16px;">
                        <span style="color:#e2e8f0;font-size:0.9rem;font-weight:600;">₹{stock['price']:,.2f}</span>
                        <span style="color:{day_color};font-size:0.75rem;">{day_arrow} {abs(stock['day_change']):.1f}%</span>
                        <span style="color:#4a9eff;font-size:0.85rem;font-weight:700;">
                            Score: {stock['composite']:.1f}/10
                        </span>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:6px;
                            padding-top:6px;border-top:1px solid #1a2332;">
                    <div>
                        <span style="color:#6b7394;font-size:0.6rem;">ENTRY</span>
                        <span style="color:#e2e8f0;font-size:0.75rem;margin-left:4px;">₹{stock['entry']:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#ff4444;font-size:0.6rem;">STOP-LOSS</span>
                        <span style="color:#ff4444;font-size:0.75rem;margin-left:4px;">₹{stock['stop_loss']:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#00ff88;font-size:0.6rem;">TARGET 1</span>
                        <span style="color:#00ff88;font-size:0.75rem;margin-left:4px;">₹{stock['target_1']:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#00ff88;font-size:0.6rem;">TARGET 2</span>
                        <span style="color:#00ff88;font-size:0.75rem;margin-left:4px;">₹{stock['target_2']:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#6b7394;font-size:0.6rem;">R:R</span>
                        <span style="color:#4a9eff;font-size:0.75rem;margin-left:4px;">{stock['risk_reward']}:1</span>
                    </div>
                    <div>
                        <span style="color:#6b7394;font-size:0.6rem;">QTY</span>
                        <span style="color:#e2e8f0;font-size:0.75rem;margin-left:4px;">{stock['position_size']} shares</span>
                    </div>
                    <div>
                        <span style="color:#6b7394;font-size:0.6rem;">RISK</span>
                        <span style="color:#ff4444;font-size:0.75rem;margin-left:4px;">₹{risk_amt:,.0f}</span>
                    </div>
                </div>
                <div style="display:flex;gap:16px;margin-top:4px;">
                    <span style="color:#4a5568;font-size:0.55rem;">
                        M:{stock['momentum']:.1f} │ Q:{stock['quality']:.1f} │ 
                        S:{stock['sentiment']:.1f} │ SM:{stock['smart_money']:.1f} │
                        RSI:{stock['rsi']:.0f} │ ATR:{stock['atr_pct']:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Export option
        st.markdown('<hr style="margin:8px 0;border-color:#1a2332;">', unsafe_allow_html=True)
        if st.button("📥 Export to CSV", key="export_picker"):
            export_df = pd.DataFrame(results)
            csv = export_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "stock_picks.csv", "text/csv")
