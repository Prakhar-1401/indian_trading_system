"""
Stock Analyzer Module — Full company deep-dive like TradingView/Tickertape.

Sections:
A) Company Profile (fundamentals, holdings, ratios)
B) Technical Dashboard (indicators summary, support/resistance)
C) News & Sentiment (live scraped)
D) Smart Money Activity
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from loguru import logger


def render_stock_analyzer(symbol: str):
    """Render full stock analysis page for the given symbol."""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            🔍 STOCK ANALYZER — {symbol}
        </span>
    </div>
    """, unsafe_allow_html=True)

    ticker = yf.Ticker(f"{symbol}.NS")

    # Fetch data
    try:
        info = ticker.info
        df = ticker.history(period="1y")
        df = df[df['Close'].notna()]
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return

    if df.empty:
        st.warning(f"No data available for {symbol}")
        return

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)

    # === SECTION A: COMPANY PROFILE ===
    with st.expander("🏢 COMPANY PROFILE & FUNDAMENTALS", expanded=True):
        # Basic info
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        mc_str = f"₹{market_cap/1e7:,.0f} Cr" if market_cap else "N/A"

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#e2e8f0;margin-bottom:8px;">
            <strong>{company_name}</strong> │ {sector} │ {industry} │ MCap: {mc_str}
        </div>""", unsafe_allow_html=True)

        # Key ratios
        f_cols = st.columns(6)
        fundamentals = [
            ("P/E Ratio", info.get('trailingPE', 'N/A'), "Lower = cheaper"),
            ("P/B Ratio", info.get('priceToBook', 'N/A'), "Lower = undervalued"),
            ("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else 'N/A', ">15% = good"),
            ("Debt/Equity", f"{info.get('debtToEquity', 0)/100:.2f}" if info.get('debtToEquity') else 'N/A', "<0.5 = safe"),
            ("Div Yield", f"{info.get('dividendYield', 0)*100:.1f}%" if info.get('dividendYield') else 'N/A', ">1% = bonus"),
            ("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else 'N/A', ">10% = good"),
        ]
        for i, (label, value, hint) in enumerate(fundamentals):
            with f_cols[i]:
                st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                    <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">{label}</div>
                    <div style="font-family:'JetBrains Mono';font-size:0.85rem;font-weight:700;color:#e2e8f0;">
                        {value}
                    </div>
                    <div style="font-family:'JetBrains Mono';font-size:0.45rem;color:#4a5568;">{hint}</div>
                </div>""", unsafe_allow_html=True)

        # Price stats
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        p_cols = st.columns(6)
        high_52w = info.get('fiftyTwoWeekHigh', max(high) if len(high) > 0 else 0)
        low_52w = info.get('fiftyTwoWeekLow', min(low) if len(low) > 0 else 0)
        current = close[-1] if len(close) > 0 else 0
        from_high = (current / high_52w - 1) * 100 if high_52w > 0 else 0
        from_low = (current / low_52w - 1) * 100 if low_52w > 0 else 0

        price_stats = [
            ("Current", f"₹{current:,.2f}", '#e2e8f0'),
            ("52W High", f"₹{high_52w:,.2f}", '#00ff88'),
            ("52W Low", f"₹{low_52w:,.2f}", '#ff4444'),
            ("From High", f"{from_high:.1f}%", '#ff4444' if from_high < -10 else '#ffaa00'),
            ("From Low", f"+{from_low:.1f}%", '#00ff88' if from_low > 20 else '#ffaa00'),
            ("Avg Volume", f"{info.get('averageVolume', 0)/1e5:.1f}L", '#6b7394'),
        ]
        for i, (label, value, color) in enumerate(price_stats):
            with p_cols[i]:
                st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                    <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">{label}</div>
                    <div style="font-family:'JetBrains Mono';font-size:0.8rem;font-weight:700;color:{color};">{value}</div>
                </div>""", unsafe_allow_html=True)

    # === SECTION B: TECHNICAL DASHBOARD ===
    with st.expander("📊 TECHNICAL ANALYSIS", expanded=True):
        # Compute indicators
        rsi_series = pd.Series(close).diff()
        gain = rsi_series.where(rsi_series > 0, 0).ewm(alpha=1/14, adjust=False).mean().values
        loss_arr = (-rsi_series.where(rsi_series < 0, 0)).ewm(alpha=1/14, adjust=False).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(loss_arr != 0, gain / loss_arr, 0)
        rsi = 100 - (100 / (1 + rs))

        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

        sma20 = pd.Series(close).rolling(20).mean().values
        sma50 = pd.Series(close).rolling(50).mean().values
        sma100 = pd.Series(close).rolling(100).mean().values
        sma200 = pd.Series(close).rolling(200).mean().values
        ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
        ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values

        # ATR
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(14).mean().values

        # Bollinger Bands
        bb_sma = pd.Series(close).rolling(20).mean().values
        bb_std = pd.Series(close).rolling(20).std().values
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std

        # Technical Verdict — count buy vs sell signals
        buy_signals = 0
        sell_signals = 0
        total_signals = 12

        # RSI
        if rsi[-1] < 30: buy_signals += 1
        elif rsi[-1] > 70: sell_signals += 1
        elif 40 <= rsi[-1] <= 60: buy_signals += 1

        # MACD
        if macd_line[-1] > signal_line[-1]: buy_signals += 1
        else: sell_signals += 1

        # Moving averages (price vs)
        for ma_val in [sma20[-1], sma50[-1], sma100[-1], sma200[-1], ema20[-1], ema50[-1]]:
            if not np.isnan(ma_val):
                if close[-1] > ma_val: buy_signals += 1
                else: sell_signals += 1

        # Golden/Death cross
        if not np.isnan(sma50[-1]) and not np.isnan(sma200[-1]):
            if sma50[-1] > sma200[-1]: buy_signals += 1
            else: sell_signals += 1

        # Bollinger
        if not np.isnan(bb_lower[-1]) and close[-1] < bb_lower[-1]: buy_signals += 1
        elif not np.isnan(bb_upper[-1]) and close[-1] > bb_upper[-1]: sell_signals += 1

        neutral = total_signals - buy_signals - sell_signals

        # Display verdict
        if buy_signals >= 8:
            verdict = "STRONG BUY"
            v_color = "#00ff88"
        elif buy_signals >= 6:
            verdict = "BUY"
            v_color = "#4a9eff"
        elif sell_signals >= 8:
            verdict = "STRONG SELL"
            v_color = "#ff4444"
        elif sell_signals >= 6:
            verdict = "SELL"
            v_color = "#ff6b6b"
        else:
            verdict = "NEUTRAL"
            v_color = "#ffaa00"

        st.markdown(f"""
        <div style="text-align:center;padding:8px;border:2px solid {v_color};border-radius:6px;margin-bottom:8px;">
            <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#6b7394;">TECHNICAL VERDICT</div>
            <div style="font-family:'JetBrains Mono';font-size:1.4rem;font-weight:700;color:{v_color};">{verdict}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#6b7394;">
                {buy_signals} BUY │ {neutral} NEUTRAL │ {sell_signals} SELL (out of {total_signals} indicators)
            </div>
        </div>""", unsafe_allow_html=True)

        # Indicator table
        t_cols = st.columns(4)
        with t_cols[0]:
            rsi_color = '#ff4444' if rsi[-1] > 70 else ('#00ff88' if rsi[-1] < 30 else '#ffaa00')
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">RSI (14)</div>
                <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{rsi_color};">{rsi[-1]:.1f}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:{rsi_color};">
                    {'Overbought' if rsi[-1] > 70 else ('Oversold' if rsi[-1] < 30 else 'Neutral')}
                </div>
            </div>""", unsafe_allow_html=True)

        with t_cols[1]:
            macd_status = "Bullish" if macd_line[-1] > signal_line[-1] else "Bearish"
            macd_color = '#00ff88' if macd_line[-1] > signal_line[-1] else '#ff4444'
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">MACD</div>
                <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{macd_color};">{macd_status}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#6b7394;">
                    Line: {macd_line[-1]:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

        with t_cols[2]:
            atr_pct = atr[-1] / close[-1] * 100 if not np.isnan(atr[-1]) else 0
            vol_label = "High" if atr_pct > 2.5 else ("Normal" if atr_pct > 1.5 else "Low")
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">ATR (14)</div>
                <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:#e2e8f0;">₹{atr[-1]:.2f}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#6b7394;">
                    {atr_pct:.1f}% │ {vol_label} Vol
                </div>
            </div>""", unsafe_allow_html=True)

        with t_cols[3]:
            bb_pos = "Upper" if close[-1] > bb_upper[-1] else ("Lower" if close[-1] < bb_lower[-1] else "Middle")
            bb_color = '#ff4444' if bb_pos == "Upper" else ('#00ff88' if bb_pos == "Lower" else '#ffaa00')
            st.markdown(f"""<div style="text-align:center;padding:4px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.55rem;color:#6b7394;">BOLLINGER</div>
                <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{bb_color};">{bb_pos}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#6b7394;">
                    Band: ₹{bb_lower[-1]:.0f} - ₹{bb_upper[-1]:.0f}
                </div>
            </div>""", unsafe_allow_html=True)

        # Moving Averages Table
        st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;margin-top:8px;">
            MOVING AVERAGES</div>""", unsafe_allow_html=True)

        ma_data = []
        for name, val in [("SMA 20", sma20[-1]), ("SMA 50", sma50[-1]),
                          ("SMA 100", sma100[-1]), ("SMA 200", sma200[-1]),
                          ("EMA 20", ema20[-1]), ("EMA 50", ema50[-1])]:
            if not np.isnan(val):
                status = "BUY ▲" if close[-1] > val else "SELL ▼"
                dist = (close[-1] / val - 1) * 100
                ma_data.append({'MA': name, 'Value': f"₹{val:,.2f}",
                               'Signal': status, 'Distance': f"{dist:+.1f}%"})

        if ma_data:
            ma_df = pd.DataFrame(ma_data)
            st.dataframe(ma_df, use_container_width=True, height=180, hide_index=True)

    # === SECTION C: NEWS & SENTIMENT ===
    with st.expander("📰 NEWS & SENTIMENT", expanded=False):
        try:
            from src.sentiment.news_analyzer import NewsSentimentAnalyzer
            analyzer = NewsSentimentAnalyzer()
            result = analyzer.get_stock_sentiment(symbol)

            score = result.get('sentiment_score', 0)
            num_articles = result.get('num_articles', 0)
            articles = result.get('articles', [])

            # Sentiment gauge
            s_color = '#00ff88' if score > 3 else ('#ff4444' if score < -3 else '#ffaa00')
            label = 'BULLISH' if score > 3 else ('BEARISH' if score < -3 else 'NEUTRAL')

            st.markdown(f"""
            <div style="text-align:center;padding:8px;border:1px solid #1a2332;border-radius:4px;margin-bottom:8px;">
                <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">
                    SENTIMENT SCORE ({num_articles} articles)
                </div>
                <div style="font-family:'JetBrains Mono';font-size:1.5rem;font-weight:700;color:{s_color};">
                    {score:+.2f}/10 — {label}
                </div>
            </div>""", unsafe_allow_html=True)

            # Articles
            for art in articles[:10]:
                title = art.get('title', 'No title')[:100]
                source = art.get('source', 'Unknown')
                compound = art.get('sentiment', {}).get('compound', 0)
                tag = '🟢 BULL' if compound > 0.2 else ('🔴 BEAR' if compound < -0.2 else '⚪ NEUTRAL')
                date_str = art.get('date', datetime.now())
                if isinstance(date_str, datetime):
                    date_str = date_str.strftime('%b %d')

                st.markdown(f"""
                <div style="padding:4px 8px;border-bottom:1px solid #1a2332;font-family:'JetBrains Mono';font-size:0.65rem;">
                    <span style="color:#6b7394;">{date_str}</span>
                    <span style="margin:0 6px;">{tag}</span>
                    <span style="color:#e2e8f0;">{title}</span>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Sentiment unavailable: {e}")

    # === SECTION D: SMART MONEY ===
    with st.expander("💰 SMART MONEY ACTIVITY", expanded=False):
        try:
            from src.smart_money.tracker import SmartMoneyTracker
            tracker = SmartMoneyTracker()

            fii_signal = tracker.get_fii_dii_signal()
            insider_signal = tracker.get_insider_signal(symbol)

            sm_cols = st.columns(2)
            with sm_cols[0]:
                fii_color = '#00ff88' if fii_signal['signal'] == 'bullish' else (
                    '#ff4444' if fii_signal['signal'] == 'bearish' else '#ffaa00')
                st.markdown(f"""<div style="text-align:center;padding:8px;border:1px solid #1a2332;border-radius:4px;">
                    <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">FII/DII FLOW</div>
                    <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{fii_color};">
                        {fii_signal['signal'].upper()}
                    </div>
                    <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#4a5568;">
                        {fii_signal.get('details', '')}
                    </div>
                </div>""", unsafe_allow_html=True)

            with sm_cols[1]:
                ins_color = '#00ff88' if insider_signal.get('signal') == 'bullish' else (
                    '#ff4444' if insider_signal.get('signal') == 'bearish' else '#ffaa00')
                st.markdown(f"""<div style="text-align:center;padding:8px;border:1px solid #1a2332;border-radius:4px;">
                    <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#6b7394;">INSIDER ACTIVITY</div>
                    <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{ins_color};">
                        {insider_signal.get('signal', 'neutral').upper()}
                    </div>
                    <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#4a5568;">
                        Score: {insider_signal.get('score', 0)}/5
                    </div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Smart money data unavailable: {e}")
