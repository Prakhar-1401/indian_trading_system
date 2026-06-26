"""
Streamlit Dashboard — Visual interface for the trading system.

Run with: streamlit run src/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fetcher import DataManager
from src.indicators.technical import get_all_indicators, compute_momentum_score
from src.ranking.ranker import StockRanker, QualityScorer
from src.sentiment.news_analyzer import NewsSentimentAnalyzer
from src.backtest.engine import BacktestEngine
from src.utils.helpers import load_config


st.set_page_config(
    page_title="Indian Trading System",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indian Stock Market Trading System")
st.markdown("*Multi-Factor Stock Ranking + Sentiment + Smart Money*")

# Sidebar
st.sidebar.header("Configuration")
config = load_config()

# Section selector
section = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Stock Analysis",
    "🏆 Stock Rankings",
    "📰 Sentiment Analysis",
    "🔄 Backtest",
    "🎯 Live Signals",
    "⚡ Market Regime",
    "🔥 Sector Heatmap",
    "📋 Paper Trading",
    "🎲 Monte Carlo",
    "🛡️ Risk Analytics",
])


@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1y"):
    dm = DataManager()
    return dm.get_stock_data(symbol, period=period)


@st.cache_data(ttl=3600)
def get_fundamentals(symbol):
    dm = DataManager()
    return dm.get_fundamentals(symbol)


# ============================
# OVERVIEW
# ============================
if section == "🏠 Overview":
    st.header("Strategy Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    weights = config.get("factor_weights", {})
    col1.metric("Momentum Weight", f"{weights.get('momentum', 0.4)*100:.0f}%")
    col2.metric("Quality Weight", f"{weights.get('quality', 0.25)*100:.0f}%")
    col3.metric("Sentiment Weight", f"{weights.get('sentiment', 0.2)*100:.0f}%")
    col4.metric("Smart Money Weight", f"{weights.get('smart_money', 0.15)*100:.0f}%")

    st.markdown("""
    ### How the Strategy Works
    1. **Universe**: Nifty 500 stocks (filtered by market cap, debt, volume)
    2. **Score**: Each stock gets 4 factor scores (0-10 each)
    3. **Rank**: Weighted composite score determines ranking
    4. **Pick**: Top 15 stocks form the portfolio
    5. **Risk**: 8% stop-loss, max 7% per stock, weekly rebalance
    """)

    # Show Nifty 50 chart
    nifty = get_stock_data("^NSEI", period="1y")
    if not nifty.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=nifty.index, open=nifty["open"], high=nifty["high"],
            low=nifty["low"], close=nifty["close"], name="Nifty 50"
        ))
        fig.update_layout(title="Nifty 50 — Last 1 Year", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================
# STOCK ANALYSIS
# ============================
elif section == "📊 Stock Analysis":
    st.header("Individual Stock Analysis")
    
    symbol = st.text_input("Enter NSE Symbol", value="RELIANCE").upper()
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    if st.button("Analyze") or symbol:
        df = get_stock_data(symbol, period=period)
        
        if df.empty:
            st.error(f"No data found for {symbol}")
        else:
            # Price chart with indicators
            indicators_df = get_all_indicators(df)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=indicators_df.index, open=indicators_df["open"],
                high=indicators_df["high"], low=indicators_df["low"],
                close=indicators_df["close"], name="Price"
            ))
            fig.add_trace(go.Scatter(
                x=indicators_df.index, y=indicators_df["sma_50"],
                name="50 DMA", line=dict(color="orange", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=indicators_df.index, y=indicators_df["sma_200"],
                name="200 DMA", line=dict(color="red", width=1)
            ))
            fig.update_layout(title=f"{symbol} Price Chart", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Scores
            col1, col2 = st.columns(2)
            
            with col1:
                m_score = compute_momentum_score(df)
                st.metric("Momentum Score", f"{m_score:.1f}/10")
                
                fundamentals = get_fundamentals(symbol)
                q_score = QualityScorer.compute(fundamentals)
                st.metric("Quality Score", f"{q_score:.1f}/10")
                
                st.subheader("Fundamentals")
                for key, val in fundamentals.items():
                    if val is not None and key not in ["symbol"]:
                        st.write(f"**{key}**: {val}")

            with col2:
                # RSI subplot
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=indicators_df.index, y=indicators_df["rsi"], name="RSI"
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI", height=250)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # MACD subplot
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=indicators_df.index, y=indicators_df["macd"], name="MACD"
                ))
                fig_macd.add_trace(go.Scatter(
                    x=indicators_df.index, y=indicators_df["signal"], name="Signal"
                ))
                fig_macd.add_trace(go.Bar(
                    x=indicators_df.index, y=indicators_df["histogram"], name="Histogram"
                ))
                fig_macd.update_layout(title="MACD", height=250)
                st.plotly_chart(fig_macd, use_container_width=True)


# ============================
# STOCK RANKINGS
# ============================
elif section == "🏆 Stock Rankings":
    st.header("Multi-Factor Stock Rankings")
    
    st.markdown("**Quick Rank** uses Momentum + Quality only (faster, no web scraping)")
    
    default_stocks = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL,ITC,LT,BAJFINANCE,SUNPHARMA,TITAN,MARUTI,WIPRO,HCLTECH"
    symbols_input = st.text_area("Symbols (comma-separated)", value=default_stocks)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    if st.button("Run Quick Ranking"):
        with st.spinner("Scoring stocks... This may take a few minutes."):
            ranker = StockRanker()
            rankings = ranker.rank_quick(symbols)
            
            if not rankings.empty:
                # Color-code scores
                st.dataframe(
                    rankings.style.background_gradient(
                        subset=["composite_score"], cmap="RdYlGn"
                    ),
                    use_container_width=True,
                )
                
                # Bar chart
                fig = px.bar(
                    rankings.head(15), x="symbol", y="composite_score",
                    color="composite_score", color_continuous_scale="RdYlGn",
                    title="Top Stocks by Composite Score"
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================
# SENTIMENT ANALYSIS
# ============================
elif section == "📰 Sentiment Analysis":
    st.header("News Sentiment Analysis")
    
    symbol = st.text_input("Enter NSE Symbol for Sentiment", value="RELIANCE").upper()
    
    if st.button("Analyze Sentiment"):
        with st.spinner(f"Analyzing news for {symbol}..."):
            analyzer = NewsSentimentAnalyzer()
            result = analyzer.get_stock_sentiment(symbol)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment Score", f"{result['sentiment_score']:.1f}/10")
            col2.metric("Articles Found", result["num_articles"])
            col3.metric("Positive %", f"{result['positive_pct']:.0f}%")
            
            if result["articles"]:
                st.subheader("Recent Articles")
                for article in result["articles"]:
                    sentiment = article.get("sentiment", {})
                    compound = sentiment.get("compound", 0)
                    emoji = "🟢" if compound > 0.2 else ("🔴" if compound < -0.2 else "⚪")
                    st.write(f"{emoji} **{article['title']}** (Score: {compound:.2f})")

    st.markdown("---")
    if st.button("Check Overall Market Mood"):
        with st.spinner("Analyzing market news..."):
            analyzer = NewsSentimentAnalyzer()
            mood = analyzer.get_market_sentiment()
            st.metric("Market Mood", mood["market_mood"].upper())
            st.metric("Score", f"{mood['score']:.1f}/10")


# ============================
# BACKTEST
# ============================
elif section == "🔄 Backtest":
    st.header("Strategy Backtesting")
    
    col1, col2, col3 = st.columns(3)
    start_date = col1.date_input("Start Date", value=pd.Timestamp("2022-01-01"))
    end_date = col2.date_input("End Date", value=pd.Timestamp("2025-12-31"))
    capital = col3.number_input("Capital (₹)", value=1000000, step=100000)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest... This may take several minutes."):
            engine = BacktestEngine()
            engine.start_date = str(start_date)
            engine.end_date = str(end_date)
            engine.initial_capital = capital
            
            results = engine.run()
            
            if results:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{results['total_return_pct']:.1f}%")
                col2.metric("CAGR", f"{results['cagr_pct']:.1f}%")
                col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                col4.metric("Max Drawdown", f"{results['max_drawdown_pct']:.1f}%")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Value", f"₹{results['final_value']:,.0f}")
                col2.metric("Win Rate", f"{results['win_rate_pct']:.0f}%")
                col3.metric("Total Trades", results["total_trades"])
                col4.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                
                # Portfolio value chart
                if "portfolio_history" in results:
                    hist = results["portfolio_history"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist["date"], y=hist["total_value"],
                        name="Portfolio Value", fill="tozeroy"
                    ))
                    fig.update_layout(title="Portfolio Value Over Time", height=400)
                    st.plotly_chart(fig, use_container_width=True)


# ============================
# LIVE SIGNALS
# ============================
elif section == "🎯 Live Signals":
    st.header("Live Trading Signals")
    st.warning("⚠️ These are algorithmic signals, NOT financial advice. Always do your own research.")
    
    default_stocks = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL,ITC,LT,BAJFINANCE,SUNPHARMA,TITAN,MARUTI,WIPRO,HCLTECH,TATAMOTORS,NTPC,POWERGRID,COALINDIA,NESTLEIND"
    symbols_input = st.text_area("Symbols", value=default_stocks)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    if st.button("Generate Signals"):
        with st.spinner("Generating signals..."):
            from src.strategy.executor import generate_quick_signals
            signals = generate_quick_signals(symbols)
            
            for s in signals:
                icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(s.action, "⚪")
                st.write(f"{icon} **{s.action}** {s.symbol} — {s.reason}")


# ============================
# MARKET REGIME
# ============================
elif section == "⚡ Market Regime":
    st.header("⚡ Market Regime Detection")

    with st.spinner("Detecting market regime..."):
        try:
            from src.strategy.regime_detector import RegimeDetector
            detector = RegimeDetector()
            result = detector.detect_regime()

            regime = result.get('regime', 'UNKNOWN')
            confidence = result.get('confidence', 0)
            trend_score = result.get('trend_score', 0)
            volatility_pct = result.get('volatility_percentile', 0)

            regime_colors = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡', 'VOLATILE': '⚡'}
            emoji = regime_colors.get(regime, '⚪')

            col1, col2, col3 = st.columns(3)
            col1.metric("Regime", f"{emoji} {regime}")
            col2.metric("Confidence", f"{confidence}%")
            col3.metric("Trend Score", f"{trend_score:+.3f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=volatility_pct,
                title={'text': "Volatility Percentile"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "red" if volatility_pct > 70 else "orange" if volatility_pct > 40 else "green"},
                       'steps': [{'range': [0, 40], 'color': "darkgreen"},
                                 {'range': [40, 70], 'color': "darkorange"},
                                 {'range': [70, 100], 'color': "darkred"}]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            recommendations = {
                'BULL': "Full exposure. Trend-following strategies. Buy dips.",
                'BEAR': "Reduce exposure 50%. Hedge with puts. Short weak stocks.",
                'SIDEWAYS': "Mean-reversion strategies. Pairs trading. Range-bound plays.",
                'VOLATILE': "Cash is king. Reduce position sizes 70%. Widen stops.",
            }
            st.info(f"💡 **Strategy**: {recommendations.get(regime, 'N/A')}")
        except Exception as e:
            st.error(f"Error: {e}")


# ============================
# SECTOR HEATMAP
# ============================
elif section == "🔥 Sector Heatmap":
    st.header("🔥 Sector Performance & Correlations")

    with st.spinner("Loading sector data..."):
        try:
            from src.backtest.sector_analysis import SectorAnalyzer
            analyzer = SectorAnalyzer()
            performance = analyzer.get_sector_performance()
            corr_matrix = analyzer.get_correlation_matrix()

            if performance is not None:
                perf_df = pd.DataFrame(performance).T
                if '1m_return' in perf_df.columns:
                    fig = px.bar(
                        perf_df.sort_values('1m_return', ascending=True).reset_index(),
                        x='1m_return', y='index', orientation='h',
                        color='1m_return', color_continuous_scale='RdYlGn',
                        title="1-Month Sector Returns (%)"
                    )
                    fig.update_layout(yaxis_title="Sector", xaxis_title="Return (%)")
                    st.plotly_chart(fig, use_container_width=True)

            if corr_matrix is not None:
                fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r',
                               aspect='auto', title="Sector Correlations")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


# ============================
# PAPER TRADING
# ============================
elif section == "📋 Paper Trading":
    st.header("📋 Paper Trading Portfolio")

    portfolio_file = os.path.join("data", "paper_portfolio.json")
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            portfolio = json.load(f)

        initial = portfolio.get('initial_capital', 1000000)
        cash = portfolio.get('cash', initial)
        positions = portfolio.get('positions', {})
        total_invested = sum(p.get('quantity', 0) * p.get('entry_price', 0) for p in positions.values())
        total_value = cash + total_invested
        pnl = total_value - initial

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"₹{total_value:,.0f}", f"{(pnl/initial)*100:+.2f}%")
        col2.metric("Cash", f"₹{cash:,.0f}")
        col3.metric("Invested", f"₹{total_invested:,.0f}")
        col4.metric("Positions", len(positions))

        if positions:
            pos_data = [{'Symbol': sym, 'Qty': p['quantity'],
                        'Entry': f"₹{p['entry_price']:,.2f}",
                        'Stop': f"₹{p.get('stop_loss', 0):,.2f}",
                        'Target': f"₹{p.get('target', 0):,.2f}"}
                       for sym, p in positions.items()]
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

            alloc = {sym: p['quantity'] * p['entry_price'] for sym, p in positions.items()}
            alloc['Cash'] = cash
            fig = px.pie(names=list(alloc.keys()), values=list(alloc.values()),
                        title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)

        trades_file = os.path.join("data", "paper_trades.json")
        if os.path.exists(trades_file):
            with open(trades_file, 'r') as f:
                trades = json.load(f)
            if trades:
                st.subheader("Trade History")
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No paper trading data. Use `python main.py paper buy SYMBOL` to start.")


# ============================
# MONTE CARLO
# ============================
elif section == "🎲 Monte Carlo":
    st.header("🎲 Monte Carlo Simulation")

    n_sims = st.slider("Simulations", 1000, 50000, 10000, step=1000)

    if st.button("Run Simulation"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            try:
                from src.backtest.monte_carlo import MonteCarloSimulator
                sim = MonteCarloSimulator()
                symbols = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
                           "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]
                results = sim.run_simulation(symbols, n_simulations=n_sims)

                if results:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Return", f"{results['mean_return']:+.2f}%")
                    col2.metric("P(Profit)", f"{results['prob_profit']:.1f}%")
                    col3.metric("VaR (5%)", f"{results['var_5']:.2f}%")
                    col4.metric("Max DD (median)", f"{results['median_max_dd']:.2f}%")

                    if 'all_returns' in results:
                        fig = px.histogram(results['all_returns'], nbins=100,
                                          title=f"Distribution of 1-Year Returns ({n_sims:,} sims)")
                        fig.add_vline(x=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ============================
# RISK ANALYTICS
# ============================
elif section == "🛡️ Risk Analytics":
    st.header("🛡️ Risk Analytics")

    symbol = st.selectbox("Stock", [
        "RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
        "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC",
        "INFY", "ICICIBANK", "LT", "WIPRO", "HCLTECH"
    ])

    data = get_stock_data(symbol)
    if not data.empty:
        returns = data['close'].pct_change().dropna()

        vol_annual = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        var_95 = np.percentile(returns, 5) * 100
        max_dd = ((data['close'] / data['close'].cummax()) - 1).min() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annual Volatility", f"{vol_annual:.1f}%")
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col3.metric("VaR (95%)", f"{var_95:.2f}%")
        col4.metric("Max Drawdown", f"{max_dd:.2f}%")

        # Price with Bollinger Bands
        sma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=sma20 + 2*std20, name='Upper BB',
                                line=dict(color='rgba(255,255,255,0.3)')))
        fig.add_trace(go.Scatter(x=data.index, y=sma20 - 2*std20, name='Lower BB',
                                line=dict(color='rgba(255,255,255,0.3)'),
                                fill='tonexty', fillcolor='rgba(100,100,255,0.1)'))
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], name=symbol,
                                line=dict(color='cyan')))
        fig.update_layout(title=f"{symbol} — Bollinger Bands", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        drawdown = (data['close'] / data['close'].cummax()) - 1
        fig_dd = px.area(x=drawdown.index, y=drawdown.values * 100,
                        title=f"{symbol} — Drawdown")
        fig_dd.update_traces(line_color='red', fillcolor='rgba(255,0,0,0.2)')
        st.plotly_chart(fig_dd, use_container_width=True)

        # Rolling volatility
        rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        fig_vol = px.line(x=rolling_vol.index, y=rolling_vol.values,
                         title=f"{symbol} — 21-Day Rolling Volatility (Annualized)")
        st.plotly_chart(fig_vol, use_container_width=True)
