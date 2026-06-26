"""
Performance Module — Trade journal, P&L analytics, equity vs benchmark.

Features:
- Trade journal with notes/tags
- P&L by sector, by strategy type
- Equity curve vs NIFTY 50 benchmark
- Win/loss streaks, drawdown analysis
- Monthly/weekly performance heatmap
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")
TRADES_FILE = DATA_DIR / "paper_trades_history.json"
JOURNAL_FILE = DATA_DIR / "paper_journal.json"
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"


def _load_json(filepath: Path):
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def render_performance():
    """Render performance analytics dashboard."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4a9eff;">
            📈 PERFORMANCE
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7394;">
            Trade journal │ P&L analytics │ Equity curve
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Load trades
    trades_data = _load_json(TRADES_FILE)
    trades = trades_data.get('trades', [])
    portfolio = _load_json(PORTFOLIO_FILE)
    starting_capital = portfolio.get('capital', 1000000)

    if not trades:
        st.info("No completed trades yet. Use Paper Trading to execute trades, then track performance here.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])

    # === TOP METRICS ===
    total_pnl = trades_df['pnl'].sum()
    total_return = total_pnl / starting_capital * 100
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(winning) / len(trades_df) * 100
    avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
    avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0
    profit_factor = winning['pnl'].sum() / abs(losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else 0
    avg_hold = trades_df['hold_days'].mean() if 'hold_days' in trades_df.columns else 0

    # Sharpe (annualized from trade returns)
    if len(trades_df) > 1 and 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'].values / 100
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / max(avg_hold, 1)) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    equity = trades_df['pnl'].cumsum() + starting_capital
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()

    m_cols = st.columns(8)
    metrics = [
        ("Total P&L", f"₹{total_pnl:+,.0f}", '#00ff88' if total_pnl >= 0 else '#ff4444'),
        ("Return", f"{total_return:+.1f}%", '#00ff88' if total_return >= 0 else '#ff4444'),
        ("Win Rate", f"{win_rate:.0f}%", '#00ff88' if win_rate >= 50 else '#ffaa00'),
        ("Profit Factor", f"{profit_factor:.2f}", '#00ff88' if profit_factor >= 1.5 else '#ffaa00'),
        ("Sharpe", f"{sharpe:.2f}", '#00ff88' if sharpe >= 1 else '#ffaa00'),
        ("Max DD", f"{max_dd:.1f}%", '#ff4444' if max_dd < -10 else '#ffaa00'),
        ("Avg Hold", f"{avg_hold:.0f}d", '#6b7394'),
        ("Trades", f"{len(trades_df)}", '#e2e8f0'),
    ]
    for i, (label, value, color) in enumerate(metrics):
        with m_cols[i]:
            st.markdown(f"""<div style="text-align:center;padding:5px;border:1px solid #1a2332;border-radius:4px;">
                <div style="font-family:'JetBrains Mono';font-size:0.5rem;color:#6b7394;">{label}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.8rem;font-weight:700;color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="margin:8px 0;border-color:#1a2332;">', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "📊 Analytics", "📋 Journal", "🏷️ By Sector"])

    # === EQUITY CURVE ===
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.05)

        # Equity
        cum_pnl = trades_df['pnl'].cumsum() + starting_capital
        fig.add_trace(go.Scatter(
            x=trades_df['exit_date'], y=cum_pnl,
            mode='lines+markers', name='Portfolio',
            line=dict(color='#4a9eff', width=2),
            marker=dict(size=4),
            fill='tozeroy', fillcolor='rgba(74,158,255,0.05)'
        ), row=1, col=1)

        # Starting capital reference
        fig.add_hline(y=starting_capital, line_dash="dash", line_color="#6b7394",
                      annotation_text="Starting Capital", row=1, col=1)

        # Drawdown
        fig.add_trace(go.Scatter(
            x=trades_df['exit_date'], y=drawdown,
            mode='lines', name='Drawdown %',
            line=dict(color='#ff4444', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,68,68,0.1)'
        ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220', height=400,
            margin=dict(l=50, r=10, t=20, b=20),
            font=dict(family='JetBrains Mono', size=10, color='#6b7394'),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor='#1a2332')
        fig.update_yaxes(gridcolor='#1a2332')
        fig.update_yaxes(title_text="Equity ₹", row=1, col=1, tickformat=',')
        fig.update_yaxes(title_text="DD %", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # === ANALYTICS ===
    with tab2:
        a_cols = st.columns(2)

        with a_cols[0]:
            # Win/Loss distribution
            st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;">
                P&L DISTRIBUTION</div>""", unsafe_allow_html=True)

            fig_dist = go.Figure()
            colors = ['#00ff88' if p > 0 else '#ff4444' for p in trades_df['pnl']]
            fig_dist.add_trace(go.Bar(
                x=list(range(len(trades_df))), y=trades_df['pnl'],
                marker_color=colors
            ))
            fig_dist.update_layout(
                paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220', height=200,
                margin=dict(l=40, r=10, t=10, b=20),
                font=dict(family='JetBrains Mono', size=9, color='#6b7394'),
                xaxis=dict(gridcolor='#1a2332', title='Trade #'),
                yaxis=dict(gridcolor='#1a2332', title='P&L ₹', tickformat=','),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with a_cols[1]:
            # Win vs Loss stats
            st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;">
                WIN / LOSS BREAKDOWN</div>""", unsafe_allow_html=True)

            breakdown = pd.DataFrame({
                'Metric': ['Count', 'Total P&L', 'Average', 'Best/Worst', 'Avg Hold Days'],
                'Winners ✅': [
                    f"{len(winning)}",
                    f"₹{winning['pnl'].sum():+,.0f}" if len(winning) > 0 else "—",
                    f"₹{avg_win:+,.0f}",
                    f"₹{winning['pnl'].max():+,.0f}" if len(winning) > 0 else "—",
                    f"{winning['hold_days'].mean():.0f}" if len(winning) > 0 and 'hold_days' in winning.columns else "—",
                ],
                'Losers ❌': [
                    f"{len(losing)}",
                    f"₹{losing['pnl'].sum():+,.0f}" if len(losing) > 0 else "—",
                    f"₹{avg_loss:+,.0f}",
                    f"₹{losing['pnl'].min():+,.0f}" if len(losing) > 0 else "—",
                    f"{losing['hold_days'].mean():.0f}" if len(losing) > 0 and 'hold_days' in losing.columns else "—",
                ],
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True, height=200)

        # Streaks
        st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;margin-top:8px;">
            STREAKS</div>""", unsafe_allow_html=True)
        wins_losses = (trades_df['pnl'] > 0).astype(int).values
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for wl in wins_losses:
            if wl == 1:
                cur_win += 1
                cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        s_cols = st.columns(4)
        with s_cols[0]:
            st.metric("Max Win Streak", f"{max_win_streak}")
        with s_cols[1]:
            st.metric("Max Loss Streak", f"{max_loss_streak}")
        with s_cols[2]:
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.metric("Avg R:R", f"{rr:.2f}")
        with s_cols[3]:
            expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            st.metric("Expectancy", f"₹{expectancy:+,.0f}")

    # === JOURNAL ===
    with tab3:
        st.markdown("""<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4a9eff;">
            TRADE LOG (most recent first)</div>""", unsafe_allow_html=True)

        display_trades = trades_df.copy()
        display_trades = display_trades.sort_values('exit_date', ascending=False)

        for _, t in display_trades.iterrows():
            pnl_color = '#00ff88' if t['pnl'] > 0 else '#ff4444'
            st.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1a2332;border-left:3px solid {pnl_color};
                        border-radius:4px;padding:6px 10px;margin-bottom:4px;font-family:'JetBrains Mono';">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#e2e8f0;font-size:0.85rem;font-weight:700;">{t['symbol']}</span>
                    <span style="color:{pnl_color};font-size:0.85rem;font-weight:700;">₹{t['pnl']:+,.0f} ({t.get('pnl_pct', 0):+.1f}%)</span>
                </div>
                <div style="color:#6b7394;font-size:0.6rem;margin-top:2px;">
                    Entry: ₹{t['entry_price']:,.2f} → Exit: ₹{t['exit_price']:,.2f} │
                    Qty: {t['qty']} │ Hold: {t.get('hold_days', '?')}d │
                    {pd.to_datetime(t['entry_date']).strftime('%b %d')} → {pd.to_datetime(t['exit_date']).strftime('%b %d, %Y')}
                </div>
            </div>""", unsafe_allow_html=True)

    # === BY SECTOR ===
    with tab4:
        # Map symbols to sectors
        from src.dashboard.stock_picker import SECTOR_MAP as PICKER_SECTORS
        trades_df['sector'] = trades_df['symbol'].map(PICKER_SECTORS).fillna('Other')
        sector_pnl = trades_df.groupby('sector')['pnl'].agg(['sum', 'count', 'mean']).reset_index()
        sector_pnl.columns = ['Sector', 'Total P&L', 'Trades', 'Avg P&L']
        sector_pnl = sector_pnl.sort_values('Total P&L', ascending=False)

        fig_sector = go.Figure()
        colors = ['#00ff88' if p > 0 else '#ff4444' for p in sector_pnl['Total P&L']]
        fig_sector.add_trace(go.Bar(
            x=sector_pnl['Sector'], y=sector_pnl['Total P&L'],
            marker_color=colors, text=sector_pnl['Total P&L'].apply(lambda x: f"₹{x:+,.0f}"),
            textposition='outside'
        ))
        fig_sector.update_layout(
            paper_bgcolor='#0a0e17', plot_bgcolor='#0d1220', height=250,
            margin=dict(l=40, r=10, t=20, b=40),
            font=dict(family='JetBrains Mono', size=10, color='#6b7394'),
            xaxis=dict(gridcolor='#1a2332'),
            yaxis=dict(gridcolor='#1a2332', tickformat=','),
            title="P&L by Sector"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

        st.dataframe(sector_pnl, use_container_width=True, hide_index=True)
