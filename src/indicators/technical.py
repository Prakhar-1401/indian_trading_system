"""
Technical Indicators Module — Calculates momentum signals.

WHAT THIS DOES:
Takes raw OHLCV price data and computes indicators like RSI, MACD,
moving averages, etc. Then converts them into a SCORE (0-10) that
feeds into our multi-factor ranking.

WHY THESE INDICATORS?
- RSI: Shows if a stock is overbought/oversold. Between 40-65 = healthy uptrend
- MACD: Shows trend direction and momentum shifts
- Moving Averages: 50 DMA above 200 DMA = "Golden Cross" (bullish)
- Returns: Recent price performance (3-month, 6-month)
- Volume: Confirms price moves (up on high volume = strong signal)
"""
import pandas as pd
import numpy as np
from loguru import logger


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    - RSI > 70: Overbought (stock might be overextended)
    - RSI < 30: Oversold (potential bounce)
    - RSI 40-65: Healthy uptrend (SWEET SPOT for momentum)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use exponential smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    - MACD above Signal Line: Bullish momentum
    - MACD below Signal Line: Bearish momentum
    - Histogram growing: Momentum accelerating
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def calculate_moving_averages(prices: pd.Series) -> pd.DataFrame:
    """
    Simple & Exponential Moving Averages.
    KEY SIGNALS:
    - Price > 200 DMA: Long-term uptrend
    - Price > 50 DMA: Medium-term uptrend
    - 50 DMA > 200 DMA: "Golden Cross" (very bullish)
    - 50 DMA < 200 DMA: "Death Cross" (bearish)
    """
    return pd.DataFrame({
        "sma_20": prices.rolling(20).mean(),
        "sma_50": prices.rolling(50).mean(),
        "sma_200": prices.rolling(200).mean(),
        "ema_21": prices.ewm(span=21, adjust=False).mean(),
    })


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands — shows volatility.
    - Price near upper band: Potentially overbought
    - Price near lower band: Potentially oversold
    - Bands squeezing: Big move coming (breakout imminent)
    """
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return pd.DataFrame({
        "bb_upper": sma + (std_dev * std),
        "bb_middle": sma,
        "bb_lower": sma - (std_dev * std),
        "bb_width": ((sma + std_dev * std) - (sma - std_dev * std)) / sma,
    })


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility in absolute terms."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume analysis — confirms price moves.
    - Price up + Volume up: Strong bullish signal
    - Price up + Volume down: Weak rally, might reverse
    - Price down + Volume up: Strong selling pressure
    """
    avg_vol_20 = df["volume"].rolling(20).mean()
    vol_ratio = df["volume"] / avg_vol_20

    price_change = df["close"].pct_change()
    vol_price_confirm = np.where(
        (price_change > 0) & (vol_ratio > 1.5), 1,   # Strong bullish
        np.where(
            (price_change < 0) & (vol_ratio > 1.5), -1,  # Strong bearish
            0  # Inconclusive
        )
    )

    return pd.DataFrame({
        "vol_ratio": vol_ratio,
        "vol_sma_20": avg_vol_20,
        "vol_price_confirm": vol_price_confirm,
    }, index=df.index)


def calculate_returns(prices: pd.Series) -> dict:
    """Calculate returns over various periods."""
    current = prices.iloc[-1]
    returns = {}
    for label, days in [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]:
        if len(prices) > days:
            past_price = prices.iloc[-days - 1]
            returns[f"return_{label}"] = ((current - past_price) / past_price) * 100
        else:
            returns[f"return_{label}"] = None
    return returns


def compute_momentum_score(df: pd.DataFrame, config: dict = None) -> float:
    """
    CORE FUNCTION: Converts all momentum indicators into a single score (0-10).

    HOW SCORING WORKS:
    We check multiple conditions and assign points. More points = stronger momentum.
    This is then normalized to 0-10 scale.
    
    Maximum raw points: 12
    """
    if df.empty or len(df) < 200:
        return 0.0

    close = df["close"]
    score = 0
    max_score = 12

    # --- RSI Score (0-3 points) ---
    rsi = calculate_rsi(close)
    latest_rsi = rsi.iloc[-1]
    if 40 <= latest_rsi <= 65:
        score += 3  # Sweet spot: healthy uptrend
    elif 30 <= latest_rsi < 40 or 65 < latest_rsi <= 70:
        score += 1  # Acceptable range
    # Overbought or oversold = 0 points

    # --- MACD Score (0-2 points) ---
    macd_df = calculate_macd(close)
    if macd_df["macd"].iloc[-1] > macd_df["signal"].iloc[-1]:
        score += 1  # Bullish crossover
    if macd_df["histogram"].iloc[-1] > macd_df["histogram"].iloc[-2]:
        score += 1  # Momentum accelerating

    # --- Moving Average Score (0-3 points) ---
    ma = calculate_moving_averages(close)
    latest_price = close.iloc[-1]

    if latest_price > ma["sma_200"].iloc[-1]:
        score += 1  # Above 200 DMA (long-term uptrend)
    if latest_price > ma["sma_50"].iloc[-1]:
        score += 1  # Above 50 DMA (medium-term uptrend)
    if ma["sma_50"].iloc[-1] > ma["sma_200"].iloc[-1]:
        score += 1  # Golden cross

    # --- Return Score (0-3 points) ---
    returns = calculate_returns(close)
    ret_3m = returns.get("return_3m", 0) or 0
    ret_6m = returns.get("return_6m", 0) or 0

    if ret_3m > 15:
        score += 2
    elif ret_3m > 5:
        score += 1

    if ret_6m > 20:
        score += 1

    # --- Volume Confirmation (0-1 point) ---
    vol = calculate_volume_signals(df)
    recent_vol_confirm = vol["vol_price_confirm"].iloc[-5:].sum()
    if recent_vol_confirm >= 2:
        score += 1  # Recent volume confirms upward price moves

    # Normalize to 0-10
    normalized = round((score / max_score) * 10, 2)
    return min(normalized, 10.0)


def get_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and append as columns to the DataFrame."""
    result = df.copy()

    # RSI
    result["rsi"] = calculate_rsi(result["close"])

    # MACD
    macd = calculate_macd(result["close"])
    result = pd.concat([result, macd], axis=1)

    # Moving Averages
    ma = calculate_moving_averages(result["close"])
    result = pd.concat([result, ma], axis=1)

    # Bollinger Bands
    bb = calculate_bollinger_bands(result["close"])
    result = pd.concat([result, bb], axis=1)

    # ATR
    result["atr"] = calculate_atr(result)

    # Volume signals
    vol = calculate_volume_signals(result)
    result = pd.concat([result, vol], axis=1)

    return result
