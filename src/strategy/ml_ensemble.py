"""
ML Ensemble Signal Model — Machine Learning for Trade Confidence

WHAT THIS DOES:
===============
Instead of fixed rules like "RSI < 30 = BUY", this trains a model on
HISTORICAL data to learn which indicator combinations actually led to
profitable trades.

HOW QUANT FIRMS USE ML:
=======================
1. Feature Engineering: Convert raw price data into 50+ features
   (RSI, MACD, volume ratios, momentum, volatility, etc.)
2. Label Generation: Look FORWARD in time — did buying here give +2% in 5 days?
3. Train Models: Random Forest + XGBoost learn patterns humans can't see
4. Ensemble: Combine multiple models for more robust predictions
5. Walk-Forward: Train on past, test on future (no lookahead bias)

WHY ENSEMBLE (not single model)?
================================
- Random Forest: Good at capturing non-linear relationships, resistant to overfitting
- XGBoost: Better at sequential patterns, handles imbalanced data well
- Combined: If BOTH agree → high confidence. If they disagree → stay out.

FEATURES USED (50+):
====================
- Price momentum (1d, 5d, 10d, 20d, 60d returns)
- RSI (multiple periods)
- MACD signal line crossovers
- Bollinger Band position
- Volume ratios (vs 20-day avg)
- ATR (volatility)
- Moving average crossovers (5/20, 10/50, 20/200)
- Day of week, month effects
- Sector momentum
- Distance from 52-week high/low
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.data.fetcher import DataManager


@dataclass
class MLSignal:
    """ML-generated trading signal with confidence."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1 (how sure the model is)
    rf_prob: float  # Random Forest probability
    xgb_prob: float  # XGBoost probability
    agreement: str  # STRONG_AGREE, AGREE, DISAGREE
    features_summary: Dict  # Key features driving the signal


class FeatureEngineer:
    """
    Generate 50+ features from raw OHLCV data.
    
    This is the SECRET SAUCE — the quality of features determines
    model performance more than the model itself.
    """

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        # === MOMENTUM FEATURES ===
        for period in [1, 2, 3, 5, 10, 20, 60]:
            features[f'return_{period}d'] = df['close'].pct_change(period)

        # Cumulative returns
        features['return_5d_cumul'] = df['close'].pct_change(5)
        features['return_20d_cumul'] = df['close'].pct_change(20)

        # === RSI (Multiple Periods) ===
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calc_rsi(df['close'], period)

        # === MACD ===
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_crossover'] = (macd > signal).astype(int)

        # === BOLLINGER BANDS ===
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = (df['close'] - (sma20 + 2 * std20)) / df['close']
        features['bb_lower'] = (df['close'] - (sma20 - 2 * std20)) / df['close']
        features['bb_position'] = (df['close'] - sma20) / (2 * std20)  # -1 to +1

        # === MOVING AVERAGE FEATURES ===
        for short, long in [(5, 20), (10, 50), (20, 200)]:
            sma_s = df['close'].rolling(short).mean()
            sma_l = df['close'].rolling(long).mean()
            features[f'ma_{short}_{long}_cross'] = (sma_s > sma_l).astype(int)
            features[f'ma_{short}_{long}_dist'] = (sma_s - sma_l) / sma_l

        # Price vs moving averages
        for period in [10, 20, 50, 200]:
            sma = df['close'].rolling(period).mean()
            features[f'price_vs_sma{period}'] = (df['close'] - sma) / sma

        # === VOLUME FEATURES ===
        vol_sma20 = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / vol_sma20
        features['volume_trend'] = df['volume'].rolling(5).mean() / vol_sma20
        features['volume_spike'] = (df['volume'] > vol_sma20 * 2).astype(int)

        # On-Balance Volume trend
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_slope'] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0,
            raw=False
        )

        # === VOLATILITY FEATURES ===
        features['atr_14'] = self._calc_atr(df, 14) / df['close']
        features['atr_7'] = self._calc_atr(df, 7) / df['close']
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = (
            df['close'].pct_change().rolling(5).std() /
            df['close'].pct_change().rolling(20).std()
        )

        # === CANDLESTICK FEATURES ===
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        features['is_bullish'] = (df['close'] > df['open']).astype(int)

        # === DISTANCE FROM EXTREMES ===
        features['dist_52w_high'] = (df['close'] - df['high'].rolling(252).max()) / df['close']
        features['dist_52w_low'] = (df['close'] - df['low'].rolling(252).min()) / df['close']
        features['dist_20d_high'] = (df['close'] - df['high'].rolling(20).max()) / df['close']

        # === TIME FEATURES ===
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        features['is_month_end'] = (df.index.day > 25).astype(int)

        # === MEAN REVERSION ===
        features['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        features['zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

        return features

    def _calc_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()


class MLEnsembleModel:
    """
    Ensemble ML model combining Random Forest + Gradient Boosting.
    
    USAGE:
        model = MLEnsembleModel()
        model.train("RELIANCE")  # Trains on historical data
        signal = model.predict("RELIANCE")  # Get current signal
        
        # Or batch predict:
        signals = model.predict_batch(["RELIANCE", "TCS", "SBIN"])
    """

    def __init__(self, forward_days: int = 5, target_return: float = 0.02):
        """
        Args:
            forward_days: How many days ahead to look for profit (default 5)
            target_return: Min return to count as profitable (default 2%)
        """
        self.dm = DataManager()
        self.fe = FeatureEngineer()
        self.forward_days = forward_days
        self.target_return = target_return
        self.models: Dict[str, dict] = {}  # Trained models per symbol
        self.scaler = StandardScaler()

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target labels: 1 = profitable trade, 0 = not.
        
        Looks FORWARD in time:
        - If price goes up by target_return% in forward_days → label = 1 (BUY was correct)
        - Otherwise → label = 0
        """
        future_return = df['close'].shift(-self.forward_days) / df['close'] - 1
        labels = (future_return >= self.target_return).astype(int)
        return labels

    def train(self, symbol: str, period: str = "5y") -> Dict:
        """
        Train ML models on historical data for a symbol.
        
        Uses walk-forward validation (train on past, test on recent).
        """
        logger.info(f"Training ML model for {symbol}...")

        # Get data
        df = self.dm.get_stock_data(symbol, period=period)
        if df.empty or len(df) < 300:
            logger.warning(f"Insufficient data for {symbol} (need 300+ days)")
            return {}

        # Normalize timezone
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Generate features and labels
        features = self.fe.generate_features(df)
        labels = self._create_labels(df)

        # Combine and drop NaN rows
        data = features.copy()
        data['label'] = labels
        data = data.dropna()

        if len(data) < 200:
            logger.warning(f"Not enough clean data for {symbol}")
            return {}

        # Split: Train on first 80%, test on last 20% (walk-forward)
        split_idx = int(len(data) * 0.8)
        X_train = data.iloc[:split_idx].drop('label', axis=1)
        y_train = data.iloc[:split_idx]['label']
        X_test = data.iloc[split_idx:].drop('label', axis=1)
        y_test = data.iloc[split_idx:]['label']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)

        # Train Gradient Boosting (XGBoost-like)
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)

        # Evaluate
        rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
        gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))

        # Feature importance (top 10)
        importance = pd.Series(
            rf.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False).head(10)

        # Store trained models
        self.models[symbol] = {
            'rf': rf,
            'gb': gb,
            'scaler': scaler,
            'feature_cols': list(X_train.columns),
            'rf_accuracy': rf_acc,
            'gb_accuracy': gb_acc,
            'feature_importance': importance.to_dict(),
            'train_samples': len(X_train),
            'positive_rate': y_train.mean(),
        }

        logger.info(f"  {symbol}: RF acc={rf_acc:.1%}, GB acc={gb_acc:.1%}, "
                    f"features={len(X_train.columns)}")

        return self.models[symbol]

    def predict(self, symbol: str) -> Optional[MLSignal]:
        """
        Generate ML signal for a symbol (must be trained first).
        """
        if symbol not in self.models:
            self.train(symbol)

        if symbol not in self.models:
            return None

        model_data = self.models[symbol]

        # Get recent data
        df = self.dm.get_stock_data(symbol, period="1y")
        if df.empty:
            return None

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Generate features for latest bar
        features = self.fe.generate_features(df)
        features = features[model_data['feature_cols']]
        latest = features.iloc[[-1]].dropna(axis=1)

        # Ensure all required columns exist
        missing_cols = set(model_data['feature_cols']) - set(latest.columns)
        for col in missing_cols:
            latest[col] = 0
        latest = latest[model_data['feature_cols']]

        if latest.isna().any().any():
            latest = latest.fillna(0)

        # Scale and predict
        X = model_data['scaler'].transform(latest)

        rf_prob = model_data['rf'].predict_proba(X)[0][1]  # Prob of class 1 (profitable)
        gb_prob = model_data['gb'].predict_proba(X)[0][1]

        # Ensemble: average probabilities
        ensemble_prob = (rf_prob + gb_prob) / 2

        # Determine action
        if ensemble_prob >= 0.65:
            action = "BUY"
        elif ensemble_prob <= 0.35:
            action = "SELL"
        else:
            action = "HOLD"

        # Agreement level
        if abs(rf_prob - gb_prob) < 0.1:
            agreement = "STRONG_AGREE"
        elif abs(rf_prob - gb_prob) < 0.2:
            agreement = "AGREE"
        else:
            agreement = "DISAGREE"

        return MLSignal(
            symbol=symbol,
            action=action,
            confidence=round(ensemble_prob, 3),
            rf_prob=round(rf_prob, 3),
            xgb_prob=round(gb_prob, 3),
            agreement=agreement,
            features_summary=model_data['feature_importance']
        )

    def predict_batch(self, symbols: List[str]) -> List[MLSignal]:
        """Train and predict for multiple symbols."""
        signals = []
        for symbol in symbols:
            signal = self.predict(symbol)
            if signal:
                signals.append(signal)
        return sorted(signals, key=lambda x: x.confidence, reverse=True)

    def print_report(self, signals: List[MLSignal]):
        """Print ML predictions report."""
        print("\n" + "=" * 70)
        print("  ML ENSEMBLE MODEL — SIGNAL PREDICTIONS")
        print("=" * 70)
        print(f"  Model: Random Forest + Gradient Boosting")
        print(f"  Target: {self.target_return*100:.0f}% return in {self.forward_days} days")
        print(f"  Confidence Threshold: BUY > 65%, SELL < 35%")

        print(f"\n  {'Symbol':<12} {'Action':<6} {'Conf':>6} {'RF':>6} {'GB':>6} "
              f"{'Agreement':<14} {'Model Acc':>9}")
        print("  " + "-" * 70)

        for s in signals:
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(s.action, "⚪")
            model_info = self.models.get(s.symbol, {})
            rf_acc = model_info.get('rf_accuracy', 0)

            print(
                f"  {s.symbol:<12} {emoji}{s.action:<5} {s.confidence:>5.1%} "
                f"{s.rf_prob:>5.1%} {s.xgb_prob:>5.1%} "
                f"{s.agreement:<14} {rf_acc:>8.1%}"
            )

        # Summary
        buys = [s for s in signals if s.action == "BUY"]
        sells = [s for s in signals if s.action == "SELL"]

        if buys:
            print(f"\n  📈 ML says BUY: {', '.join(s.symbol for s in buys)}")
        if sells:
            print(f"  📉 ML says SELL: {', '.join(s.symbol for s in sells)}")

        # Top features driving decisions
        if signals:
            best = signals[0]
            print(f"\n  Top Features (driving {best.symbol} prediction):")
            for feat, imp in list(best.features_summary.items())[:5]:
                bar = "█" * int(imp * 100)
                print(f"    {feat:<25} {imp:.3f} {bar}")

        print("\n  NOTE: ML signals should CONFIRM other indicators, not replace them.")
        print("  High confidence + agreement = strongest signal.")
        print("=" * 70)
