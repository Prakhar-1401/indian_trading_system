"""
Unified Decision Engine
========================
The BRAIN of the trading system. Takes outputs from ALL modules
and produces ONE final verdict: TAKE / WATCH / SKIP

This is the key differentiator from a regular dashboard.
Instead of showing 10 different signals and confusing the trader,
we synthesize everything into a single actionable decision.

SCORING SYSTEM (0-100):
    Signal Score     (0-25):  Composite ranking score
    ML Confidence    (0-20):  ML ensemble prediction confidence
    Kelly Edge       (0-20):  Positive edge from Kelly criterion
    Regime Bonus     (0-15):  Favorable regime = bonus, unfavorable = penalty
    Event Safety     (0-10):  No upcoming events = safe
    Geo Safety       (0-10):  Low geopolitical risk = safe

VERDICT:
    75-100: TAKE  — Strong multi-factor agreement. Execute.
    50-74:  WATCH — Some factors align but not all. Monitor.
    0-49:   SKIP  — Not enough evidence. Do not trade.

CONVICTION (how many modules agree):
    5/5: Maximum conviction
    4/5: High conviction
    3/5: Moderate
    2/5: Low
    1/5: Very low — skip
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class TradeDecision:
    """Final trade decision for a stock."""
    symbol: str
    verdict: str  # TAKE, WATCH, SKIP
    score: float  # 0-100
    conviction: int  # How many modules agree (0-5)
    
    # Component scores
    signal_score: float = 0.0
    ml_score: float = 0.0
    kelly_score: float = 0.0
    regime_score: float = 0.0
    event_score: float = 0.0
    geo_score: float = 0.0
    
    # Trade plan (filled only for TAKE)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    position_size: int = 0
    position_value: float = 0.0
    max_loss: float = 0.0
    risk_reward: float = 0.0
    
    # Reason codes
    reasons_for: List[str] = field(default_factory=list)
    reasons_against: List[str] = field(default_factory=list)


class DecisionEngine:
    """
    Unified decision engine that synthesizes all modules into
    a single TAKE/WATCH/SKIP verdict per stock.
    """
    
    def __init__(self, capital: float = 1000000):
        self.capital = capital
        self.dm = DataManager()
        self._regime_cache = None
        self._geo_cache = None
    
    def analyze_stock(self, symbol: str) -> TradeDecision:
        """
        Full analysis of a single stock through all modules.
        Returns a TradeDecision with verdict and trade plan.
        """
        decision = TradeDecision(symbol=symbol, verdict="SKIP", score=0, conviction=0)
        modules_agreeing = 0
        
        # 1. SIGNAL SCORE (0-25 points)
        signal_pts, signal_reasons = self._score_signal(symbol)
        decision.signal_score = signal_pts
        if signal_pts >= 15:
            modules_agreeing += 1
            decision.reasons_for.extend(signal_reasons)
        elif signal_pts < 10:
            decision.reasons_against.append("Weak ranking signal")
        
        # 2. ML CONFIDENCE (0-20 points)
        ml_pts, ml_reasons = self._score_ml(symbol)
        decision.ml_score = ml_pts
        if ml_pts >= 13:
            modules_agreeing += 1
            decision.reasons_for.extend(ml_reasons)
        elif ml_pts < 7:
            decision.reasons_against.append("ML predicts downside")
        
        # 3. KELLY EDGE (0-20 points)
        kelly_pts, kelly_reasons = self._score_kelly(symbol)
        decision.kelly_score = kelly_pts
        if kelly_pts >= 10:
            modules_agreeing += 1
            decision.reasons_for.extend(kelly_reasons)
        elif kelly_pts < 5:
            decision.reasons_against.append("Negative mathematical edge")
        
        # 4. REGIME (0-15 points)
        regime_pts, regime_reasons = self._score_regime()
        decision.regime_score = regime_pts
        if regime_pts >= 10:
            modules_agreeing += 1
            decision.reasons_for.extend(regime_reasons)
        elif regime_pts < 5:
            decision.reasons_against.extend(regime_reasons)
        
        # 5. EVENT SAFETY (0-10 points)
        event_pts, event_reasons = self._score_events(symbol)
        decision.event_score = event_pts
        if event_pts >= 8:
            modules_agreeing += 1
        elif event_pts < 4:
            decision.reasons_against.extend(event_reasons)
        
        # 6. GEO SAFETY (0-10 points)
        geo_pts, geo_reasons = self._score_geo()
        decision.geo_score = geo_pts
        if geo_pts < 4:
            decision.reasons_against.extend(geo_reasons)
        
        # TOTAL SCORE
        total = signal_pts + ml_pts + kelly_pts + regime_pts + event_pts + geo_pts
        decision.score = min(total, 100)
        decision.conviction = modules_agreeing
        
        # VERDICT
        if total >= 75 and modules_agreeing >= 4:
            decision.verdict = "TAKE"
        elif total >= 50 and modules_agreeing >= 3:
            decision.verdict = "WATCH"
        else:
            decision.verdict = "SKIP"
        
        # TRADE PLAN (only for TAKE/WATCH)
        if decision.verdict in ["TAKE", "WATCH"]:
            self._build_trade_plan(decision)
        
        return decision
    
    def analyze_watchlist(self, symbols: List[str] = None) -> List[TradeDecision]:
        """Analyze full watchlist and return sorted decisions."""
        if symbols is None:
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
                "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
                "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
            ]
        
        decisions = []
        for symbol in symbols:
            try:
                d = self.analyze_stock(symbol)
                decisions.append(d)
            except Exception as e:
                logger.warning(f"Decision engine error for {symbol}: {e}")
        
        # Sort: TAKE first, then WATCH, then SKIP, within each group by score
        order = {"TAKE": 0, "WATCH": 1, "SKIP": 2}
        decisions.sort(key=lambda x: (order.get(x.verdict, 3), -x.score))
        return decisions
    
    # ──────────────────────────────────────────────────────────────
    # SCORING FUNCTIONS
    # ──────────────────────────────────────────────────────────────
    
    def _score_signal(self, symbol: str) -> tuple:
        """Score from ranking/signal module (0-25 points)."""
        try:
            from src.ranking.ranker import StockRanker
            ranker = StockRanker()
            rankings = ranker.rank_quick([symbol])
            if rankings.empty:
                return 5, []
            
            composite = rankings.iloc[0]['composite_score']
            # Map 0-10 composite to 0-25 points
            pts = min((composite / 10) * 25, 25)
            
            reasons = []
            if composite > 6:
                reasons.append(f"Strong signal ({composite:.1f}/10)")
            elif composite > 5:
                reasons.append(f"Moderate signal ({composite:.1f}/10)")
            
            return round(pts, 1), reasons
        except Exception as e:
            logger.debug(f"Signal scoring failed for {symbol}: {e}")
            return 10, []  # Neutral
    
    def _score_ml(self, symbol: str) -> tuple:
        """Score from ML ensemble (0-20 points)."""
        try:
            from src.strategy.ml_ensemble import MLEnsembleModel
            model = MLEnsembleModel()
            result = model.predict(symbol)
            
            if result is None:
                return 10, []
            
            confidence = result.confidence
            prediction = result.action  # BUY/SELL/HOLD
            
            if prediction == 'BUY':
                pts = confidence * 20  # Higher confidence = more points
                reasons = [f"ML BUY ({confidence*100:.0f}% conf)"]
            elif prediction == 'SELL':
                pts = (1 - confidence) * 20
                reasons = [f"ML SELL ({confidence*100:.0f}% conf)"]
            else:
                pts = 10
                reasons = []
            
            return round(min(pts, 20), 1), reasons
        except Exception as e:
            logger.debug(f"ML scoring failed for {symbol}: {e}")
            return 10, []
    
    def _score_kelly(self, symbol: str) -> tuple:
        """Score from Kelly criterion (0-20 points)."""
        try:
            from src.strategy.kelly_sizing import KellyPositionSizer
            sizer = KellyPositionSizer(capital=self.capital)
            result = sizer.calculate(symbol)
            
            if result is None:
                return 10, []
            
            edge = result.edge
            
            if edge > 1.5:
                pts = 20
                reasons = [f"Strong edge (+{edge:.1f}%)"]
            elif edge > 0.5:
                pts = 15
                reasons = [f"Positive edge (+{edge:.1f}%)"]
            elif edge > 0:
                pts = 10
                reasons = [f"Marginal edge (+{edge:.1f}%)"]
            else:
                pts = max(0, 10 + edge * 5)  # Negative edge reduces score
                reasons = []
            
            return round(pts, 1), reasons
        except Exception as e:
            logger.debug(f"Kelly scoring failed for {symbol}: {e}")
            return 10, []
    
    def _score_regime(self) -> tuple:
        """Score from market regime (0-15 points). Cached per session."""
        if self._regime_cache is not None:
            return self._regime_cache
        
        try:
            from src.strategy.regime_detector import MarketRegimeDetector
            detector = MarketRegimeDetector()
            result = detector.detect_regime()
            
            regime = result.regime
            
            if regime == 'BULL':
                pts = 15
                reasons = ["Bull regime — full exposure OK"]
            elif regime == 'SIDEWAYS':
                pts = 10
                reasons = ["Sideways — selective entries"]
            elif regime == 'VOLATILE':
                pts = 4
                reasons = ["Volatile regime — reduce 70%"]
            elif regime == 'BEAR':
                pts = 2
                reasons = ["Bear regime — avoid longs"]
            else:
                pts = 8
                reasons = []
            
            self._regime_cache = (pts, reasons)
            return pts, reasons
        except Exception as e:
            logger.debug(f"Regime scoring failed: {e}")
            self._regime_cache = (8, [])
            return 8, []
    
    def _score_events(self, symbol: str) -> tuple:
        """Score from event calendar (0-10 points). No event = safe."""
        try:
            from src.strategy.event_driven import EventDrivenTrader
            trader = EventDrivenTrader()
            events = trader.scan_events()
            
            # Filter events for this symbol and check for dangerous upcoming ones
            symbol_events = [e for e in events if hasattr(e, 'symbol') and e.symbol == symbol]
            dangerous = [e for e in symbol_events if hasattr(e, 'days_until') and 0 <= e.days_until <= 5]
            
            if not dangerous:
                return 10, []
            else:
                event_types = [e.event_type for e in dangerous[:2]]
                return 3, [f"Event risk: {', '.join(event_types)} in <5 days"]
        except Exception as e:
            logger.debug(f"Event scoring failed for {symbol}: {e}")
            return 8, []  # Assume safe if can't check
    
    def _score_geo(self) -> tuple:
        """Score from geopolitical risk (0-10 points). Cached per session."""
        if self._geo_cache is not None:
            return self._geo_cache
        
        try:
            from src.sentiment.geopolitical import GeopoliticalMonitor
            monitor = GeopoliticalMonitor()
            report = monitor.get_risk_report()
            
            risk_level = report.get('risk_level', 'MEDIUM')
            
            if risk_level == 'LOW':
                pts = 10
                reasons = []
            elif risk_level == 'MEDIUM':
                pts = 7
                reasons = []
            elif risk_level == 'HIGH':
                pts = 3
                reasons = ["High geo risk — reduce exposure"]
            else:  # CRITICAL
                pts = 0
                reasons = ["CRITICAL geo risk — avoid trading"]
            
            self._geo_cache = (pts, reasons)
            return pts, reasons
        except Exception as e:
            logger.debug(f"Geo scoring failed: {e}")
            self._geo_cache = (7, [])
            return 7, []
    
    def _build_trade_plan(self, decision: TradeDecision):
        """Build complete trade plan with entry, stop, target, sizing."""
        try:
            from src.strategy.risk_manager import DynamicRiskManager
            rm = DynamicRiskManager(capital=self.capital)
            risk_data = rm.calculate_risk(decision.symbol)
            
            if risk_data:
                decision.entry_price = risk_data.get('current_price', 0)
                decision.stop_loss = risk_data.get('stop_loss', 0)
                decision.position_size = risk_data.get('shares', 0)
                decision.position_value = risk_data.get('position_value', 0)
                decision.max_loss = risk_data.get('max_loss', 0)
                
                # Target = 2x risk (risk:reward = 1:2)
                risk_per_share = decision.entry_price - decision.stop_loss
                decision.target = decision.entry_price + (risk_per_share * 2)
                
                if risk_per_share > 0:
                    decision.risk_reward = 2.0
                
                # Adjust for regime
                regime_pts = decision.regime_score
                if regime_pts < 5:
                    # Reduce position in bad regime
                    decision.position_size = int(decision.position_size * 0.3)
                    decision.position_value = decision.position_size * decision.entry_price
                    decision.max_loss = decision.position_size * risk_per_share
        except Exception as e:
            logger.debug(f"Trade plan failed for {decision.symbol}: {e}")
    
    def print_report(self, decisions: List[TradeDecision]):
        """Print formatted decision report."""
        print("\n" + "═" * 70)
        print("  🧠 DECISION ENGINE — UNIFIED TRADE VERDICTS")
        print("═" * 70)
        
        takes = [d for d in decisions if d.verdict == "TAKE"]
        watches = [d for d in decisions if d.verdict == "WATCH"]
        skips = [d for d in decisions if d.verdict == "SKIP"]
        
        if takes:
            print(f"\n  🟢 TAKE ({len(takes)}):")
            print(f"  {'Symbol':<12} {'Score':>5} {'Conv':>4} {'Signal':>6} {'ML':>4} {'Kelly':>5} {'Regime':>6} {'Reason'}")
            print("  " + "─" * 65)
            for d in takes:
                reason = d.reasons_for[0] if d.reasons_for else ""
                print(f"  {d.symbol:<12} {d.score:>5.0f} {d.conviction:>3}/5 "
                      f"{d.signal_score:>5.0f} {d.ml_score:>4.0f} {d.kelly_score:>5.0f} "
                      f"{d.regime_score:>5.0f}  {reason}")
                if d.entry_price > 0:
                    print(f"  {'':12} → Entry ₹{d.entry_price:,.0f} | Stop ₹{d.stop_loss:,.0f} | "
                          f"Target ₹{d.target:,.0f} | {d.position_size} shares | "
                          f"Risk ₹{d.max_loss:,.0f}")
        
        if watches:
            print(f"\n  🟡 WATCH ({len(watches)}):")
            print(f"  {'Symbol':<12} {'Score':>5} {'Conv':>4} {'Why Watch'}")
            print("  " + "─" * 50)
            for d in watches:
                why = d.reasons_against[0] if d.reasons_against else "Needs more confirmation"
                print(f"  {d.symbol:<12} {d.score:>5.0f} {d.conviction:>3}/5  {why}")
        
        if skips:
            print(f"\n  🔴 SKIP ({len(skips)}):")
            for d in skips[:5]:  # Only show top 5 skips
                why = d.reasons_against[0] if d.reasons_against else "Insufficient score"
                print(f"  {d.symbol:<12} {d.score:>5.0f}  {why}")
            if len(skips) > 5:
                print(f"  ... and {len(skips)-5} more")
        
        print("\n" + "═" * 70)
