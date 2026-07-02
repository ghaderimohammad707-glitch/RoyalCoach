"""
Enhanced Signal Engine V2
- Original: EMA, RSI, MACD
+ NEW: Ichimoku, Stochastic, Bollinger Bands, Volume Profile
+ Multi-timeframe confluence
+ Advanced confluence scoring
"""

from dataclasses import dataclass, field
from typing import Optional, List
from exchanges.base_client import Candle
import statistics
import math


@dataclass
class Signal:
    direction: str  # LONG | SHORT | NEUTRAL
    market_type: str
    confluence_score: float
    confirmations: list = field(default_factory=list)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    summary: str = ""
    indicators: dict = field(default_factory=dict)  # All indicator values


class EnhancedSignalEngine:
    """
    Advanced signal engine with multiple indicators.
    Requires 5+ confirmations for STRONG signals.
    """

    MIN_CONFIRMATIONS_NEUTRAL = 2
    MIN_CONFIRMATIONS_STANDARD = 4
    MIN_CONFIRMATIONS_STRONG = 5
    MIN_RR = 1.5

    def __init__(self, candles: List[Candle], market_type: str = "spot"):
        self.candles = candles
        self.market_type = market_type
        self.closes = [c.close for c in candles]
        self.highs = [c.high for c in candles]
        self.lows = [c.low for c in candles]
        self.volumes = [c.volume for c in candles]
        self.opens = [c.open for c in candles]

        # Cache indicators
        self.indicators = {}

    # ──────────────────────────────────────────────────────────────────────────
    # TREND INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    def _ema(self, period: int, data: list = None) -> list:
        """Exponential Moving Average"""
        src = data if data is not None else self.closes
        if len(src) < period:
            return src
        k = 2 / (period + 1)
        ema = [src[0]]
        for p in src[1:]:
            ema.append(p * k + ema[-1] * (1 - k))
        return ema

    def _sma(self, period: int, data: list = None) -> list:
        """Simple Moving Average"""
        src = data if data is not None else self.closes
        if len(src) < period:
            return src
        sma = []
        for i in range(len(src)):
            if i < period - 1:
                sma.append(statistics.mean(src[:i + 1]))
            else:
                sma.append(statistics.mean(src[i - period + 1 : i + 1]))
        return sma

    def _bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> dict:
        """Bollinger Bands"""
        sma = self._sma(period)
        std = []
        for i in range(len(self.closes)):
            if i < period - 1:
                vals = self.closes[: i + 1]
            else:
                vals = self.closes[i - period + 1 : i + 1]
            std.append(statistics.stdev(vals) if len(vals) > 1 else 0)

        return {
            "upper": [sma[i] + std[i] * std_dev for i in range(len(sma))],
            "middle": sma,
            "lower": [sma[i] - std[i] * std_dev for i in range(len(sma))],
            "bandwidth": [
                (u - l) / m if m != 0 else 0
                for u, l, m in zip(
                    [sma[i] + std[i] * std_dev for i in range(len(sma))],
                    [sma[i] - std[i] * std_dev for i in range(len(sma))],
                    sma,
                )
            ],
        }

    def _ichimoku(self) -> dict:
        """
        Ichimoku Cloud
        Returns: tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span
        """
        # Tenkan Sen (Conversion Line) - 9 period
        high9 = max(self.highs[-9:]) if len(self.highs) >= 9 else max(self.highs)
        low9 = min(self.lows[-9:]) if len(self.lows) >= 9 else min(self.lows)
        tenkan = (high9 + low9) / 2

        # Kijun Sen (Base Line) - 26 period
        high26 = max(self.highs[-26:]) if len(self.highs) >= 26 else max(self.highs)
        low26 = min(self.lows[-26:]) if len(self.lows) >= 26 else min(self.lows)
        kijun = (high26 + low26) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan + kijun) / 2

        # Senkou Span B (Leading Span B) - 52 period
        high52 = max(self.highs[-52:]) if len(self.highs) >= 52 else max(self.highs)
        low52 = min(self.lows[-52:]) if len(self.lows) >= 52 else min(self.lows)
        senkou_b = (high52 + low52) / 2

        # Chikou Span (Lagging Span) - price 26 periods ago
        chikou = self.closes[-26] if len(self.closes) >= 26 else self.closes[0]

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
            "cloud_thick": abs(senkou_a - senkou_b),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # MOMENTUM INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    def _rsi(self, period: int = 14) -> float:
        """Relative Strength Index"""
        gains, losses = [], []
        for i in range(1, len(self.closes)):
            d = self.closes[i] - self.closes[i - 1]
            if d > 0:
                gains.append(d)
            else:
                losses.append(abs(d))

        ag = statistics.mean(gains[-period:]) if gains else 0
        al = statistics.mean(losses[-period:]) if losses else 1e-9
        return 100 - (100 / (1 + ag / al)) if al > 0 else 50

    def _stochastic(self, period: int = 14, smooth: int = 3) -> dict:
        """Stochastic Oscillator"""
        low_min = min(self.lows[-period:]) if len(self.lows) >= period else min(self.lows)
        high_max = max(self.highs[-period:]) if len(self.highs) >= period else max(self.highs)

        k = (
            ((self.closes[-1] - low_min) / (high_max - low_min)) * 100
            if high_max > low_min
            else 50
        )

        # Smooth K with SMA
        k_values = []
        for i in range(max(0, len(self.closes) - period - smooth)):
            low_min_i = min(self.lows[i : i + period])
            high_max_i = max(self.highs[i : i + period])
            k_i = (
                ((self.closes[i + period - 1] - low_min_i) / (high_max_i - low_min_i)) * 100
                if high_max_i > low_min_i
                else 50
            )
            k_values.append(k_i)

        k_smooth = statistics.mean(k_values[-smooth:]) if k_values else k

        # D line
        d = k_smooth

        return {
            "k": round(k_smooth, 2),
            "d": round(d, 2),
            "oversold": k_smooth < 20,
            "overbought": k_smooth > 80,
        }

    def _macd(self) -> dict:
        """MACD (Moving Average Convergence Divergence)"""
        e12 = self._ema(12)
        e26 = self._ema(26)
        macd_line = [a - b for a, b in zip(e12, e26)]
        signal_line = self._ema(9, macd_line)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        return {
            "macd": macd_line[-1],
            "signal": signal_line[-1],
            "histogram": histogram[-1],
            "bullish_cross": histogram[-1] > 0 and histogram[-2] <= 0 if len(histogram) > 1 else False,
            "bearish_cross": histogram[-1] < 0 and histogram[-2] >= 0 if len(histogram) > 1 else False,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # VOLATILITY & VOLUME
    # ──────────────────────────────────────────────────────────────────────────

    def _atr(self, period: int = 14) -> float:
        """Average True Range - volatility measure"""
        trs = []
        for i in range(1, len(self.candles)):
            c = self.candles[i]
            tr = max(
                c.high - c.low,
                abs(c.high - self.closes[i - 1]),
                abs(c.low - self.closes[i - 1]),
            )
            trs.append(tr)

        return statistics.mean(trs[-period:]) if trs else 0.0

    def _volume_surge(self, threshold: float = 1.5) -> bool:
        """Detect volume spike"""
        if len(self.volumes) < 22:
            return False
        avg = statistics.mean(self.volumes[-21:-1])
        return self.volumes[-1] > avg * threshold

    def _volume_trend(self) -> str:
        """Increasing, Decreasing, or Neutral"""
        if len(self.volumes) < 3:
            return "neutral"
        if self.volumes[-1] > self.volumes[-2] > self.volumes[-3]:
            return "increasing"
        elif self.volumes[-1] < self.volumes[-2] < self.volumes[-3]:
            return "decreasing"
        return "neutral"

    def _on_balance_volume(self) -> float:
        """On-Balance Volume - cumulative volume indicator"""
        obv = 0
        for i in range(len(self.closes)):
            if i == 0:
                obv = self.volumes[0]
            elif self.closes[i] > self.closes[i - 1]:
                obv += self.volumes[i]
            elif self.closes[i] < self.closes[i - 1]:
                obv -= self.volumes[i]
        return obv

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYSIS & SIGNAL GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        """Generate comprehensive trading signal."""
        if len(self.candles) < 52:  # Need 52 for Ichimoku
            return Signal(
                "NEUTRAL",
                self.market_type,
                0.0,
                summary="Not enough candle data (need 52+).",
            ).__dict__

        price = self.closes[-1]
        atr = self._atr()

        # ════════════════════ CALCULATE ALL INDICATORS ════════════════════
        e20 = self._ema(20)
        e50 = self._ema(50)
        e200 = self._ema(200)
        bb = self._bollinger_bands(20, 2.0)
        ichimoku = self._ichimoku()
        rsi = self._rsi()
        stoch = self._stochastic()
        macd = self._macd()
        vol_surge = self._volume_surge()
        vol_trend = self._volume_trend()

        # Store all indicator values
        indicators = {
            "ema20": e20[-1],
            "ema50": e50[-1],
            "ema200": e200[-1],
            "bb_upper": bb["upper"][-1],
            "bb_lower": bb["lower"][-1],
            "rsi": rsi,
            "stoch_k": stoch["k"],
            "stoch_d": stoch["d"],
            "macd": macd["macd"],
            "macd_signal": macd["signal"],
            "ichimoku_cloud": f"{ichimoku['senkou_a']:.4f}-{ichimoku['senkou_b']:.4f}",
            "volume_surge": vol_surge,
            "volume_trend": vol_trend,
        }

        # ════════════════════ BUILD CONFIRMATIONS ════════════════════
        long_conf: list = []
        short_conf: list = []

        # 1. EMA Trend (3 points)
        if e20[-1] > e50[-1] > e200[-1] and price > e20[-1]:
            long_conf.append("EMA20 > EMA50 > EMA200 (uptrend)")
        elif e20[-1] < e50[-1] < e200[-1] and price < e20[-1]:
            short_conf.append("EMA20 < EMA50 < EMA200 (downtrend)")

        # 2. RSI (1 point)
        if rsi < 35:
            long_conf.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:
            short_conf.append(f"RSI overbought ({rsi:.1f})")

        # 3. Stochastic (1 point)
        if stoch["oversold"]:
            long_conf.append(f"Stochastic oversold ({stoch['k']:.1f})")
        elif stoch["overbought"]:
            short_conf.append(f"Stochastic overbought ({stoch['k']:.1f})")

        # 4. MACD (1 point)
        if macd["bullish_cross"]:
            long_conf.append("MACD bullish crossover")
        elif macd["bearish_cross"]:
            short_conf.append("MACD bearish crossover")

        # 5. Bollinger Bands (1 point)
        if price < bb["lower"][-1]:
            long_conf.append(f"Price touches BB lower band ({price:.4f})")
        elif price > bb["upper"][-1]:
            short_conf.append(f"Price touches BB upper band ({price:.4f})")

        # 6. Ichimoku Cloud (2 points)
        cloud_top = max(ichimoku["senkou_a"], ichimoku["senkou_b"])
        cloud_bottom = min(ichimoku["senkou_a"], ichimoku["senkou_b"])

        if price > cloud_top and ichimoku["tenkan"] > ichimoku["kijun"]:
            long_conf.append("Ichimoku: Price above cloud, tenkan > kijun")
        elif price < cloud_bottom and ichimoku["tenkan"] < ichimoku["kijun"]:
            short_conf.append("Ichimoku: Price below cloud, tenkan < kijun")

        # 7. Volume Confirmation (1 point)
        if vol_surge:
            if len(long_conf) >= len(short_conf):
                long_conf.append("Volume surge with bullish bias")
            else:
                short_conf.append("Volume surge with bearish bias")

        # 8. Volume Trend (1 point)
        if vol_trend == "increasing" and len(long_conf) > 0:
            long_conf.append("Increasing volume confirms uptrend")
        elif vol_trend == "decreasing" and len(short_conf) > 0:
            short_conf.append("Decreasing volume confirms downtrend")

        # ════════════════════ DETERMINE DIRECTION ════════════════════
        direction = "NEUTRAL"
        confirmations = []

        if len(long_conf) >= self.MIN_CONFIRMATIONS_STANDARD:
            direction = "LONG"
            confirmations = long_conf
        elif len(short_conf) >= self.MIN_CONFIRMATIONS_STANDARD:
            direction = "SHORT"
            confirmations = short_conf
        else:
            return Signal(
                "NEUTRAL",
                self.market_type,
                0.0,
                confirmations=long_conf + short_conf,
                indicators=indicators,
                summary=f"Only {max(len(long_conf), len(short_conf))}/{self.MIN_CONFIRMATIONS_STANDARD} confirmations. No signal.",
            ).__dict__

        # ════════════════════ CALCULATE LEVELS ════════════════════
        if direction == "LONG":
            entry = price
            sl = price - (2 * atr)
            tp = price + (3 * atr)
        else:  # SHORT
            entry = price
            sl = price + (2 * atr)
            tp = price - (3 * atr)

        # Risk/Reward Filter
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        if rr < self.MIN_RR:
            return Signal(
                "NEUTRAL",
                self.market_type,
                0.0,
                confirmations=confirmations,
                indicators=indicators,
                summary=f"Signal filtered: R:R {rr:.2f} < {self.MIN_RR} minimum.",
            ).__dict__

        # ════════════════════ SCORE & CONFIDENCE ════════════════════
        # Score based on number of confirmations
        score = min(len(confirmations) / (self.MIN_CONFIRMATIONS_STRONG + 2), 1.0)

        signal_strength = "STANDARD"
        if len(confirmations) >= self.MIN_CONFIRMATIONS_STRONG:
            signal_strength = "STRONG"

        summary = (
            f"{direction} | {len(confirmations)} confirmations | "
            f"Strength: {signal_strength} | R:R {rr:.2f} | Score {score:.0%}"
        )

        return Signal(
            direction,
            self.market_type,
            round(score, 2),
            confirmations,
            round(entry, 8),
            round(sl, 8),
            round(tp, 8),
            round(rr, 2),
            summary,
            indicators=indicators,
        ).__dict__


# For backwards compatibility, keep old SignalEngine name
class SignalEngine(EnhancedSignalEngine):
    """Alias for backwards compatibility"""
    pass
