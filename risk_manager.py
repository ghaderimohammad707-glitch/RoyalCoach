"""
Position Sizing & Risk Management Module
- Kelly Criterion for optimal position sizing
- Fixed Risk per Trade
- Leverage calculation for futures
- Max Drawdown protection
- Correlation-based diversification
"""

import math
import statistics
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class PositionSizing:
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_usd: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    kelly_percentage: float
    leverage: int = 1
    notes: str = ""


class RiskManager:
    """
    Advanced position sizing and risk management.
    Based on: Kelly Criterion, Fixed Risk, Max Drawdown
    """

    def __init__(
        self,
        account_balance: float,
        max_risk_per_trade: float = 2.0,  # % of account
        max_daily_loss: float = 5.0,  # % of account
        max_leverage: int = 10,  # For futures
        win_rate: float = 50.0,  # % historical
        avg_win: float = 2.0,  # % average win
        avg_loss: float = -1.0,  # % average loss
    ):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_leverage = max_leverage
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.current_daily_loss = 0.0
        self.trades_today = 0

    # ──────────────────────────────────────────────────────────────────────────
    # KELLY CRITERION
    # ──────────────────────────────────────────────────────────────────────────

    def kelly_criterion(
        self, win_rate: Optional[float] = None, win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Kelly Criterion: f = (bp - q) / b
        where:
        - b = win/loss ratio
        - p = win probability
        - q = loss probability (1-p)

        Returns: percentage of bankroll to risk (0-100)
        """
        p = (win_rate or self.win_rate) / 100
        q = 1 - p

        if win_loss_ratio is None:
            # Calculate from avg_win and avg_loss
            b = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 1.0
        else:
            b = win_loss_ratio

        kelly = (b * p - q) / b if b != 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% for safety

        return round(kelly * 100, 2)

    # ──────────────────────────────────────────────────────────────────────────
    # FIXED RISK SIZING
    # ──────────────────────────────────────────────────────────────────────────

    def fixed_risk_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount_usd: Optional[float] = None,
        market_type: str = "spot",
    ) -> PositionSizing:
        """
        Position sizing based on fixed dollar risk.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount_usd: Dollar amount to risk (if None, use max_risk_per_trade)
            market_type: 'spot' or 'futures'

        Returns: PositionSizing object with details
        """
        # Check daily loss limit
        if self.current_daily_loss >= self.account_balance * (self.max_daily_loss / 100):
            return PositionSizing(
                quantity=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=0,
                position_size_usd=0,
                risk_amount=0,
                reward_amount=0,
                risk_reward_ratio=0,
                kelly_percentage=0,
                notes="❌ Daily loss limit reached. Trading paused.",
            )

        # Calculate risk per trade
        if risk_amount_usd is None:
            risk_amount_usd = self.account_balance * (self.max_risk_per_trade / 100)

        # Calculate quantity based on stop loss distance
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return PositionSizing(
                quantity=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=0,
                position_size_usd=0,
                risk_amount=0,
                reward_amount=0,
                risk_reward_ratio=0,
                kelly_percentage=0,
                notes="❌ Invalid stop loss (same as entry)",
            )

        quantity = risk_amount_usd / risk_per_unit
        position_size_usd = quantity * entry_price

        return PositionSizing(
            quantity=round(quantity, 4),
            entry_price=round(entry_price, 8),
            stop_loss=round(stop_loss, 8),
            take_profit=0,  # Will be set separately
            position_size_usd=round(position_size_usd, 2),
            risk_amount=round(risk_amount_usd, 2),
            reward_amount=0,
            risk_reward_ratio=0,
            kelly_percentage=self.kelly_criterion(),
            notes="✓ Fixed risk position sizing",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # R:R BASED SIZING
    # ──────────────────────────────────────────────────────────────────────────

    def rr_based_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_usd: Optional[float] = None,
    ) -> PositionSizing:
        """
        Position sizing with reward consideration.
        Ensures R:R is at least 1.5:1
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return PositionSizing(
                quantity=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_usd=0,
                risk_amount=0,
                reward_amount=0,
                risk_reward_ratio=0,
                kelly_percentage=0,
                notes="❌ Invalid stop loss",
            )

        rr_ratio = reward / risk
        if rr_ratio < 1.5:
            return PositionSizing(
                quantity=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_usd=0,
                risk_amount=0,
                reward_amount=0,
                risk_reward_ratio=round(rr_ratio, 2),
                kelly_percentage=0,
                notes=f"❌ R:R too low ({rr_ratio:.2f}). Need 1.5+ minimum.",
            )

        # Calculate sizing
        if risk_usd is None:
            risk_usd = self.account_balance * (self.max_risk_per_trade / 100)

        quantity = risk_usd / risk
        position_size_usd = quantity * entry_price
        reward_usd = quantity * reward

        return PositionSizing(
            quantity=round(quantity, 4),
            entry_price=round(entry_price, 8),
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            position_size_usd=round(position_size_usd, 2),
            risk_amount=round(risk_usd, 2),
            reward_amount=round(reward_usd, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            kelly_percentage=self.kelly_criterion(),
            notes=f"✓ R:R {rr_ratio:.2f}:1 - Optimal sizing",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LEVERAGE CALCULATION (FUTURES)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_leverage(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        position_size_usd: float,
    ) -> int:
        """
        Calculate safe leverage for futures trading.
        Ensures liquidation price is far from stop loss.
        """
        risk_per_unit = abs(entry_price - stop_loss)
        risk_percentage = (risk_per_unit / entry_price) * 100

        # Simple rule: leverage inversely proportional to risk
        if risk_percentage >= 10:
            leverage = 1  # Too risky for leverage
        elif risk_percentage >= 5:
            leverage = 2
        elif risk_percentage >= 3:
            leverage = 3
        elif risk_percentage >= 2:
            leverage = 5
        elif risk_percentage >= 1:
            leverage = 10
        else:
            leverage = 15

        # Cap at max leverage
        leverage = min(leverage, self.max_leverage)

        return leverage

    # ──────────────────────────────────────────────────────────────────────────
    # PORTFOLIO HEAT (TOTAL RISK)
    # ──────────────────────────────────────────────────────────────────────────

    def portfolio_heat(self, open_positions: List[dict]) -> dict:
        """
        Calculate total portfolio risk (sum of all open positions' risk).
        Prevents over-leveraging the entire account.
        """
        total_risk_usd = 0
        total_risk_pct = 0

        for pos in open_positions:
            risk = pos.get("risk_amount", 0)
            total_risk_usd += risk

        total_risk_pct = (total_risk_usd / self.account_balance) * 100 if self.account_balance > 0 else 0

        max_portfolio_risk = self.account_balance * (self.max_risk_per_trade * 5 / 100)  # 5x single trade

        return {
            "total_risk_usd": round(total_risk_usd, 2),
            "total_risk_pct": round(total_risk_pct, 2),
            "max_portfolio_risk": round(max_portfolio_risk, 2),
            "safe": total_risk_usd <= max_portfolio_risk,
            "heat_level": "🔴 DANGER" if total_risk_pct > 15 else "🟡 WARNING" if total_risk_pct > 10 else "🟢 SAFE",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # DRAWDOWN TRACKING
    # ──────────────────────────────────────────────────────────────────────────

    def check_drawdown(self, current_balance: float, peak_balance: float) -> dict:
        """Monitor and limit drawdown."""
        drawdown = ((peak_balance - current_balance) / peak_balance * 100) if peak_balance > 0 else 0
        max_allowed_drawdown = self.max_daily_loss * 2  # Daily limit x2 for session

        return {
            "current_drawdown_pct": round(drawdown, 2),
            "max_allowed": max_allowed_drawdown,
            "drawdown_exceeded": drawdown > max_allowed_drawdown,
            "recommendation": "STOP TRADING" if drawdown > max_allowed_drawdown else "CONTINUE",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # CORRELATION-BASED DIVERSIFICATION
    # ──────────────────────────────────────────────────────────────────────────

    def optimal_position_allocation(
        self, symbols: List[str], correlations: dict, total_capital: float
    ) -> dict:
        """
        Allocate position sizes across multiple symbols based on correlation.
        Lower correlation = more diversification possible.

        Args:
            symbols: List of symbols to trade
            correlations: Dict of correlations {(sym1, sym2): value}
            total_capital: Total trading capital

        Returns: Allocation percentages per symbol
        """
        n = len(symbols)
        if n == 0:
            return {}

        # Calculate average correlation
        if correlations:
            avg_corr = statistics.mean(correlations.values())
        else:
            avg_corr = 0

        # Diversification factor: lower correlation allows more positions
        diversification_factor = max(0.5, 1 - avg_corr)

        # Base allocation
        base_allocation = (1 / n) * diversification_factor

        allocation = {}
        for sym in symbols:
            allocation[sym] = round(base_allocation * total_capital, 2)

        return allocation

    # ──────────────────────────────────────────────────────────────────────────
    # RECORD TRADE
    # ──────────────────────────────────────────────────────────────────────────

    def record_trade_result(self, pnl_usd: float, pnl_pct: float):
        """Update daily loss tracking after trade closes."""
        self.current_daily_loss += pnl_usd if pnl_usd < 0 else 0
        self.trades_today += 1

    def reset_daily_limits(self):
        """Reset daily counters (should be called at market close)."""
        self.current_daily_loss = 0.0
        self.trades_today = 0

    # ──────────────────────────────────────────────────────────────────────────
    # QUICK RECOMMENDATION
    # ──────────────────────��───────────────────────────────────────────────────

    def get_recommendation(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_strength: str = "STANDARD",
    ) -> dict:
        """
        All-in-one recommendation for position sizing.
        """
        # Check daily limit
        if self.current_daily_loss >= self.account_balance * (self.max_daily_loss / 100):
            return {
                "recommended": False,
                "reason": "Daily loss limit exceeded",
                "position_size": None,
            }

        # R:R based sizing
        position = self.rr_based_position_size(entry_price, stop_loss, take_profit)

        if position.quantity == 0:
            return {
                "recommended": False,
                "reason": position.notes,
                "position_size": None,
            }

        # Adjust for signal strength
        if signal_strength == "STRONG":
            position.quantity *= 1.2  # 20% larger for strong signals
            position.position_size_usd *= 1.2
        elif signal_strength == "WEAK":
            position.quantity *= 0.8  # 20% smaller for weak signals
            position.position_size_usd *= 0.8

        return {
            "recommended": True,
            "position_size": position,
            "kelly_suggestion": f"{position.kelly_percentage}% of account",
            "leverage": self.calculate_leverage(
                entry_price, stop_loss, self.account_balance, position.position_size_usd
            ),
        }
