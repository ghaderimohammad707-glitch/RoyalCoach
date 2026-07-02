"""
Psychology Analysis - Behavioral Pattern Detection & Risk Assessment
Analyzes trading patterns for psychological biases and discipline scoring.
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import aiosqlite

from database import DB_PATH

logger_name = "psychology"


@dataclass
class PsychologyReport:
    total_trades: int
    scores: dict
    streaks: dict
    warnings: list
    patterns: list
    hourly_performance: list
    day_performance: list
    behavioral_insights: list


async def _get_trades(days: int = 30) -> list:
    """Fetch trades from journal (last N days)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        async with db.execute(
            """SELECT * FROM journal 
               WHERE closed_at >= ? 
               ORDER BY closed_at ASC""",
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _calculate_discipline_score(trades: list) -> float:
    """
    Score: 0-100
    Factors:
    - Win rate (30%)
    - Trade frequency (20%)
    - Risk/Reward consistency (20%)
    - Revenge trading (15%)
    - Emotional exits (15%)
    """
    if not trades:
        return 0.0

    score = 50.0  # baseline

    # 1. Win Rate (30%)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    wr_score = (wr / 100) * 30 if wr <= 50 else 30 - ((wr - 50) / 50) * 5
    score += wr_score - 15

    # 2. Frequency (20%) - too many trades = low discipline
    trades_per_day = len(trades) / max(1, _get_days_span(trades))
    if trades_per_day > 10:
        freq_score = 0
    elif trades_per_day > 5:
        freq_score = 5
    elif trades_per_day > 3:
        freq_score = 10
    else:
        freq_score = 20
    score += freq_score - 10

    # 3. Risk/Reward (20%)
    rr_scores = []
    for t in trades:
        if t.get("pnl", 0) > 0:
            rr_scores.append(1)  # Good exit
        elif t.get("pnl", 0) < 0:
            rr_scores.append(-1)  # Bad exit
    rr_avg = statistics.mean(rr_scores) if rr_scores else 0
    rr_score = (rr_avg + 1) / 2 * 20
    score += rr_score - 10

    # 4. Revenge Trading (15%)
    revenge_count = _detect_revenge_trades(trades)
    if revenge_count == 0:
        revenge_score = 15
    elif revenge_count <= 2:
        revenge_score = 10
    elif revenge_count <= 5:
        revenge_score = 5
    else:
        revenge_score = 0
    score += revenge_score - 7.5

    # 5. Emotional Exits (15%)
    emotional = _detect_emotional_trades(trades)
    if emotional <= 1:
        emotional_score = 15
    elif emotional <= 3:
        emotional_score = 10
    elif emotional <= 5:
        emotional_score = 5
    else:
        emotional_score = 0
    score += emotional_score - 7.5

    return max(0, min(100, round(score, 1)))


def _get_days_span(trades: list) -> int:
    """Get number of unique days with trades."""
    if not trades:
        return 1
    dates = set()
    for t in trades:
        closed = t.get("closed_at", "")
        if closed:
            dates.add(closed[:10])
    return max(1, len(dates))


def _detect_revenge_trades(trades: list) -> int:
    """Detect revenge trading: losing trade followed immediately by another."""
    count = 0
    for i in range(len(trades) - 1):
        curr = trades[i]
        next_t = trades[i + 1]

        # Check if current trade lost and next trade opened within 1 hour
        if curr.get("pnl", 0) < 0:
            curr_close = datetime.fromisoformat(curr.get("closed_at", ""))
            next_open = datetime.fromisoformat(next_t.get("opened_at", ""))
            if (next_open - curr_close).total_seconds() < 3600:
                count += 1

    return count


def _detect_emotional_trades(trades: list) -> int:
    """Detect emotional trading: huge gains followed by big losses (profit taking) or vice versa."""
    count = 0
    for i in range(len(trades) - 1):
        pnl = trades[i].get("pnl_pct", 0)
        next_pnl = trades[i + 1].get("pnl_pct", 0)

        # Big win followed by big loss (emotional exit)
        if pnl > 5 and next_pnl < -5:
            count += 1
        # Big loss followed by aggressive trade
        if pnl < -5 and trades[i + 1].get("quantity", 0) > trades[i].get("quantity", 1) * 1.5:
            count += 1

    return count


def _analyze_streaks(trades: list) -> dict:
    """Analyze win/loss streaks."""
    if not trades:
        return {
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "current_streak": 0,
            "current_streak_type": "none",
        }

    win_streak = 0
    loss_streak = 0
    max_win = 0
    max_loss = 0
    current_streak = 0
    current_type = "none"

    for t in trades:
        is_win = t.get("pnl", 0) > 0

        if is_win:
            win_streak += 1
            loss_streak = 0
            max_win = max(max_win, win_streak)
            current_streak = win_streak
            current_type = "win"
        else:
            loss_streak += 1
            win_streak = 0
            max_loss = max(max_loss, loss_streak)
            current_streak = loss_streak
            current_type = "loss"

    return {
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
        "current_streak": current_streak,
        "current_streak_type": current_type,
    }


def _detect_warnings(trades: list, discipline_score: float) -> list:
    """Detect behavioral red flags."""
    warnings = []

    if not trades:
        return warnings

    # 1. Revenge Trading
    revenge = _detect_revenge_trades(trades)
    if revenge > 0:
        warnings.append({
            "title": "Revenge Trading Detected",
            "severity": "high" if revenge > 3 else "medium",
            "message": f"You entered {revenge} trades immediately after losses. This is emotional trading.",
            "advice": "Wait at least 1-2 hours after a loss before trading again. Go for a walk.",
        })

    # 2. Overtrading
    trades_per_day = len(trades) / max(1, _get_days_span(trades))
    if trades_per_day > 10:
        warnings.append({
            "title": "Overtrading Alert",
            "severity": "high",
            "message": f"Average {trades_per_day:.1f} trades/day. High frequency = lower win rate.",
            "advice": "Limit to 3-5 trades per day maximum. Quality > Quantity.",
        })

    # 3. Position Sizing Issues
    quantities = [t.get("quantity", 0) for t in trades if t.get("quantity", 0) > 0]
    if quantities:
        avg_qty = statistics.mean(quantities)
        max_qty = max(quantities)
        if max_qty > avg_qty * 3:
            warnings.append({
                "title": "Inconsistent Position Sizing",
                "severity": "medium",
                "message": f"Position sizes vary wildly (avg {avg_qty:.2f}, max {max_qty:.2f}).",
                "advice": "Use fixed risk per trade (1-2% of portfolio). Use calculator.",
            })

    # 4. Low Win Rate
    wins = len([t for t in trades if t.get("pnl", 0) > 0])
    wr = wins / len(trades) * 100
    if wr < 30:
        warnings.append({
            "title": "Poor Win Rate",
            "severity": "high",
            "message": f"Win rate only {wr:.1f}%. Need at least 40% for profitability.",
            "advice": "Review your signal criteria. Use backtesting to validate.",
        })

    # 5. Discipline Score Low
    if discipline_score < 40:
        warnings.append({
            "title": "Discipline Crisis",
            "severity": "high",
            "message": f"Discipline score {discipline_score:.0f}/100. Trading emotionally.",
            "advice": "Stop trading for 24 hours. Journal review. Reset.",
        })

    return warnings


def _detect_patterns(trades: list) -> list:
    """Detect positive and negative patterns."""
    patterns = []

    if not trades:
        return patterns

    # 1. Consistency
    outcomes = [1 if t.get("pnl", 0) > 0 else -1 for t in trades]
    if len(outcomes) >= 5:
        last_5 = outcomes[-5:]
        if sum(last_5) >= 4:
            patterns.append({
                "type": "positive",
                "text": "✅ Strong consistency: Last 5 trades mostly profitable.",
            })

    # 2. Symbol Specialization
    symbols = {}
    for t in trades:
        sym = t.get("symbol", "")
        symbols[sym] = symbols.get(sym, 0) + 1
    if symbols:
        top_sym, count = max(symbols.items(), key=lambda x: x[1])
        if count >= 5 and count / len(trades) > 0.5:
            patterns.append({
                "type": "positive",
                "text": f"🎯 Specialization: {count} trades on {top_sym}. Expertise builds.",
            })

    # 3. Time-based patterns
    hours = {}
    for t in trades:
        opened = t.get("opened_at", "")
        if opened:
            hour = int(opened.split("T")[1].split(":")[0])
            hours[hour] = hours.get(hour, 0) + 1
    if hours:
        best_hour, best_count = max(hours.items(), key=lambda x: x[1])
        if best_count >= 3:
            patterns.append({
                "type": "neutral",
                "text": f"⏰ Best trading hour: {best_hour}:00 UTC ({best_count} trades).",
            })

    # 4. Losing Streaks
    streaks = _analyze_streaks(trades)
    if streaks["max_loss_streak"] >= 5:
        patterns.append({
            "type": "negative",
            "text": f"❌ Dangerous: Max loss streak of {streaks['max_loss_streak']}. Review strategy.",
        })

    return patterns


def _hourly_performance(trades: list) -> list:
    """Calculate performance by hour of day."""
    hourly = {}

    for t in trades:
        opened = t.get("opened_at", "")
        if not opened:
            continue

        hour = int(opened.split("T")[1].split(":")[0])
        if hour not in hourly:
            hourly[hour] = {"trades": [], "pnl": []}

        hourly[hour]["trades"].append(1)
        hourly[hour]["pnl"].append(t.get("pnl_pct", 0))

    result = []
    for hour in sorted(hourly.keys()):
        data = hourly[hour]
        win_count = len([p for p in data["pnl"] if p > 0])
        wr = (win_count / len(data["pnl"]) * 100) if data["pnl"] else 0
        avg_pnl = statistics.mean(data["pnl"]) if data["pnl"] else 0

        result.append({
            "hour": hour,
            "trades": len(data["trades"]),
            "win_rate": round(wr, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
        })

    return result


def _day_performance(trades: list) -> list:
    """Calculate performance by day of week."""
    days_map = {
        "Monday": "MON",
        "Tuesday": "TUE",
        "Wednesday": "WED",
        "Thursday": "THU",
        "Friday": "FRI",
        "Saturday": "SAT",
        "Sunday": "SUN",
    }

    daily = {}

    for t in trades:
        closed = t.get("closed_at", "")
        if not closed:
            continue

        try:
            dt = datetime.fromisoformat(closed)
            day_name = dt.strftime("%A")
            if day_name not in daily:
                daily[day_name] = {"trades": [], "pnl": []}
            daily[day_name]["trades"].append(1)
            daily[day_name]["pnl"].append(t.get("pnl_pct", 0))
        except Exception:
            continue

    result = []
    for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if day_name not in daily:
            continue
        data = daily[day_name]
        win_count = len([p for p in data["pnl"] if p > 0])
        wr = (win_count / len(data["pnl"]) * 100) if data["pnl"] else 0
        avg_pnl = statistics.mean(data["pnl"]) if data["pnl"] else 0

        result.append({
            "day": days_map[day_name],
            "trades": len(data["trades"]),
            "win_rate": round(wr, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
        })

    return result


def _behavioral_insights(trades: list, discipline_score: float) -> list:
    """Generate actionable insights."""
    insights = []

    if not trades:
        return ["Start trading to get insights!"]

    # 1. Best time
    hourly = _hourly_performance(trades)
    if hourly:
        best = max(hourly, key=lambda x: x["win_rate"])
        insights.append(f"🕐 Trade best at {best['hour']}:00 UTC ({best['win_rate']}% win rate)")

    # 2. Worst time
    if hourly:
        worst = min(hourly, key=lambda x: x["win_rate"])
        insights.append(f"⚠️ Avoid {worst['hour']}:00 UTC (only {worst['win_rate']}% win rate)")

    # 3. Discipline trend
    if len(trades) >= 10:
        recent = trades[-10:]
        past = trades[:-10]
        recent_score = _calculate_discipline_score(recent)
        past_score = _calculate_discipline_score(past)
        if recent_score > past_score + 10:
            insights.append("📈 Improvement: Discipline improving over time!")
        elif recent_score < past_score - 10:
            insights.append("📉 Warning: Discipline declining. Review your process.")

    # 4. Best Symbol
    symbols = {}
    for t in trades:
        sym = t.get("symbol", "")
        if sym not in symbols:
            symbols[sym] = []
        symbols[sym].append(t.get("pnl_pct", 0))

    if symbols:
        best_sym = max(symbols.items(), key=lambda x: statistics.mean(x[1]))
        insights.append(f"🎯 Best symbol: {best_sym[0]} ({statistics.mean(best_sym[1]):.2f}% avg)")

    # 5. Risk Management
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    if losses:
        avg_loss = statistics.mean([t.get("pnl_pct", 0) for t in losses])
        if avg_loss < -5:
            insights.append(
                f"🛑 Risk Alert: Avg loss is {avg_loss:.2f}%. Tighten stops to -2%."
            )

    return insights


async def get_psychology_report() -> dict:
    """Generate complete psychology report."""
    trades = await _get_trades(days=30)

    if not trades:
        return {
            "total_trades": 0,
            "scores": {
                "discipline_score": 0,
                "win_rate": 0,
                "avg_daily_trades": 0,
                "max_daily_trades": 0,
                "revenge_trading_count": 0,
            },
            "streaks": {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "current_streak": 0,
                "current_streak_type": "none",
            },
            "warnings": [],
            "patterns": [],
            "hourly_performance": [],
            "day_performance": [],
            "behavioral_insights": [],
        }

    discipline = _calculate_discipline_score(trades)
    wins = len([t for t in trades if t.get("pnl", 0) > 0])
    wr = wins / len(trades) * 100 if trades else 0
    days_span = _get_days_span(trades)
    revenge_count = _detect_revenge_trades(trades)

    return {
        "total_trades": len(trades),
        "scores": {
            "discipline_score": round(discipline, 1),
            "win_rate": round(wr, 1),
            "avg_daily_trades": round(len(trades) / max(1, days_span), 1),
            "max_daily_trades": len(trades),  # Can be computed more precisely
            "revenge_trading_count": revenge_count,
        },
        "streaks": _analyze_streaks(trades),
        "warnings": _detect_warnings(trades, discipline),
        "patterns": _detect_patterns(trades),
        "hourly_performance": _hourly_performance(trades),
        "day_performance": _day_performance(trades),
        "behavioral_insights": _behavioral_insights(trades, discipline),
    }
