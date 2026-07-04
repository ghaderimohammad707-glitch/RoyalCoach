"""
Database Optimization - Indexes & Query Performance
Improves query speed for market data, journal, and portfolio tracking
"""

# This file contains the SQL statements to optimize database performance
# Run these commands in SQLite to add indexes

DATABASE_OPTIMIZATION_SQL = """
-- ═══════════════════════════════════════════════════════════════════════════
-- MARKET DATA INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast lookups by exchange, symbol, and time (most common queries)
CREATE INDEX IF NOT EXISTS idx_market_data_exchange_symbol_time 
ON market_data(exchange_id, symbol, timestamp DESC);

-- Fast queries by timeframe
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe 
ON market_data(timeframe);

-- ═══════════════════════════════════════════════════════════════════════════
-- JOURNAL INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast journal queries by date (most common)
CREATE INDEX IF NOT EXISTS idx_journal_closed_date 
ON journal(closed_at DESC);

-- Filter by exchange and symbol
CREATE INDEX IF NOT EXISTS idx_journal_exchange_symbol 
ON journal(exchange_id, symbol);

-- Filter by direction
CREATE INDEX IF NOT EXISTS idx_journal_direction 
ON journal(direction);

-- ═══════════════════════════════════════════════════════════════════════════
-- TICKER SNAPSHOTS INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Primary query: get latest price for symbol
CREATE INDEX IF NOT EXISTS idx_ticker_exchange_symbol 
ON ticker_snapshots(exchange_id, symbol);

-- ═══════════════════════════════════════════════════════════════════════════
-- SIGNALS INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast signal lookup by symbol
CREATE INDEX IF NOT EXISTS idx_signals_exchange_symbol 
ON signals(exchange_id, symbol);

-- Filter recent signals
CREATE INDEX IF NOT EXISTS idx_signals_created_date 
ON signals(created_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- ALERTS INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Recent alerts
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
ON alerts(ts DESC);

-- Filter by level and category
CREATE INDEX IF NOT EXISTS idx_alerts_level_category 
ON alerts(level, category);

-- ═══════════════════════════════════════════════════════════════════════════
-- PORTFOLIO SNAPSHOTS INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast history queries
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts 
ON portfolio_snapshots(ts DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- TRADES INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast trade lookups
CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol 
ON trades(exchange_id, symbol);

-- Filter by date
CREATE INDEX IF NOT EXISTS idx_trades_closed_date 
ON trades(closed_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- EXCHANGE KEYS INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

-- Fast exchange lookup
CREATE INDEX IF NOT EXISTS idx_exchange_keys_id 
ON exchange_keys(exchange_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- VACUUM & ANALYZE (run periodically for optimization)
-- ═══════════════════════════════════════════════════════════════════════════

-- Rebuild database to reclaim unused space
-- VACUUM;

-- Update statistics for query planner
-- ANALYZE;
"""


async def optimize_database(db_connection):
    """Apply all database optimizations."""
    optimizations = DATABASE_OPTIMIZATION_SQL.split(";")
    
    for sql in optimizations:
        sql = sql.strip()
        if sql and not sql.startswith("--"):
            try:
                await db_connection.execute(sql)
                await db_connection.commit()
            except Exception as e:
                print(f"⚠️  Optimization skipped (may already exist): {e}")
    
    print("✅ Database optimization complete!")


# Optimized query examples for reference

OPTIMIZED_QUERIES = {
    "get_latest_prices": """
        SELECT exchange_id, symbol, price, change_24h, volume_24h
        FROM ticker_snapshots
        WHERE exchange_id = ? AND symbol = ?
        ORDER BY ts DESC LIMIT 1;
    """,
    
    "get_market_data": """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE exchange_id = ? AND symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC LIMIT ?;
    """,
    
    "get_journal_stats": """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade
        FROM journal
        WHERE closed_at >= datetime('now', '-30 days');
    """,
    
    "get_recent_alerts": """
        SELECT level, category, title, message, ts
        FROM alerts
        WHERE ts >= datetime('now', '-1 day')
        ORDER BY ts DESC
        LIMIT 50;
    """,
    
    "get_portfolio_history": """
        SELECT total_usdt, ts
        FROM portfolio_snapshots
        WHERE ts >= datetime('now', '-30 days')
        ORDER BY ts ASC;
    """,
    
    "get_trades_by_symbol": """
        SELECT symbol, direction, entry_price, exit_price, pnl, pnl_pct, closed_at
        FROM journal
        WHERE exchange_id = ? AND symbol = ?
        ORDER BY closed_at DESC
        LIMIT 100;
    """,
}
