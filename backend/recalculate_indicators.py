import sqlite3
import pandas as pd
import logging
from init_db import calculate_adx_dmi, calculate_stochastic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DB_PATH = "/data/stocks.db"

def recalculate_indicators(ticker, date):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, open, high, low, close, volume, company_name
        FROM ohlcv
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 30
    """
    df = pd.read_sql_query(query, conn, params=(ticker, date))
    if len(df) < 14:
        logger.warning(f"Insufficient data for {ticker} on {date}: {len(df)} rows")
        conn.close()
        return
    df = df.sort_values('date')
    adx_dmi = calculate_adx_dmi(df)
    stochastic = calculate_stochastic(df)
    if not adx_dmi.empty and not stochastic.empty:
        latest_row = df[df['date'] == date]
        if not latest_row.empty:
            adx = adx_dmi.loc[adx_dmi['date'] == date, 'adx'].iloc[0] if date in adx_dmi['date'].values else None
            pdi = adx_dmi.loc[adx_dmi['date'] == date, 'pdi'].iloc[0] if date in adx_dmi['date'].values else None
            mdi = adx_dmi.loc[adx_dmi['date'] == date, 'mdi'].iloc[0] if date in adx_dmi['date'].values else None
            k = stochastic.loc[stochastic['date'] == date, 'k'].iloc[0] if date in stochastic['date'].values else None
            d = stochastic.loc[stochastic['date'] == date, 'd'].iloc[0] if date in stochastic['date'].values else None
            logger.info(f"Calculated for {ticker} on {date}: ADX={adx}, PDI={pdi}, MDI={mdi}, K={k}, D={d}")
            conn.execute("""
                UPDATE ohlcv
                SET adx = ?, pdi = ?, mdi = ?, k = ?, d = ?
                WHERE ticker = ? AND date = ?
            """, (adx, pdi, mdi, k, d, ticker, date))
            conn.commit()
    else:
        logger.warning(f"Failed to calculate indicators for {ticker} on {date}")
    conn.close()

if __name__ == "__main__":
    recalculate_indicators('A', '2025-06-23')