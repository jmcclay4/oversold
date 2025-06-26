import sqlite3
import pandas as pd
import numpy as np
import logging
from init_db import calculate_adx_dmi, calculate_stochastic, SP500_TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DB_PATH = "/data/stocks.db"

def recalculate_indicators(ticker, date, period=9, k_period=9, d_period=3):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, open, high, low, close, volume, company_name
        FROM ohlcv
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 30
    """
    df = pd.read_sql_query(query, conn, params=(ticker, date))
    if len(df) < period:
        logger.warning(f"Insufficient data for {ticker} on {date}: {len(df)} rows, need {period}")
        conn.close()
        return
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    df = df.sort_values('date')
    try:
        adx_dmi = calculate_adx_dmi(df, period=period)
        stochastic = calculate_stochastic(df, k_period=k_period, d_period=d_period)
        if adx_dmi is None or stochastic is None or adx_dmi.empty or stochastic.empty:
            logger.warning(f"Failed to calculate indicators for {ticker} on {date}")
            conn.close()
            return
        latest_row = df[df['date'] == date]
        if not latest_row.empty:
            adx = adx_dmi.loc[adx_dmi['date'] == date, 'adx'].iloc[0] if date in adx_dmi['date'].values else None
            pdi = adx_dmi.loc[adx_dmi['date'] == date, 'pdi'].iloc[0] if date in adx_dmi['date'].values else None
            mdi = adx_dmi.loc[adx_dmi['date'] == date, 'mdi'].iloc[0] if date in adx_dmi['date'].values else None
            k = stochastic.loc[stochastic['date'] == date, 'k'].iloc[0] if date in stochastic['date'].values else None
            d = stochastic.loc[stochastic['date'] == date, 'd'].iloc[0] if date in stochastic['date'].values else None
            if all(v is not None for v in [adx, pdi, mdi, k, d]):
                logger.info(f"Calculated for {ticker} on {date}: ADX={adx}, PDI={pdi}, MDI={mdi}, K={k}, D={d}")
                conn.execute("""
                    UPDATE ohlcv
                    SET adx = ?, pdi = ?, mdi = ?, k = ?, d = ?
                    WHERE ticker = ? AND date = ?
                """, (adx, pdi, mdi, k, d, ticker, date))
                conn.commit()
            else:
                logger.warning(f"Missing indicator values for {ticker} on {date}")
        else:
            logger.warning(f"No data for {ticker} on {date}")
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker} on {date}: {e}")
    conn.close()

if __name__ == "__main__":
    target_date = '2025-06-23'
    for ticker in SP500_TICKERS:
        logger.info(f"Processing ticker {ticker}")
        recalculate_indicators(ticker, target_date)