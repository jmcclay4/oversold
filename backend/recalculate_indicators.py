import sqlite3
import pandas as pd
import numpy as np
import logging
from init_db import calculate_adx_dmi, calculate_stochastic, SP500_TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DB_PATH = "/data/stocks.db"

def recalculate_indicators(ticker, dmi_period=9, adx_period=9, k_period=9, d_period=3):
    conn = sqlite3.connect(DB_PATH)
    query_dates = """
        SELECT DISTINCT date
        FROM ohlcv
        WHERE ticker = ?
        ORDER BY date
    """
    dates_df = pd.read_sql_query(query_dates, conn, params=(ticker,))
    dates = dates_df['date'].tolist()
    
    for date in dates:
        query = """
            SELECT date, open, high, low, close, volume, company_name
            FROM ohlcv
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 30
        """
        df = pd.read_sql_query(query, conn, params=(ticker, date))
        if len(df) < max(dmi_period + 1, k_period + d_period - 1):
            logger.warning(f"Insufficient data for {ticker} on {date}: {len(df)} rows, need {max(dmi_period + 1, k_period + d_period - 1)}")
            continue
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        df = df.sort_values('date')
        try:
            adx, pdi, mdi = calculate_adx_dmi(df, dmi_period=dmi_period, adx_period=adx_period)
            k, d = calculate_stochastic(df, k_period=k_period, d_period=d_period)
            adx_dmi = pd.DataFrame({
                'date': df['date'],
                'adx': adx,
                'pdi': pdi,
                'mdi': mdi
            })
            stochastic = pd.DataFrame({
                'date': df['date'],
                'k': k,
                'd': d
            })
            if adx_dmi.empty or stochastic.empty:
                logger.warning(f"Failed to calculate indicators for {ticker} on {date}")
                continue
            latest_row = df[df['date'] == date]
            if not latest_row.empty:
                adx_val = adx_dmi.loc[adx_dmi['date'] == date, 'adx'].iloc[0] if date in adx_dmi['date'].values else None
                pdi_val = adx_dmi.loc[adx_dmi['date'] == date, 'pdi'].iloc[0] if date in adx_dmi['date'].values else None
                mdi_val = adx_dmi.loc[adx_dmi['date'] == date, 'mdi'].iloc[0] if date in adx_dmi['date'].values else None
                k_val = stochastic.loc[stochastic['date'] == date, 'k'].iloc[0] if date in stochastic['date'].values else None
                d_val = stochastic.loc[stochastic['date'] == date, 'd'].iloc[0] if date in stochastic['date'].values else None
                if all(v is not None for v in [adx_val, pdi_val, mdi_val, k_val, d_val]):
                    logger.info(f"Calculated for {ticker} on {date}: ADX={adx_val}, PDI={pdi_val}, MDI={mdi_val}, K={k_val}, D={d_val}")
                    conn.execute("""
                        UPDATE ohlcv
                        SET adx = ?, pdi = ?, mdi = ?, k = ?, d = ?
                        WHERE ticker = ? AND date = ?
                    """, (adx_val, pdi_val, mdi_val, k_val, d_val, ticker, date))
                    conn.commit()
                else:
                    logger.warning(f"Missing indicator values for {ticker} on {date}")
            else:
                logger.warning(f"No data for {ticker} on {date}")
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker} on {date}: {e}")
    conn.close()

if __name__ == "__main__":
    for ticker in SP500_TICKERS:
        logger.info(f"Processing ticker {ticker}")
        recalculate_indicators(ticker)