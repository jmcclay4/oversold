import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
import os
import time
from typing import List
from sp500_tickers import SP500_TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "stocks.db")

def wilders_smoothing(data: np.ndarray, period: int) -> np.ndarray:
    smoothed = np.array([None] * len(data), dtype=float)
    if len(data) < period:
        return smoothed
    valid_count = sum(1 for x in data[:period] if x is not None and not np.isnan(x))
    if valid_count >= period / 2:
        smoothed[period-1] = np.mean([x for x in data[:period] if x is not None and not np.isnan(x)])
    for i in range(period, len(data)):
        if data[i] is None or np.isnan(data[i]):
            smoothed[i] = smoothed[i-1]
        elif smoothed[i-1] is None or np.isnan(smoothed[i-1]):
            valid_count = sum(1 for x in data[i-period+1:i+1] if x is not None and not np.isnan(x))
            if valid_count > 0:
                smoothed[i] = np.mean([x for x in data[i-period+1:i+1] if x is not None and not np.isnan(x)])
            else:
                smoothed[i] = None
        else:
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
    return smoothed

def calculate_adx_dmi(df: pd.DataFrame, dmi_period: int = 14, adx_period: int = 14):
    logger.info("Calculating ADX and DMI...")
    try:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        n = len(df)
        if n < dmi_period + 1:
            logger.warning(f"Not enough data for DMI calculation (need {dmi_period + 1}, got {n})")
            return np.array([None] * n), np.array([None] * n), np.array([None] * n)
        
        logger.info(f"Last row OHLCV: Date={df['Date'].iloc[-1]}, High={high[-1]}, Low={low[-1]}, Close={close[-1]}")
        
        tr = np.array([None] * n)
        dm_plus = np.array([None] * n)
        dm_minus = np.array([None] * n)
        for i in range(1, n):
            if (high[i] is None or low[i] is None or close[i-1] is None or 
                high[i-1] is None or low[i-1] is None or
                high[i] <= low[i] or close[i] <= 0):
                logger.warning(f"Skipping row {i} (Date={df['Date'].iloc[i]}) due to invalid OHLCV")
                continue
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
            if i == n-1:
                logger.info(f"Last row (Date={df['Date'].iloc[i]}): TR={tr[i]}, +DM={dm_plus[i]}, -DM={dm_minus[i]}")
        
        smoothed_tr = wilders_smoothing(tr, dmi_period)
        smoothed_dm_plus = wilders_smoothing(dm_plus, dmi_period)
        smoothed_dm_minus = wilders_smoothing(dm_minus, dmi_period)
        
        pdi = np.array([None] * n)
        mdi = np.array([None] * n)
        for i in range(dmi_period-1, n):
            if smoothed_tr[i] is None or smoothed_tr[i] == 0 or smoothed_dm_plus[i] is None or smoothed_dm_minus[i] is None:
                pdi[i] = mdi[i] = None
            else:
                pdi[i] = 100 * smoothed_dm_plus[i] / smoothed_tr[i]
                mdi[i] = 100 * smoothed_dm_minus[i] / smoothed_tr[i]
            if i == n-1:
                logger.info(f"Last row smoothing: Smoothed_TR={smoothed_tr[i]}, Smoothed_+DM={smoothed_dm_plus[i]}, Smoothed_-DM={smoothed_dm_minus[i]}")
                logger.info(f"Last row indicators: PDI={pdi[i]}, MDI={mdi[i]}")
        
        pdi = np.where(pdi == 0, None, pdi)
        mdi = np.where(mdi == 0, None, mdi)
        
        adx = np.array([None] * n)  # Skip ADX
        return adx, pdi, mdi
    except Exception as e:
        logger.error(f"Error in ADX/DMI calculation: {e}")
        return np.array([None] * len(df)), np.array([None] * len(df)), np.array([None] * len(df))

# Rest of init_db.py remains unchanged, but set tickers=["ABBV"] in rebuild_database
def rebuild_database():
    logger.info("Starting database rebuild...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS ohlcv")
        cursor.execute("DROP TABLE IF EXISTS company_names")
        cursor.execute("DROP TABLE IF EXISTS metadata")
        conn.commit()
        logger.info("Existing tables dropped")
        init_db()
        tickers = ["ABBV"]
        logger.info(f"Populating database with {len(tickers)} ticker(s)")
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        for ticker in tickers:
            logger.info(f"Fetching initial data for {ticker} from {start_date} to {end_date}")
            df = fetch_yfinance_data(ticker, start_date, end_date)
            logger.info(f"Fetched {len(df)} rows for {ticker}, last date: {df['Date'].iloc[-1] if not df.empty else 'empty'}")
            store_stock_data(ticker, df)
            time.sleep(0.2)
        update_metadata()
        logger.info("Database rebuild completed")
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# Include other functions from previous init_db.py (omitted for brevity)