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
        logger.warning(f"Insufficient data for Wilder's smoothing: need {period}, got {len(data)}")
        return smoothed
    valid_data = [x for x in data[:period] if x is not None and not np.isnan(x)]
    if len(valid_data) >= period / 2:
        smoothed[period-1] = np.mean(valid_data)
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
        if i == len(data)-1:
            logger.info(f"Wilder's smoothing for last row: Input={data[i]}, Smoothed={smoothed[i]}")
    return smoothed

def init_db():
    logger.info("Starting init_db function...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                company_name TEXT,
                adx REAL,
                pdi REAL,
                mdi REAL,
                k REAL,
                d REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_names (
                ticker TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                last_update TEXT
            )
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO metadata (key, last_update)
            VALUES ('last_ohlcv_update', ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
        conn.commit()
        logger.info(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

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
        tickers = ["ABBV"]  # Test with ABBV only
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

def update_metadata():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, last_update)
            VALUES ('last_ohlcv_update', ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
        conn.commit()
        logger.info("Updated metadata with last OHLCV update time")
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")
    finally:
        conn.close()

def get_cached_company_name(ticker: str) -> str | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT company_name FROM company_names WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        logger.info(f"Retrieved company name for {ticker}: {result[0] if result else None}")
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error retrieving cached company name for {ticker}: {e}")
        return None
    finally:
        conn.close()

def store_company_name(ticker: str, company_name: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO company_names (ticker, company_name)
            VALUES (?, ?)
        """, (ticker, company_name))
        conn.commit()
        logger.info(f"Cached company name for {ticker}")
    except Exception as e:
        logger.error(f"Error storing company name: {e}")
    finally:
        conn.close()

def get_all_tickers() -> List[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        logger.info(f"Retrieved {len(tickers)} tickers from database: {tickers}")
        if not tickers:
            logger.warning("No tickers found in database, using S&P 500 tickers")
            tickers = SP500_TICKERS
        return tickers
    except Exception as e:
        logger.error(f"Error retrieving tickers: {e}")
        return SP500_TICKERS

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
        
        tr = np.array([None] * n, dtype=float)
        dm_plus = np.array([None] * n, dtype=float)
        dm_minus = np.array([None] * n, dtype=float)
        for i in range(1, n):
            if (high[i] is None or low[i] is None or close[i-1] is None or 
                high[i-1] is None or low[i-1] is None or
                high[i] <= low[i] or close[i] <= 0):
                logger.warning(f"Skipping row {i} (Date={df['Date'].iloc[i]}) due to invalid OHLCV: High={high[i]}, Low={low[i]}, Close={close[i]}")
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
        
        pdi = np.array([None] * n, dtype=float)
        mdi = np.array([None] * n, dtype=float)
        for i in range(dmi_period-1, n):
            if smoothed_tr[i] is None or smoothed_tr[i] == 0 or smoothed_dm_plus[i] is None or smoothed_dm_minus[i] is None:
                pdi[i] = mdi[i] = None
            else:
                pdi[i] = 100 * smoothed_dm_plus[i] / smoothed_tr[i]
                mdi[i] = 100 * smoothed_dm_minus[i] / smoothed_tr[i]
            if i == n-1:
                logger.info(f"Last row smoothing: Smoothed_TR={smoothed_tr[i]}, Smoothed_+DM={smoothed_dm_plus[i]}, Smoothed_-DM={smoothed_dm_minus[i]}")
                logger.info(f"Last row indicators: PDI={pdi[i]}, MDI={mdi[i]}")
        
        pdi = np.where(np.isnan(pdi) | (pdi == 0), None, pdi)
        mdi = np.where(np.isnan(mdi) | (mdi == 0), None, mdi)
        
        adx = np.array([None] * n)  # Skip ADX for now to focus on PDI/MDI
        return adx, pdi, mdi
    except Exception as e:
        logger.error(f"Error in ADX/DMI calculation: {e}")
        return np.array([None] * len(df)), np.array([None] * len(df)), np.array([None] * len(df))

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    logger.info("Calculating Stochastic...")
    try:
        n = len(df)
        if n < k_period + d_period - 1:
            logger.warning(f"Not enough data for Stochastic calculation (need {k_period + d_period - 1}, got {n})")
            return np.array([None] * n), np.array([None] * n)
        
        low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        logger.info(f"Stochastic %K for last row: {k.values[-1] if n > 0 else None}, %D: {d.values[-1] if n > 0 else None}")
        
        k = np.where(np.isnan(k) | (k == 0), None, k.values)
        d = np.where(np.isnan(d) | (d == 0), None, d.values)
        
        return k, d
    except Exception as e:
        logger.error(f"Error in Stochastic calculation: {e}")
        return np.array([None] * len(df)), np.array([None] * len(df))

def fetch_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    retries = 3
    delay = 0.2
    cached_name = get_cached_company_name(ticker)
    company_name = cached_name or f"{ticker} Inc."
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching yfinance data for {ticker}, attempt {attempt}, start: {start_date}, end: {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df = df.sort_values('Date').drop_duplicates('Date', keep='last')
            logger.info(f"Fetched {len(df)} rows for {ticker}, dates: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
            if not cached_name:
                company_name = stock.info.get('longName', f"{ticker} Inc.")
                store_company_name(ticker, company_name)
            df['company_name'] = company_name
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns) or df[required_columns].isnull().any().any():
                logger.warning(f"Invalid or missing OHLCV data for {ticker}")
                return pd.DataFrame()
            if (df['Close'] <= 0).any() or (df['High'] <= df['Low']).any():
                logger.warning(f"Suspicious OHLCV data for {ticker} (zero/negative close or high<=low)")
                return pd.DataFrame()
            try:
                adx, pdi, mdi = calculate_adx_dmi(df)
                k, d = calculate_stochastic(df)
            except Exception as e:
                logger.error(f"Indicator calculation failed for {ticker}: {e}")
                adx = pdi = mdi = k = d = np.array([None] * len(df))
            df['adx'] = adx
            df['pdi'] = pdi
            df['mdi'] = mdi
            df['k'] = k
            df['d'] = d
            if df[['adx', 'pdi', 'mdi', 'k', 'd']].iloc[-1].isnull().any():
                logger.warning(f"NaN detected in indicators for {ticker} on {df['Date'].iloc[-1]}")
            else:
                logger.info(f"Last row for {ticker}: Date={df['Date'].iloc[-1]}, ADX={df['adx'].iloc[-1]}, PDI={df['pdi'].iloc[-1]}, MDI={df['mdi'].iloc[-1]}, %K={df['k'].iloc[-1]}")
            return df
        except Exception as e:
            logger.error(f"Retry {attempt}/{retries} for {ticker}: {e}")
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
            else:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                return pd.DataFrame()

def store_stock_data(ticker: str, df: pd.DataFrame):
    if df.empty:
        logger.warning(f"No data to store for {ticker}")
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume, company_name, adx, pdi, mdi, k, d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row['Date'],
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                int(row['Volume']),
                row['company_name'],
                row.get('adx'),
                row.get('pdi'),
                row.get('mdi'),
                row.get('k'),
                row.get('d')
            ))
        conn.commit()
        logger.info(f"Stored data for {ticker}")
    except Exception as e:
        logger.error(f"Error storing data for {ticker}: {e}")
    finally:
        conn.close()

def get_latest_date(ticker: str) -> str | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else None
    except Exception as e:
        logger.error(f"Error getting latest date for {ticker}: {e}")
        return None
    finally:
        conn.close()

def delete_old_data(max_age_days: int = 180):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
        cursor.execute("DELETE FROM ohlcv WHERE date < ?", (cutoff_date,))
        conn.commit()
        logger.info(f"Deleted data older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Error deleting old data: {e}")
    finally:
        conn.close()

def update_data(batch_size: int = 100):
    logger.info("Updating stock data...")
    try:
        tickers = get_all_tickers()
        logger.info(f"Processing {len(tickers)} tickers from database")
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with tickers: {batch}")
            for ticker in batch:
                latest_date = get_latest_date(ticker)
                if latest_date:
                    start_date = (pd.to_datetime(latest_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                    if start_date > end_date:
                        logger.info(f"No new data needed for {ticker}, latest date: {latest_date}")
                        continue
                else:
                    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
                df = fetch_yfinance_data(ticker, start_date, end_date)
                logger.info(f"Fetched {len(df)} rows for {ticker}, last date: {df['Date'].iloc[-1] if not df.empty else 'empty'}")
                store_stock_data(ticker, df)
                time.sleep(0.2)
        delete_old_data(max_age_days=180)
        update_metadata()
        logger.info("Data update completed")
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        raise

if __name__ == "__main__":
    logger.info("Running init_db.py as main script")
    init_db()
    update_data()
    logger.info("init_db.py execution completed")