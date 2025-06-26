import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
import os
import time
from typing import List, Optional
from sp500_tickers import SP500_TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/data/stocks.db"

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
            smoothed[i] = None
        elif smoothed[i-1] is None or np.isnan(smoothed[i-1]):
            valid_count = sum(1 for x in data[i-period+1:i+1] if x is not None and not np.isnan(x))
            if valid_count > 0:
                smoothed[i] = np.mean([x for x in data[i-period+1:i+1] if x is not None and not np.isnan(x)])
            else:
                smoothed[i] = None
        else:
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        if i == len(data)-1:
            logger.info(f"Wilder's smoothing: Input={data[i]}, Smoothed={smoothed[i]}")
    return smoothed

def init_db():
    logger.info("Starting init_db function")
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
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
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def rebuild_database():
    logger.info("Starting database rebuild")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS ohlcv")
        cursor.execute("DROP TABLE IF EXISTS company_names")
        cursor.execute("DROP TABLE IF EXISTS metadata")
        conn.commit()
        logger.info("Existing tables dropped")
        init_db()
        tickers = [t for t in SP500_TICKERS if isinstance(t, str)]
        logger.info(f"Populating database with {len(tickers)} valid ticker(s)")
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=181)).strftime('%Y-%m-%d')
        for ticker in tickers:
            logger.info(f"Fetching data for {ticker}")
            df = fetch_yfinance_data(ticker, start_date, end_date)
            logger.info(f"Fetched {len(df)} rows for {ticker}, last date: {df['date'].iloc[-1] if not df.empty else 'empty'}")
            if not df.empty:
                try:
                    adx, pdi, mdi = calculate_adx_dmi(df)
                    k, d = calculate_stochastic(df)
                    df['adx'] = adx
                    df['pdi'] = pdi
                    df['mdi'] = mdi
                    df['k'] = k
                    df['d'] = d
                except Exception as e:
                    logger.error(f"Indicator calculation failed for {ticker}: {e}")
                    df['adx'] = df['pdi'] = df['mdi'] = df['k'] = df['d'] = None
                store_stock_data(ticker, df)
            time.sleep(0.2)
        logger.info(f"Successfully populated database with data for {len(tickers)} tickers")
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

def get_cached_company_name(ticker: str) -> Optional[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT company_name FROM company_names WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        logger.info(f"Retrieved company name for {ticker}: {result[0] if result else None}")
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error retrieving company name for {ticker}: {e}")
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
        logger.info(f"Retrieved {len(tickers)} tickers from database")
        if not tickers:
            logger.warning("No tickers found, using S&P 500 tickers")
            tickers = [t for t in SP500_TICKERS if isinstance(t, str)]
        return tickers
    except Exception as e:
        logger.error(f"Error retrieving tickers: {e}")
        return [t for t in SP500_TICKERS if isinstance(t, str)]

def get_latest_date(ticker: str) -> Optional[str]:
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

def get_latest_db_date() -> Optional[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM ohlcv")
        result = cursor.fetchone()
        latest_date = result[0] if result and result[0] else None
        logger.info(f"Latest date in database: {latest_date}")
        return latest_date
    except Exception as e:
        logger.error(f"Error retrieving latest date: {e}")
        return None
    finally:
        conn.close()

def trim_excess_entries(ticker: str, max_entries: int = 200):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE ticker = ?", (ticker,))
        count = cursor.fetchone()[0]
        if count > max_entries:
            cursor.execute("""
                DELETE FROM ohlcv
                WHERE ticker = ? AND date IN (
                    SELECT date FROM ohlcv
                    WHERE ticker = ?
                    ORDER BY date ASC
                    LIMIT ?
                )
            """, (ticker, ticker, count - max_entries))
            conn.commit()
            logger.info(f"Deleted {count - max_entries} oldest entries for {ticker}")
    except Exception as e:
        logger.error(f"Error trimming entries for {ticker}: {e}")
    finally:
        conn.close()

def get_historical_data(ticker: str, days: int = 30) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date, open, high, low, close, volume, company_name
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker, days))
        df = df.rename(columns={'date': 'date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df.sort_values('date')  # Sort ascending for indicator calculations
        logger.info(f"Retrieved {len(df)} rows of historical data for {ticker}, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error retrieving historical data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def calculate_adx_dmi(df: pd.DataFrame, dmi_period: int = 14, adx_period: int = 14):
    logger.info("Calculating ADX and DMI")
    try:
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for ADX/DMI: {df.columns}")
            return np.array([None] * len(df)), np.array([None] * len(df)), np.array([None] * len(df))
        
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        n = len(df)
        if n < dmi_period + 1:
            logger.warning(f"Not enough data for DMI (need {dmi_period + 1}, got {n})")
            return np.array([None] * n), np.array([None] * n), np.array([None] * n)
        
        logger.info(f"Last row OHLCV: Date={df['date'].iloc[-1]}, High={high[-1]}, Low={low[-1]}, Close={close[-1]}")
        
        tr = np.array([None] * n, dtype=float)
        dm_plus = np.array([None] * n, dtype=float)
        dm_minus = np.array([None] * n, dtype=float)
        for i in range(1, n):
            if (high[i] is None or low[i] is None or close[i-1] is None or 
                high[i-1] is None or low[i-1] is None or
                high[i] <= low[i] or close[i] <= 0):
                logger.warning(f"Skipping row {i} (Date={df['date'].iloc[i]}): High={high[i]}, Low={low[i]}, Close={close[i]}")
                continue
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
            if i == n-1:
                logger.info(f"Last row (Date={df['date'].iloc[i]}): TR={tr[i]}, +DM={dm_plus[i]}, -DM={dm_minus[i]}")
        
        smoothed_tr = wilders_smoothing(tr, dmi_period)
        smoothed_dm_plus = wilders_smoothing(dm_plus, dmi_period)
        smoothed_dm_minus = wilders_smoothing(dm_minus, dmi_period)
        
        pdi = np.array([None] * n, dtype=float)
        mdi = np.array([None] * n, dtype=float)
        dx = np.array([None] * n, dtype=float)
        for i in range(dmi_period-1, n):
            if smoothed_tr[i] is None or smoothed_tr[i] == 0 or smoothed_dm_plus[i] is None or smoothed_dm_minus[i] is None:
                pdi[i] = mdi[i] = dx[i] = None
                continue
            pdi[i] = 100 * smoothed_dm_plus[i] / smoothed_tr[i]
            mdi[i] = 100 * smoothed_dm_minus[i] / smoothed_tr[i]
            di_sum = pdi[i] + mdi[i]
            if di_sum == 0:
                dx[i] = 0
            else:
                dx[i] = 100 * abs(pdi[i] - mdi[i]) / di_sum
            if i == n-1:
                logger.info(f"Last row smoothing: Smoothed_TR={smoothed_tr[i]}, Smoothed_+DM={smoothed_dm_plus[i]}, Smoothed_-DM={smoothed_dm_minus[i]}")
                logger.info(f"Last row indicators: PDI={pdi[i]}, MDI={mdi[i]}, DX={dx[i]}")
        
        adx = wilders_smoothing(dx, adx_period)
        for i in range(dmi_period-1, n):
            if i == n-1 and adx[i] is not None:
                logger.info(f"Last row ADX: {adx[i]}")
        
        adx = np.where(np.isnan(adx) | (adx == 0), None, adx)
        pdi = np.where(np.isnan(pdi) | (pdi == 0), None, pdi)
        mdi = np.where(np.isnan(mdi) | (mdi == 0), None, mdi)
        
        return adx, pdi, mdi
    except Exception as e:
        logger.error(f"Error in ADX/DMI calculation: {e}")
        return np.array([None] * len(df)), np.array([None] * len(df)), np.array([None] * len(df))

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    logger.info("Calculating Stochastic")
    try:
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for Stochastic: {df.columns}")
            return np.array([None] * len(df)), np.array([None] * len(df))
        
        n = len(df)
        if n < k_period + d_period - 1:
            logger.warning(f"Not enough data for Stochastic (need {k_period + d_period - 1}, got {n})")
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
    if not isinstance(ticker, str):
        logger.error(f"Invalid ticker type for {ticker}: expected str, got {type(ticker)}")
        return pd.DataFrame()
    retries = 5  # Increased retries
    delay = 1.0  # Increased delay
    cached_name = get_cached_company_name(ticker)
    company_name = cached_name or f"{ticker} Inc."
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching yfinance data for {ticker}, attempt {attempt}, start: {start_date}, end: {end_date}")
            stock = yf.Ticker(ticker.upper())
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.sort_values('date').drop_duplicates('date', keep='last')
            logger.info(f"Fetched {len(df)} rows for {ticker}, dates: {df['date'].iloc[0]} to {df['date'].iloc[-1]}, columns: {list(df.columns)}")
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
        required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for storage: {df.columns}")
            return
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume, company_name, adx, pdi, mdi, k, d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row['date'],
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close'),
                int(row.get('Volume', 0)),
                row.get('company_name'),
                row.get('adx'),
                row.get('pdi'),
                row.get('mdi'),
                row.get('k'),
                row.get('d')
            ))
        conn.commit()
        logger.info(f"Stored {len(df)} rows for {ticker}")
    except Exception as e:
        logger.error(f"Error storing data for {ticker}: {e}")
        raise
    finally:
        conn.close()

def update_data(batch_size: int = 100, max_entries: int = 200, historical_days: int = 30, use_db_only: bool = False):
    logger.info("Updating stock data")
    try:
        # Get the latest date in the database
        latest_db_date = get_latest_db_date()
        current_date = datetime.now().date()
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
        if not latest_db_date:
            logger.warning("No data in database, triggering full rebuild")
            rebuild_database()
            return
        
        # Check if data is stale (not same as current date)
        latest_db_datetime = pd.to_datetime(latest_db_date).date()
        logger.info(f"Latest database date: {latest_db_date}, current date: {current_date}")
        if latest_db_datetime >= current_date:
            logger.info("Database is up-to-date (same as or newer than current date), no update needed")
            return
        
        # Generate list of missing dates
        missing_dates = []
        current = latest_db_datetime + timedelta(days=1)
        while current <= current_date - timedelta(days=1):
            missing_dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        logger.info(f"Missing dates to fetch: {missing_dates}")
        
        # Verify ticker consistency
        tickers = get_all_tickers()
        logger.info(f"Processing {len(tickers)} tickers from database")
        inconsistent_tickers = []
        for ticker in tickers:
            ticker_latest_date = get_latest_date(ticker)
            if ticker_latest_date != latest_db_date:
                inconsistent_tickers.append((ticker, ticker_latest_date))
        if inconsistent_tickers:
            logger.warning(f"Inconsistent latest dates detected: {inconsistent_tickers}")
            logger.info("Triggering full rebuild to ensure consistency")
            rebuild_database()
            return
        
        # Define required columns
        required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'company_name']
        
        # Process each ticker
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with tickers: {batch}")
            for ticker in batch:
                # Fetch historical data (last 30 days)
                historical_df = get_historical_data(ticker, days=historical_days)
                if historical_df.empty and latest_db_date:
                    logger.warning(f"No historical data for {ticker}, fetching full range")
                
                # Fetch new data from yfinance unless using db only
                new_df = pd.DataFrame()
                if not use_db_only and missing_dates:
                    start_date = min(missing_dates)
                    new_df = fetch_yfinance_data(ticker, start_date, end_date)
                    logger.info(f"Fetched {len(new_df)} rows for {ticker}, last date: {new_df['date'].iloc[-1] if not new_df.empty else 'empty'}, columns: {list(new_df.columns) if not new_df.empty else 'empty'}")
                
                # Combine historical and new data
                combined_df = pd.concat([historical_df, new_df], ignore_index=True)
                if combined_df.empty:
                    logger.warning(f"Combined DataFrame is empty for {ticker}, skipping")
                    continue
                
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                logger.info(f"Combined {len(combined_df)} rows for {ticker}, dates: {combined_df['date'].iloc[0] if not combined_df.empty else 'empty'} to {combined_df['date'].iloc[-1] if not combined_df.empty else 'empty'}, columns: {list(combined_df.columns)}")
                
                # Ensure consistent columns
                missing_cols = [col for col in required_columns if col not in combined_df.columns]
                for col in missing_cols:
                    combined_df[col] = None
                combined_df = combined_df[required_columns]
                
                # Recalculate indicators if enough data
                if len(combined_df) >= max(14 + 1, 14 + 3):
                    try:
                        adx, pdi, mdi = calculate_adx_dmi(combined_df)
                        k, d = calculate_stochastic(combined_df)
                        combined_df['adx'] = adx
                        combined_df['pdi'] = pdi
                        combined_df['mdi'] = mdi
                        combined_df['k'] = k
                        combined_df['d'] = d
                    except Exception as e:
                        logger.error(f"Indicator calculation failed for {ticker}: {e}")
                        combined_df['adx'] = combined_df['pdi'] = combined_df['mdi'] = combined_df['k'] = combined_df['d'] = None
                
                # Store recalculated indicators for the latest historical date or new data
                update_df = combined_df[combined_df['date'] >= latest_db_date]
                if not update_df.empty:
                    store_stock_data(ticker, update_df)
                    trim_excess_entries(ticker, max_entries)
                else:
                    logger.warning(f"No data to store for {ticker} after date {latest_db_date}")
                time.sleep(0.5)  # Increased delay for yfinance stability
        
        # Delete data older than 180 days
        delete_old_data(max_age_days=180)
        update_metadata()
        logger.info("Data update completed")
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        raise
    logger.info("Updating stock data")
    try:
        # Get the latest date in the database
        latest_db_date = get_latest_db_date()
        current_date = datetime.now().date()
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
        if not latest_db_date:
            logger.warning("No data in database, triggering full rebuild")
            rebuild_database()
            return
        
        # Check if data is stale (not same as current date)
        latest_db_datetime = pd.to_datetime(latest_db_date).date()
        logger.info(f"Latest database date: {latest_db_date}, current date: {current_date}")
        if latest_db_datetime >= current_date:
            logger.info("Database is up-to-date (same as or newer than current date), no update needed")
            return
        
        # Generate list of missing dates
        missing_dates = []
        current = latest_db_datetime + timedelta(days=1)
        while current <= current_date - timedelta(days=1):
            missing_dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        logger.info(f"Missing dates to fetch: {missing_dates}")
        
        # Verify ticker consistency
        tickers = get_all_tickers()
        logger.info(f"Processing {len(tickers)} tickers from database")
        inconsistent_tickers = []
        for ticker in tickers:
            ticker_latest_date = get_latest_date(ticker)
            if ticker_latest_date != latest_db_date:
                inconsistent_tickers.append((ticker, ticker_latest_date))
        if inconsistent_tickers:
            logger.warning(f"Inconsistent latest dates detected: {inconsistent_tickers}")
            logger.info("Triggering full rebuild to ensure consistency")
            rebuild_database()
            return
        
        # Define required columns
        required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'company_name']
        
        # Process each ticker
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with tickers: {batch}")
            for ticker in batch:
                # Fetch historical data (last 30 days)
                historical_df = get_historical_data(ticker, days=historical_days)
                if historical_df.empty and latest_db_date:
                    logger.warning(f"No historical data for {ticker}, fetching full range")
                
                # Fetch new data from yfinance unless using db only
                new_df = pd.DataFrame()
                if not use_db_only and missing_dates:
                    start_date = min(missing_dates)
                    new_df = fetch_yfinance_data(ticker, start_date, end_date)
                    logger.info(f"Fetched {len(new_df)} rows for {ticker}, last date: {new_df['date'].iloc[-1] if not new_df.empty else 'empty'}, columns: {list(new_df.columns) if not new_df.empty else 'empty'}")
                
                # Combine historical and new data
                combined_df = pd.concat([historical_df, new_df], ignore_index=True)
                if combined_df.empty:
                    logger.warning(f"Combined DataFrame is empty for {ticker}, skipping")
                    continue
                
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                logger.info(f"Combined {len(combined_df)} rows for {ticker}, dates: {combined_df['date'].iloc[0] if not combined_df.empty else 'empty'} to {combined_df['date'].iloc[-1] if not combined_df.empty else 'empty'}, columns: {list(combined_df.columns)}")
                
                # Ensure consistent columns
                missing_cols = [col for col in required_columns if col not in combined_df.columns]
                for col in missing_cols:
                    combined_df[col] = None
                combined_df = combined_df[required_columns]
                
                # Recalculate indicators if enough data
                if len(combined_df) >= max(14 + 1, 14 + 3):
                    try:
                        adx, pdi, mdi = calculate_adx_dmi(combined_df)
                        k, d = calculate_stochastic(combined_df)
                        combined_df['adx'] = adx
                        combined_df['pdi'] = pdi
                        combined_df['mdi'] = mdi
                        combined_df['k'] = k
                        combined_df['d'] = d
                    except Exception as e:
                        logger.error(f"Indicator calculation failed for {ticker}: {e}")
                        combined_df['adx'] = combined_df['pdi'] = combined_df['mdi'] = combined_df['k'] = combined_df['d'] = None
                
                # Store recalculated indicators for the latest historical date or new data
                update_df = combined_df[combined_df['date'] >= latest_db_date]
                if not update_df.empty:
                    store_stock_data(ticker, update_df)
                    trim_excess_entries(ticker, max_entries)
                else:
                    logger.warning(f"No data to store for {ticker} after date {latest_db_date}")
                time.sleep(0.2)
        
        # Delete data older than 180 days
        delete_old_data(max_age_days=180)
        update_metadata()
        logger.info("Data update completed")
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        raise
    logger.info("Updating stock data")
    try:
        # Get the latest date in the database
        latest_db_date = get_latest_db_date()
        if not latest_db_date:
            logger.warning("No data in database, triggering full rebuild")
            rebuild_database()
            return
        
        # Check if data is stale (not same as current date)
        latest_db_datetime = pd.to_datetime(latest_db_date).date()
        current_date = datetime.now().date()
        logger.info(f"Latest database date: {latest_db_date}, current date: {current_date}")
        if latest_db_datetime >= current_date:
            logger.info("Database is up-to-date (same as or newer than current date), no update needed")
            return
        
        # Determine the most recent available date (yesterday)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (latest_db_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Verify the latest date is consistent across tickers
        tickers = get_all_tickers()
        logger.info(f"Processing {len(tickers)} tickers from database")
        inconsistent_tickers = []
        for ticker in tickers:
            ticker_latest_date = get_latest_date(ticker)
            if ticker_latest_date != latest_db_date:
                inconsistent_tickers.append((ticker, ticker_latest_date))
        if inconsistent_tickers:
            logger.warning(f"Inconsistent latest dates detected: {inconsistent_tickers}")
            logger.info("Triggering full rebuild to ensure consistency")
            rebuild_database()
            return
        
        # Define required columns
        required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'company_name']
        
        # Fetch and update data for each ticker
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with tickers: {batch}")
            for ticker in batch:
                # Fetch historical data from database (last 30 days)
                historical_df = get_historical_data(ticker, days=historical_days)
                if historical_df.empty and latest_db_date:
                    logger.warning(f"No historical data for {ticker}, fetching full range")
                
                # Fetch new data from yfinance
                new_df = fetch_yfinance_data(ticker, start_date, end_date)
                logger.info(f"Fetched {len(new_df)} rows for {ticker}, last date: {new_df['date'].iloc[-1] if not new_df.empty else 'empty'}, columns: {list(new_df.columns) if not new_df.empty else 'empty'}")
                
                # Check if both DataFrames are empty
                if historical_df.empty and new_df.empty:
                    logger.warning(f"No data available for {ticker}, skipping")
                    continue
                
                # Ensure consistent columns
                if not historical_df.empty:
                    historical_df = historical_df[required_columns]
                if not new_df.empty:
                    new_df = new_df[required_columns]
                
                # Combine historical and new data
                combined_df = pd.concat([historical_df, new_df], ignore_index=True)
                if combined_df.empty:
                    logger.warning(f"Combined DataFrame is empty for {ticker}, skipping")
                    continue
                
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                logger.info(f"Combined {len(combined_df)} rows for {ticker}, dates: {combined_df['date'].iloc[0] if not combined_df.empty else 'empty'} to {combined_df['date'].iloc[-1] if not combined_df.empty else 'empty'}, columns: {list(combined_df.columns)}")
                
                # Recalculate indicators if there's enough data
                if len(combined_df) >= max(14 + 1, 14 + 3):
                    try:
                        adx, pdi, mdi = calculate_adx_dmi(combined_df)
                        k, d = calculate_stochastic(combined_df)
                        combined_df['adx'] = adx
                        combined_df['pdi'] = pdi
                        combined_df['mdi'] = mdi
                        combined_df['k'] = k
                        combined_df['d'] = d
                    except Exception as e:
                        logger.error(f"Indicator calculation failed for {ticker}: {e}")
                        combined_df['adx'] = combined_df['pdi'] = combined_df['mdi'] = combined_df['k'] = combined_df['d'] = None
                
                # Store only the new data (from start_date onward)
                if not new_df.empty:
                    update_df = combined_df[combined_df['date'] >= start_date]
                    if not update_df.empty:
                        store_stock_data(ticker, update_df)
                        trim_excess_entries(ticker, max_entries)
                time.sleep(0.2)
        
        # Delete data older than 180 days
        delete_old_data(max_age_days=180)
        update_metadata()
        logger.info("Data update completed")
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        raise

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

if __name__ == "__main__":
    logger.info("Running init_db.py as main script")
    init_db()
    update_data()
    logger.info("init_db.py execution completed")