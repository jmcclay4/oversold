import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
import os
import time
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "stocks.db")

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
        # Insert initial metadata record if not exists
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

def calculate_adx_dmi(df: pd.DataFrame, dmi_period: int = 14, adx_period: int = 14):
    logger.info("Calculating ADX and DMI...")
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        tr[i] = max(high[i] - low[i], abs(h_diff), abs(l_diff))
        dm_plus[i] = h_diff if h_diff > l_diff and h_diff > 0 else 0
        dm_minus[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0
    atr = pd.Series(tr).rolling(window=dmi_period).mean().values
    di_plus = 100 * pd.Series(dm_plus).rolling(window=dmi_period).mean() / atr
    di_minus = 100 * pd.Series(dm_minus).rolling(window=dmi_period).mean() / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=adx_period).mean().values
    return adx, di_plus, di_minus

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    logger.info("Calculating Stochastic...")
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    return k.values, d.values

def fetch_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    retries = 2
    delay = 1
    cached_name = get_cached_company_name(ticker)
    company_name = cached_name or f"{ticker} Inc."
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching yfinance data for {ticker}, attempt {attempt}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            if not cached_name:
                company_name = stock.info.get('longName', f"{ticker} Inc.")
                store_company_name(ticker, company_name)
            df['company_name'] = company_name
            adx, pdi, mdi = calculate_adx_dmi(df)
            k, d = calculate_stochastic(df)
            df['adx'] = adx
            df['pdi'] = pdi
            df['mdi'] = mdi
            df['k'] = k
            df['d'] = d
            logger.info(f"Successfully fetched data for {ticker}")
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
                row.get('adx', None),
                row.get('pdi', None),
                row.get('mdi', None),
                row.get('k', None),
                row.get('d', None)
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

def update_data(batch_size: int = 50):
    logger.info("Updating stock data...")
    try:
        ticker_file = os.path.join(os.path.dirname(__file__), "tickers.txt")
        if not os.path.exists(ticker_file):
            logger.error(f"Ticker file {ticker_file} not found, using default tickers")
            tickers = ["MMM", "AOS", "ABT", "TSLA"]
        else:
            with open(ticker_file, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers from {ticker_file}")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} tickers")
            for ticker in batch:
                latest_date = get_latest_date(ticker)
                if latest_date:
                    start_date = (pd.to_datetime(latest_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                if start_date <= end_date:
                    df = fetch_yfinance_data(ticker, start_date, end_date)
                    store_stock_data(ticker, df)
                    time.sleep(1)
        
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