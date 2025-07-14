# init_db.py
# This file manages the SQLite database for stock data, fetching OHLCV from yfinance,
# calculating technical indicators (ADX, +DI, -DI, Stochastic %K/%D), and storing results.
# Key features:
# - Initializes tables for OHLCV and metadata.
# - Updates data incrementally, checking last date per ticker (MAX(date) query).
# - Processes tickers in batches (BATCH_SIZE=10) to minimize memory usage on Fly.io (256MB limit).
# - Per ticker: Fetches buffer (30 days before last date) from DB, new data from yfinance (from last_date+1 to today+1).
# - Skips if api_start_date >= end_date (no new data needed, avoids yfinance error for future dates).
# - Flattens columns if MultiIndex from yfinance.
# - Concatenates, calculates indicators, inserts new rows.
# - Commits after each batch to persist changes and free resources.
# - Logs every step (buffer fetch, yfinance fetch, column names, data points added).
# - Falls back to SP500_TICKERS for initial runs.
# - Handles yfinance column quirks (e.g., 'Ticker' vs. 'ticker').
# - Standardized all column names to lowercase for consistency.
# - Added retry for yfinance.download (up to 3 attempts on timeout/errors).
# - At end, updates metadata to MIN(MAX(date) per ticker) for frontend consistency.

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import aiohttp
import asyncio
from typing import List
from dotenv import load_dotenv
import pytz
import time  # For rate limiting and retries
import gc  # For manual garbage collection

# Import S&P 500 tickers for initial population or fallback
from sp500_tickers import SP500_TICKERS as TICKERS

# Configure logging to track progress and debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to SQLite database (persistent on Fly.io volume)
DB_PATH = "/data/stocks.db"

# Buffer days for historical data (covers 14-day ADX/DMI/Stochastic + smoothing)
BUFFER_DAYS = 30

# Batch size for processing tickers (small to avoid OOM on Fly.io free tier)
BATCH_SIZE = 10

# Number of retries for yfinance.download on errors (e.g., timeout)
YF_RETRY_COUNT = 3
YF_RETRY_SLEEP = 5  # seconds

def init_db():
    """
    Initializes the SQLite database with two tables:
    - 'ohlcv': Stores stock data (ticker, date, OHLCV, company name, indicators).
    - 'metadata': Stores key-value pairs (e.g., last_ohlcv_update date).
    Called on app startup if DB or tables are missing.
    """
    logger.info("Initializing database")
    try:
        # Connect to database (creates file if not exists)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create ohlcv table with PRIMARY KEY to prevent duplicate ticker-date entries
        cursor.execute('''
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
        ''')
        
        # Create metadata table for tracking overall last update (min max date across tickers)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                last_update TEXT
            )
        ''')
        
        # Commit changes to save table creation
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for a DataFrame:
    - ADX (Average Directional Index, 14-day)
    - +DI (Plus Directional Indicator, 14-day)
    - -DI (Minus Directional Indicator, 14-day)
    - Stochastic %K (14-day) and %D (3-day SMA of %K)
    Uses pandas rolling operations on small DFs (~40 rows/ticker).
    Converts outputs to float32 for memory efficiency.
    Expects lowercase columns: 'open', 'high', 'low', 'close', 'volume'.
    """
    logger.info(f"Calculating technical indicators for DF with {len(df)} rows")
    try:
        # Extract price series (lowercase columns)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Define period for indicators
        period = 14
        
        # Calculate True Range (TR): max of high-low, |high-prev_close|, |low-prev_close|
        delta_high = high.diff()
        delta_low = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Directional Movement (+DM, -DM)
        plus_dm = delta_high.where(delta_high > 0, 0)
        minus_dm = abs(delta_low.where(delta_low > 0, 0))
        
        # Calculate +DI and -DI (directional indicators)
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # Calculate DX and ADX (trend strength)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        # Calculate Stochastic %K and %D
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        
        # Assign indicators to DF, using float32 to save memory
        df['adx'] = adx.astype('float32') if adx.notnull().any() else np.nan
        df['pdi'] = plus_di.astype('float32') if plus_di.notnull().any() else np.nan
        df['mdi'] = minus_di.astype('float32') if minus_di.notnull().any() else np.nan
        df['k'] = k.astype('float32') if k.notnull().any() else np.nan
        df['d'] = d.astype('float32') if d.notnull().any() else np.nan
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise

def get_tracked_tickers():
    """
    Retrieves unique tickers from the ohlcv table.
    If none exist (e.g., initial run), returns the TICKERS list from sp500_tickers.py.
    """
    logger.info("Fetching tracked tickers from database")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ohlcv")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Fetched {len(tickers)} tracked tickers from database")
        if not tickers:
            tickers = TICKERS
            logger.info(f"No tickers in DB, using fallback list with {len(tickers)} tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tracked tickers: {e}")
        return TICKERS

def get_last_date_for_ticker(conn, ticker):
    """
    Queries the MAX(date) for a specific ticker from ohlcv table.
    Returns the date as str, or None if no data.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()[0]
    return result if result else None

def update_data():
    """
    Updates the ohlcv table with new stock data, checking last date per ticker.
    - Fetches tickers (471 per logs, from DB or TICKERS).
    - For each batch (10 tickers):
      - Queries last date per ticker in batch.
      - Per ticker: Sets api_start_date = last_date +1 or full year if none.
      - If api_start_date >= end_date, skips (no new data needed).
      - Fetches buffer (from last_date - BUFFER_DAYS) from DB.
      - Fetches new data from yfinance with retry on errors.
      - Flattens columns if MultiIndex.
      - Concatenates, sorts, calculates indicators, inserts new rows.
    - Commits after each batch to persist changes and free memory.
    - At end, updates metadata to MIN(MAX(date) per ticker) for frontend.
    - Skips non-trading days (yfinance returns empty for weekends/holidays).
    - Uses float32 for memory efficiency.
    """
    logger.info("Updating database with new stock data")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get tickers (471 per logs)
        tickers = get_tracked_tickers()
        if not tickers:
            raise ValueError("No tickers available for update")
        
        # End date for all fetches (tomorrow to include today, since end is exclusive)
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Track overall inserted rows and min max date for metadata
        inserted_rows = 0
        all_max_dates = []
        
        # Process tickers in batches
        for i in range(0, len(tickers), BATCH_SIZE):
            batch_tickers = tickers[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch_tickers)} tickers ({', '.join(batch_tickers)})")
            
            batch_inserted = 0
            batch_max_dates = []
            
            for ticker in batch_tickers:
                logger.info(f"Processing ticker {ticker} in batch")
                try:
                    # Get last date for this ticker
                    last_date_str = get_last_date_for_ticker(conn, ticker)
                    if last_date_str:
                        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                        api_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                        db_start_date = (last_date - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')
                        logger.info(f"Incremental for {ticker}: DB buffer from {db_start_date}, API from {api_start_date} to {end_date}")
                    else:
                        api_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                        db_start_date = None
                        logger.info(f"Initial full fetch for {ticker} from {api_start_date} to {end_date}")
                    
                    # Skip if no new data possible (start >= end)
                    if api_start_date >= end_date:
                        logger.info(f"No new data needed for {ticker} (up to date as of {last_date_str})")
                        batch_max_dates.append(last_date_str or current_date)
                        time.sleep(0.5)
                        continue
                    
                    # Step 1: Fetch buffer data from DB
                    buffer_df = pd.DataFrame()
                    if db_start_date:
                        logger.info(f"Fetching buffer data from DB for {ticker} since {db_start_date}")
                        query = """
                            SELECT date, open, high, low, close, volume, company_name
                            FROM ohlcv
                            WHERE ticker = ? AND date >= ?
                            ORDER BY date
                        """
                        buffer_df = pd.read_sql_query(query, conn, params=(ticker, db_start_date))
                        buffer_df = buffer_df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
                        buffer_df['ticker'] = ticker  # Add ticker column
                        logger.info(f"Fetched buffer DF with {len(buffer_df)} rows for {ticker}")
                    
                    # Step 2: Fetch new data from yfinance with retry
                    new_data = None
                    for attempt in range(1, YF_RETRY_COUNT + 1):
                        try:
                            logger.info(f"Fetching new data from yfinance for {ticker} (attempt {attempt})")
                            new_data = yf.download(ticker, start=api_start_date, end=end_date, auto_adjust=False, progress=False)
                            if not new_data.empty:
                                break
                        except Exception as e:
                            logger.warning(f"yfinance error for {ticker} on attempt {attempt}: {e}")
                            if attempt < YF_RETRY_COUNT:
                                time.sleep(YF_RETRY_SLEEP)
                            else:
                                logger.error(f"Failed to fetch data for {ticker} after {YF_RETRY_COUNT} attempts")
                                break
                    
                    if new_data is None or new_data.empty:
                        logger.warning(f"No new data for {ticker}, skipping")
                        batch_max_dates.append(last_date_str or current_date)
                        time.sleep(0.5)
                        continue
                    
                    # Flatten columns if MultiIndex
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)
                        logger.info(f"Flattened MultiIndex columns for {ticker}")
                    
                    # Handle DF structure (single ticker)
                    new_data = new_data.reset_index()
                    new_data['ticker'] = ticker
                    available_columns = set(new_data.columns)
                    logger.info(f"yfinance returned columns for {ticker}: {list(available_columns)}")
                    expected_columns = ['Date', 'ticker', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
                    if set(expected_columns).issubset(available_columns):
                        new_data = new_data[expected_columns]
                        new_data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                        new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')
                        new_data = new_data.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
                    else:
                        logger.error(f"Missing expected columns for {ticker}. Required: {expected_columns}, Actual: {list(available_columns)}; skipping")
                        continue
                    
                    # Add company name
                    try:
                        company_name = yf.Ticker(ticker).info.get('longName', ticker)
                    except:
                        company_name = ticker
                    new_data['company_name'] = company_name
                    logger.info(f"Fetched new DF with {len(new_data)} rows for {ticker}")
                    
                    # Step 3: Concat buffer and new data
                    if not buffer_df.empty:
                        buffer_df['company_name'] = company_name
                        combined_df = pd.concat([buffer_df, new_data], ignore_index=True).drop_duplicates(subset=['date'])
                    else:
                        combined_df = new_data
                    combined_df = combined_df.sort_values('date')
                    logger.info(f"Combined DF with {len(combined_df)} rows for {ticker}")
                    
                    # Step 4: Calculate indicators
                    min_date = combined_df['date'].min()
                    max_date = combined_df['date'].max()
                    logger.info(f"Calculating indicators for {ticker} on dates from {min_date} to {max_date}")
                    combined_df = calculate_indicators(combined_df)
                    
                    # Step 5: Insert new rows
                    count = 0
                    ticker_max_date = last_date_str or '1900-01-01'
                    for _, row in combined_df.iterrows():
                        if row['date'] > ticker_max_date:
                            cursor.execute('''
                                INSERT OR REPLACE INTO ohlcv (
                                    ticker, date, open, high, low, close, volume, 
                                    company_name, adx, pdi, mdi, k, d
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                ticker,
                                row['date'],
                                row['open'],
                                row['high'],
                                row['low'],
                                row['close'],
                                int(row['volume']) if pd.notna(row['volume']) else None,
                                row['company_name'],
                                row['adx'] if pd.notna(row['adx']) else None,
                                row['pdi'] if pd.notna(row['pdi']) else None,
                                row['mdi'] if pd.notna(row['mdi']) else None,
                                row['k'] if pd.notna(row['k']) else None,
                                row['d'] if pd.notna(row['d']) else None
                            ))
                            count += 1
                            logger.info(f"Added data point for {ticker} on {row['date']}")
                            if row['date'] > ticker_max_date:
                                ticker_max_date = row['date']
                    if count > 0:
                        logger.info(f"Added {count} new data points for {ticker}")
                    inserted_rows += count
                    batch_max_dates.append(ticker_max_date)
                    
                    # Free memory per ticker
                    del combined_df
                    gc.collect()
                    
                    time.sleep(0.5)  # Short sleep to avoid rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing ticker {ticker}: {e}")
                    continue
            
            # Commit batch changes
            conn.commit()
            logger.info(f"Committed changes for batch, total inserted so far: {inserted_rows}")
        
        # Update metadata to MIN of all tickers' MAX(date) for frontend (overall "up to" date)
        if inserted_rows > 0:
            logger.info("Calculating min max date across all tickers for metadata")
            cursor.execute("""
                SELECT MIN(last_date) FROM (SELECT MAX(date) as last_date FROM ohlcv GROUP BY ticker)
            """)
            min_max_date = cursor.fetchone()[0]
            if min_max_date:
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata (key, last_update)
                    VALUES ('last_ohlcv_update', ?)
                ''', (min_max_date,))
                conn.commit()
                logger.info(f"Updated metadata with last_ohlcv_update: {min_max_date}")
            else:
                logger.warning("No min max date found; metadata not updated")
        
        logger.info(f"Total inserted or replaced {inserted_rows} rows into ohlcv table")
        logger.info("Database updated successfully")
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def rebuild_database():
    """
    Rebuilds the database from scratch:
    - Deletes existing DB file.
    - Initializes tables.
    - Runs update_data() for a full year of data.
    """
    logger.info("Rebuilding database")
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info("Existing database removed")
        init_db()
        update_data()
        logger.info("Database rebuilt successfully")
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        raise

# Unchanged from original; kept for completeness (used by /live-prices endpoint)
async def fetch_live_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches live prices using Alpaca API (async for efficiency).
    Not used in update_data, but included for endpoint compatibility.
    """
    logger.info(f"Fetching live prices for {len(tickers)} tickers: {tickers}")
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        logger.info(f"API Key present: {bool(api_key)}, Secret Key present: {bool(secret_key)}")
        if not api_key or not secret_key:
            logger.error("Alpaca API credentials missing")
            return pd.DataFrame([{"ticker": t, "price": None, "timestamp": None, "volume": None} for t in tickers])
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        base_url = "https://data.alpaca.markets/v2"
        
        async def fetch_batch(batch: List[str], attempt: int = 1) -> pd.DataFrame:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{base_url}/stocks/quotes/latest?symbols={','.join(batch)}"
                    logger.info(f"Attempt {attempt} - Sending request to: {url}")
                    async with session.get(url, headers=headers) as response:
                        logger.info(f"Attempt {attempt} - Response status: {response.status}")
                        if response.status != 200:
                            text = await response.text()
                            logger.warning(f"Attempt {attempt} - Alpaca API error for batch {batch}: {response.status} - {text}")
                            if attempt < 2:
                                logger.info(f"Retrying batch {batch}")
                                await asyncio.sleep(0.1)
                                return await fetch_batch(batch, attempt + 1)
                            return pd.DataFrame([{"ticker": t, "price": None, "timestamp": None, "volume": None} for t in batch])
                        data = await response.json()
                        logger.info(f"Attempt {attempt} - Response data: {data}")
                        quotes = data.get("quotes", {})
                        results = []
                        est_tz = pytz.timezone('America/New_York')
                        for ticker in batch:
                            quote = quotes.get(ticker, {})
                            timestamp = quote.get("t")
                            volume = quote.get("v") if quote.get("v") is not None else None
                            price = quote.get("ap") if quote.get("ap") is not None else None
                            if price == 0 or price is None:
                                logger.warning(f"Invalid price for {ticker}: {price}")
                                if attempt < 2:
                                    logger.info(f"Retrying {ticker} due to invalid price")
                                    await asyncio.sleep(0.1)
                                    retry_result = await fetch_batch([ticker], attempt + 1)
                                    if not retry_result.empty and retry_result.iloc[0]["price"] is not None:
                                        results.append(retry_result.iloc[0].to_dict())
                                        continue
                            if timestamp:
                                try:
                                    timestamp = timestamp[:26] + 'Z' if timestamp.endswith('Z') else timestamp[:26]
                                    utc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    est_dt = utc_dt.astimezone(est_tz)
                                    timestamp = est_dt.strftime('%Y-%m-%d %H:%M:%S')
                                    logger.info(f"Converted timestamp for {ticker}: UTC {utc_dt} to EST {timestamp}")
                                except ValueError as e:
                                    logger.warning(f"Invalid timestamp format for {ticker}: {timestamp} - {e}")
                                    timestamp = None
                            results.append({
                                "ticker": ticker,
                                "price": price,
                                "timestamp": timestamp,
                                "volume": volume
                            })
                        return pd.DataFrame(results)
            except Exception as e:
                logger.error(f"Attempt {attempt} - Error fetching batch {batch}: {e}")
                if attempt < 2:
                    logger.info(f"Retrying batch {batch}")
                    await asyncio.sleep(0.1)
                    return await fetch_batch(batch, attempt + 1)
                return pd.DataFrame([{"ticker": t, "price": None, "timestamp": None, "volume": None} for t in batch])
        
        batch_size = 100
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        tasks = [fetch_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = pd.concat([r for r in results if isinstance(r, pd.DataFrame)], ignore_index=True)
        logger.info(f"Returning live prices for {len(final_results)} tickers")
        return final_results
    except Exception as e:
        logger.error(f"Error fetching live prices: {e}")
        return pd.DataFrame([{"ticker": t, "price": None, "timestamp": None, "volume": None} for t in tickers])