# init_db.py
# This file manages the SQLite database for stock data, fetching OHLCV from yfinance,
# calculating technical indicators (ADX, +DI, -DI, Stochastic %K/%D), and storing results.
# Key features:
# - Initializes tables for OHLCV and metadata.
# - Updates data incrementally, assuming a uniform last update date across tickers.
# - Processes tickers in batches (BATCH_SIZE=10) to minimize memory usage on Fly.io (256MB limit).
# - Fetches buffer data (30 days) from DB and new data from yfinance (using Adj Close).
# - Concatenates buffer + new data, sorts by ticker/date.
# - Groups by ticker, calculates indicators, inserts new rows.
# - Commits after each batch to persist changes and free resources.
# - Logs every step (buffer fetch, yfinance fetch, column names, data points added).
# - Falls back to SP500_TICKERS for initial runs.
# - Handles yfinance column quirks (e.g., 'Ticker' vs. 'ticker').
# - Standardized all column names to lowercase for consistency.

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
import time  # For rate limiting
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
        
        # Create metadata table for tracking last update date
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

def update_data():
    """
    Updates the ohlcv table with new stock data since the last update (from metadata).
    - Fetches tickers (471 per logs, from DB or TICKERS).
    - Gets last update date from metadata (assumes uniform across tickers).
    - For each batch (10 tickers):
      - Fetches buffer data (30 days) from DB for indicator calculations.
      - Fetches new data from yfinance (from last_update+1 to today).
      - Concatenates buffer + new data, sorts by ticker/date.
      - Groups by ticker, calculates indicators, inserts new rows.
    - Commits after each batch to persist changes and free memory.
    - Updates metadata with the latest date inserted.
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
        
        # Get last update date from metadata
        cursor.execute("SELECT last_update FROM metadata WHERE key = 'last_ohlcv_update'")
        result = cursor.fetchone()
        last_update = result[0] if result else None
        
        # Set date range
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # Up to today
        if last_update:
            initial_last_date_str = last_update.split()[0] if ' ' in last_update else last_update
            last_date = datetime.strptime(initial_last_date_str, '%Y-%m-%d')
            api_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            db_start_date = (last_date - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')
            logger.info(f"Incremental update: DB buffer from {db_start_date}, API from {api_start_date} to {end_date}")
        else:
            api_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            db_start_date = None
            initial_last_date_str = '1900-01-01'
            logger.info(f"Initial full fetch from {api_start_date} to {end_date}")
        
        # Track latest date for metadata update
        new_latest_date = initial_last_date_str
        inserted_rows = 0
        
        # Process tickers in batches
        for i in range(0, len(tickers), BATCH_SIZE):
            batch_tickers = tickers[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch_tickers)} tickers ({', '.join(batch_tickers)})")
            
            # Step 1: Fetch buffer data from DB (if not initial)
            buffer_df = pd.DataFrame()
            if db_start_date:
                logger.info(f"Fetching buffer data from DB for batch since {db_start_date}")
                placeholders = ','.join(['?'] * len(batch_tickers))
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume, company_name
                    FROM ohlcv
                    WHERE ticker IN ({placeholders}) AND date >= ?
                    ORDER BY ticker, date
                """
                buffer_df = pd.read_sql_query(query, conn, params=[*batch_tickers, db_start_date])
                buffer_df = buffer_df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
                logger.info(f"Fetched buffer DF with {len(buffer_df)} rows for batch")
            
            # Step 2: Fetch new data from yfinance
            logger.info(f"Fetching new data from yfinance for batch from {api_start_date} to {end_date}")
            new_data = yf.download(batch_tickers, start=api_start_date, end=end_date, auto_adjust=False, progress=False)
            if new_data.empty:
                logger.warning(f"No new data returned from yfinance for batch {batch_tickers}")
                continue
            
            # Handle DataFrame structure
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data = new_data.stack(future_stack=True).reset_index()
                new_data = new_data.rename(columns={'level_1': 'ticker', 'Ticker': 'ticker'})  # Handle capitalization
                logger.info("Handled MultiIndex DF from yfinance")
            else:
                if len(batch_tickers) == 1:
                    new_data = new_data.reset_index()
                    new_data['ticker'] = batch_tickers[0]
                    logger.info(f"Handled single-ticker DF for {batch_tickers[0]}")
                else:
                    logger.error(f"Unexpected non-MultiIndex DF for batch {batch_tickers}; skipping")
                    continue
            
            # Select and rename columns to lowercase, using 'Adj Close' as 'close'
            expected_columns = ['Date', 'ticker', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
            available_columns = set(new_data.columns)
            logger.info(f"yfinance returned columns: {list(available_columns)}")
            if set(expected_columns).issubset(available_columns):
                new_data = new_data[expected_columns]
                new_data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')
                new_data = new_data.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
            else:
                logger.error(f"Missing expected columns in new_data for batch {batch_tickers}. Required: {expected_columns}, Actual: {list(available_columns)}; skipping")
                continue
            
            # Add company names
            company_names = {}
            for t in batch_tickers:
                try:
                    company_names[t] = yf.Ticker(t).info.get('longName', t)
                    time.sleep(0.1)
                except:
                    company_names[t] = t
            new_data['company_name'] = new_data['ticker'].map(company_names)
            logger.info(f"Fetched new DF with {len(new_data)} rows for batch")
            
            # Step 3: Concat buffer and new data
            if not buffer_df.empty:
                buffer_df['company_name'] = buffer_df['ticker'].map(company_names)
                combined_df = pd.concat([buffer_df, new_data], ignore_index=True).drop_duplicates(subset=['ticker', 'date'])
            else:
                combined_df = new_data
            combined_df = combined_df.sort_values(['ticker', 'date'])
            logger.info(f"Combined DF with {len(combined_df)} rows for batch")
            
            # Free memory
            del buffer_df, new_data
            gc.collect()
            
            # Step 4: Process each ticker in the batch
            grouped = combined_df.groupby('ticker')
            for ticker, group in grouped:
                logger.info(f"Processing ticker {ticker} in batch")
                try:
                    min_date = group['date'].min()
                    max_date = group['date'].max()
                    logger.info(f"Calculating indicators for {ticker} on dates from {min_date} to {max_date}")
                    group = calculate_indicators(group)  # No rename needed; function uses lowercase
                    
                    # Insert new rows
                    count = 0
                    for _, row in group.iterrows():
                        if row['date'] > initial_last_date_str:
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
                            if row['date'] > new_latest_date:
                                new_latest_date = row['date']
                    if count > 0:
                        logger.info(f"Added {count} new data points for {ticker}")
                    inserted_rows += count
                except Exception as e:
                    logger.error(f"Error processing ticker {ticker}: {e}")
                    continue
            
            # Commit batch changes
            conn.commit()
            logger.info(f"Committed changes for batch, total inserted so far: {inserted_rows}")
            del combined_df, grouped
            gc.collect()
        
        # Update metadata
        if new_latest_date > initial_last_date_str:
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, last_update)
                VALUES ('last_ohlcv_update', ?)
            ''', (new_latest_date,))
            conn.commit()
            logger.info(f"Updated metadata with last_ohlcv_update: {new_latest_date}")
        else:
            logger.info("No new data inserted, metadata unchanged")
        
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