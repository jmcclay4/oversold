# init_db.py
# This file contains functions to initialize, update, and rebuild the SQLite database for stock data.
# It fetches OHLCV data from yfinance, calculates technical indicators (ADX, +DI, -DI, Stochastic %K/%D),
# and stores them in the database.
# Key changes for optimization:
# - Assumes uniform last update date across all tickers (from metadata table).
# - Fetches new data in a single batch call to yfinance for efficiency.
# - Queries buffer historical data (last 30 days) in a single DB query.
# - Concatenates buffer + new data into one DataFrame, groups by ticker, and processes per group.
# - Uses astype(float32) to reduce memory usage.
# - Calculates indicators using pandas rolling (small DFs per ticker ~40 rows, low memory).
# - Inserts only new rows, commits after each ticker to release resources early.
# - Added garbage collection and logging for monitoring.
# - If no tickers in DB (initial run), uses SP500_TICKERS list.

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
import time  # For rate limiting if needed
import gc  # For manual garbage collection

# Import the list of S&P 500 tickers for initial population
from sp500_tickers import SP500_TICKERS as TICKERS

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the SQLite database file (persistent volume on Fly.io)
DB_PATH = "/data/stocks.db"

# Buffer days for historical data needed to calculate indicators (14 for ADX/DMI/Stoch + extra for smoothing)
BUFFER_DAYS = 30

def init_db():
    """
    Initializes the database by creating the necessary tables if they don't exist.
    - 'ohlcv' table: Stores stock data with OHLCV, company name, and calculated indicators.
    - 'metadata' table: Stores key-value pairs, like the last update date for OHLCV data.
    This function is called on app startup if the DB or tables are missing.
    """
    logger.info("Initializing database")
    try:
        # Connect to the database (creates file if not exists)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create the ohlcv table with columns for stock data and indicators
        # PRIMARY KEY ensures no duplicates per ticker-date pair
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
        
        # Create the metadata table for storing last update date, etc.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                last_update TEXT
            )
        ''')
        
        # Commit changes and log success
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        # Log any errors during initialization
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        # Always close the connection
        conn.close()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators on the DataFrame:
    - ADX (Average Directional Index), +DI (Plus Directional Indicator), -DI (Minus Directional Indicator)
    - Stochastic Oscillator %K and %D
    Uses pandas rolling operations. To optimize memory, assumes small DF (~40 rows per ticker).
    """
    logger.info(f"Calculating technical indicators for DF with {len(df)} rows")
    try:
        # Extract necessary series
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Parameters for indicators
        period = 14  # Standard period for DMI/ADX/Stochastic
        
        # Calculate True Range (TR)
        delta_high = high.diff()
        delta_low = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DM and -DM
        plus_dm = delta_high.where(delta_high > 0, 0)
        minus_dm = abs(delta_low.where(delta_low > 0, 0))
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        # Calculate Stochastic %K and %D
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        
        # Add indicators to DF
        df['adx'] = adx.astype('float32') if adx.notnull().any() else None
        df['pdi'] = plus_di.astype('float32') if plus_di.notnull().any() else None
        df['mdi'] = minus_di.astype('float32') if minus_di.notnull().any() else None
        df['k'] = k.astype('float32') if k.notnull().any() else None
        df['d'] = d.astype('float32') if d.notnull().any() else None
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise

def get_tracked_tickers():
    """
    Fetches the list of unique tickers from the ohlcv table in the database.
    If no tickers (initial run), falls back to SP500_TICKERS list.
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
            tickers = TICKERS  # Fallback to static list for initial population
            logger.info(f"No tickers in DB, using fallback list with {len(tickers)} tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tracked tickers: {e}")
        return TICKERS  # Fallback on error

def update_data():
    """
    Updates the database with new stock data since the last update.
    - Assumes uniform last update date across tickers (from metadata).
    - Fetches buffer data (last 30 days) from DB in one query.
    - Fetches new data from yfinance in one batch call.
    - Concatenates, groups by ticker, calculates indicators per group.
    - Inserts only new rows per ticker, committing after each to save memory.
    - Uses float32 for reduced memory.
    - Backfills missing trading days (yfinance skips weekends/holidays).
    """
    logger.info("Updating database with new stock data")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of tickers
        tickers = get_tracked_tickers()
        if not tickers:
            raise ValueError("No tickers available for update")
        
        # Get the last uniform update date from metadata
        cursor.execute("SELECT last_update FROM metadata WHERE key = 'last_ohlcv_update'")
        result = cursor.fetchone()
        last_update = result[0] if result else None
        
        # Determine dates for fetch
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # Up to today
        if last_update:
            # Strip time if present (from previous bug)
            last_update_date_str = last_update.split()[0] if ' ' in last_update else last_update
            last_date = datetime.strptime(last_update_date_str, '%Y-%m-%d')
            api_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')  # New data start
            db_start_date = (last_date - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')  # Buffer for indicators
            logger.info(f"Incremental update: DB buffer from {db_start_date}, API from {api_start_date} to {end_date}")
        else:
            # Initial full year fetch
            api_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            db_start_date = None
            logger.info(f"Initial full fetch from {api_start_date} to {end_date}")
        
        # Step 1: Fetch buffer historical data from DB (if not initial)
        buffer_df = pd.DataFrame()
        if db_start_date:
            logger.info(f"Fetching buffer data from DB for all {len(tickers)} tickers since {db_start_date}")
            placeholders = ','.join(['?'] * len(tickers))
            query = f"""
                SELECT ticker, date, open, high, low, close, volume, company_name
                FROM ohlcv
                WHERE ticker IN ({placeholders}) AND date >= ?
                ORDER BY ticker, date
            """
            buffer_df = pd.read_sql_query(query, conn, params=[*tickers, db_start_date])
            buffer_df = buffer_df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
            logger.info(f"Fetched buffer DF with {len(buffer_df)} rows")
        
        # Step 2: Fetch new data from yfinance in batch
        logger.info(f"Fetching new data from yfinance for {len(tickers)} tickers from {api_start_date} to {end_date}")
        new_data = yf.download(tickers, start=api_start_date, end=end_date, auto_adjust=True, progress=False)
        if new_data.empty:
            logger.warning("No new data returned from yfinance")
            return
        
        # Handle multi-index if multi-ticker
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data = new_data.stack(future_stack=True).reset_index()
            new_data = new_data.rename(columns={'level_1': 'ticker'})
        else:
            # Single ticker case
            new_data['ticker'] = tickers[0]
        
        new_data = new_data[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        new_data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')
        new_data = new_data.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'int32'})
        
        # Add company_name (fetch once per ticker, but batch via Ticker)
        company_names = {}
        for ticker in tickers:
            try:
                company_names[ticker] = yf.Ticker(ticker).info.get('longName', ticker)
                time.sleep(0.1)  # Light rate limit for info calls
            except:
                company_names[ticker] = ticker
        new_data['company_name'] = new_data['ticker'].map(company_names)
        
        logger.info(f"Fetched new DF with {len(new_data)} rows")
        
        # Step 3: Concat buffer and new data
        if not buffer_df.empty:
            buffer_df['company_name'] = buffer_df['ticker'].map(company_names)  # Align company names
            combined_df = pd.concat([buffer_df, new_data], ignore_index=True).drop_duplicates(subset=['ticker', 'date'])
        else:
            combined_df = new_data
        combined_df = combined_df.sort_values(['ticker', 'date'])
        logger.info(f"Combined DF with {len(combined_df)} rows")
        
        # Free unused DFs
        del buffer_df, new_data
        gc.collect()
        
        # Step 4: Group by ticker and process each group
        inserted_rows = 0
        latest_date = last_update_date_str if last_update else '1900-01-01'
        grouped = combined_df.groupby('ticker')
        
        for ticker, group in grouped:
            logger.info(f"Processing ticker: {ticker}")
            try:
                # Calculate indicators on the group DF
                min_date = group['date'].min()
                max_date = group['date'].max()
                logger.info(f"Calculating indicators for {ticker} on dates from {min_date} to {max_date}")
                group = calculate_indicators(group.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}))
                
                # Insert only new rows (date > global last_update, assuming uniform)
                count = 0
                for _, row in group.iterrows():
                    if row['date'] > latest_date:
                        cursor.execute('''
                            INSERT OR REPLACE INTO ohlcv (
                                ticker, date, open, high, low, close, volume, 
                                company_name, adx, pdi, mdi, k, d
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker,
                            row['date'],
                            row['Open'],
                            row['High'],
                            row['Low'],
                            row['Close'],
                            int(row['Volume']) if pd.notna(row['Volume']) else None,
                            row['company_name'],
                            row['adx'] if pd.notna(row['adx']) else None,
                            row['pdi'] if pd.notna(row['pdi']) else None,
                            row['mdi'] if pd.notna(row['mdi']) else None,
                            row['k'] if pd.notna(row['k']) else None,
                            row['d'] if pd.notna(row['d']) else None
                        ))
                        inserted_rows += 1
                        count += 1
                        logger.info(f"Added data point for {ticker} on {row['date']}")
                        if row['date'] > latest_date:
                            latest_date = row['date']
                
                if count > 0:
                    logger.info(f"Added {count} new data points for {ticker}")
                
                # Commit after each ticker to save changes and release locks/memory
                conn.commit()
                
                # Garbage collect
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        # Update metadata with the new latest date
        logger.info(f"Inserted or replaced {inserted_rows} rows into ohlcv table")
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, last_update)
            VALUES ('last_ohlcv_update', ?)
        ''', (latest_date,))
        conn.commit()
        logger.info(f"Updated metadata with last_ohlcv_update: {latest_date}")
        
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
    - Deletes existing DB file if present.
    - Initializes tables.
    - Runs full update_data() to populate with 1 year of data.
    Useful for resetting or initial setup.
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

# The fetch_live_prices function remains unchanged, as it's for live data (Alpaca API) and not part of the update process.
async def fetch_live_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches live prices using Alpaca API (async for efficiency).
    Not used in update_data, but kept for completeness.
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
                                await asyncio.sleep(0.1)  # 0.1-second delay
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
                                    await asyncio.sleep(0.1)  # 0.1-second delay
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
                    await asyncio.sleep(0.1)  # 0.1-second delay
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