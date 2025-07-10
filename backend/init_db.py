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

try:
    from sp500_tickers import TICKERS
except ImportError:
    TICKERS = []  # Replace with your list of S&P 500 tickers if the import fails

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/data/stocks.db"

def init_db():
    logger.info("Initializing database")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                last_update TEXT
            )
        ''')
        
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating technical indicators")
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        period = 14
        delta_high = high.diff()
        delta_low = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_dm = delta_high.where(delta_high > 0, 0)
        minus_dm = abs(delta_low.where(delta_low > 0, 0))
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        
        df['adx'] = adx
        df['pdi'] = plus_di
        df['mdi'] = minus_di
        df['k'] = k
        df['d'] = d
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise

def get_tracked_tickers():
    logger.info("Fetching tracked tickers from database")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ohlcv")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Fetched {len(tickers)} tracked tickers from database")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tracked tickers: {e}")
        return []

def fetch_historical_from_db(conn, ticker, start_date):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, open, high, low, close, volume
        FROM ohlcv
        WHERE ticker = ? AND date >= ?
        ORDER BY date ASC
    ''', (ticker, start_date))
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

def update_data():
    logger.info("Updating database with new stock data")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        tickers = get_tracked_tickers()
        if not tickers and TICKERS:
            tickers = TICKERS
            logger.info(f"Using fallback TICKERS list with {len(tickers)} tickers for initial update")
        if not tickers:
            logger.error("No tickers found in database or fallback list, aborting update")
            raise ValueError("No tickers found")
        
        cursor.execute("SELECT last_update FROM metadata WHERE key = 'last_ohlcv_update'")
        result = cursor.fetchone()
        last_update = result[0] if result else None
        
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        buffer_days = 30  # Sufficient for ADX (14+14) and Stochastic (14+3), with safety
        
        if last_update:
            last_update_date_str = last_update.split()[0] if ' ' in last_update else last_update
            last_date = datetime.strptime(last_update_date_str, '%Y-%m-%d')
            api_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            db_start_date = (last_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            logger.info(f"Incremental update: DB buffer from {db_start_date}, API from {api_start_date} to {end_date}")
        else:
            api_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            db_start_date = None
            logger.info(f"Initial full fetch from {api_start_date} to {end_date}")
        
        inserted_rows = 0
        latest_date = last_update_date_str if last_update else '1900-01-01'
        
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            try:
                # Fetch historical from DB if incremental
                hist_df = pd.DataFrame()
                if db_start_date:
                    hist_df = fetch_historical_from_db(conn, ticker, db_start_date)
                
                # Fetch new data from API
                new_data = yf.download(ticker, start=api_start_date, end=end_date, auto_adjust=True, progress=False)
                if new_data.empty:
                    logger.warning(f"No new data from yfinance for {ticker}")
                    continue
                
                new_df = new_data.reset_index()
                new_df['ticker'] = ticker
                new_df = new_df[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
                new_df.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
                
                ticker_obj = yf.Ticker(ticker)
                company_name = ticker_obj.info.get('longName', None)
                new_df['company_name'] = company_name
                
                # Combine hist and new if hist exists
                if not hist_df.empty:
                    hist_df = hist_df.reset_index()
                    hist_df['date'] = hist_df['date'].dt.strftime('%Y-%m-%d')
                    hist_df['ticker'] = ticker
                    hist_df['company_name'] = company_name
                    combined_df = pd.concat([hist_df, new_df], ignore_index=True).drop_duplicates(subset=['date'])
                else:
                    combined_df = new_df
                
                min_date = combined_df['date'].min()
                max_date = combined_df['date'].max()
                logger.info(f"Calculating indicators for {ticker} on dates from {min_date} to {max_date}")
                
                combined_df = calculate_indicators(combined_df)
                
                # Insert only new rows
                count = 0
                for _, row in combined_df.iterrows():
                    if row['date'] > latest_date:
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
                        inserted_rows += 1
                        count += 1
                        logger.info(f"Added data point for {ticker} on {row['date']}")
                        if row['date'] > latest_date:
                            latest_date = row['date']
                
                if count > 0:
                    logger.info(f"Added {count} new data points for {ticker}")
                
                # Free memory
                del hist_df, new_df, combined_df
                gc.collect()
                
                time.sleep(1)  # Rate limit
                
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        logger.info(f"Inserted or replaced {inserted_rows} rows into ohlcv table")
        
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, last_update)
            VALUES ('last_ohlcv_update', ?)
        ''', (latest_date,))
        logger.info(f"Updated metadata with last_ohlcv_update: {latest_date}")
        
        conn.commit()
        logger.info("Database updated successfully")
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def rebuild_database():
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

async def fetch_live_prices(tickers: List[str]) -> pd.DataFrame:
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