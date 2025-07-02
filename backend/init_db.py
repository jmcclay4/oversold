
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
        if not tickers:
            logger.warning("No tickers found in ohlcv table")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tracked tickers: {e}")
        return []

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)
        logger.info(f"Raw yfinance data shape: {data.shape}, columns: {list(data.columns)}")
        if data.empty:
            logger.warning("No data returned from yfinance")
            return pd.DataFrame()
        
        if len(tickers) == 1:
            data['ticker'] = tickers[0]
            data = data.reset_index()[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        else:
            # Handle multi-ticker case with robust column checking
            if isinstance(data.columns, pd.MultiIndex):
                logger.info("MultiIndex detected, flattening columns")
                data = data.stack(future_stack=True).reset_index()
                if 'ticker' not in data.columns:
                    data = data.rename(columns={'level_1': 'ticker'})
                expected_columns = ['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in expected_columns if col not in data.columns]
                if missing_cols:
                    logger.error(f"Missing columns in yfinance data: {missing_cols}")
                    return pd.DataFrame()
                data = data[expected_columns]
                data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            else:
                logger.warning("Unexpected DataFrame structure from yfinance")
                return pd.DataFrame()
        
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        data['company_name'] = None
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                data.loc[data['ticker'] == ticker, 'company_name'] = ticker_obj.info.get('longName', None)
            except Exception as e:
                logger.warning(f"Could not fetch company name for {ticker}: {e}")
                data.loc[data['ticker'] == ticker, 'company_name'] = None
        logger.info(f"Processed data shape: {data.shape}, latest date: {data['date'].max() if not data.empty else 'N/A'}")
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

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

def update_data():
    logger.info("Updating database with new stock data")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        tickers = get_tracked_tickers()
        if not tickers:
            logger.error("No tickers found in database, aborting update")
            raise ValueError("No tickers found")
        
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # Include today
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        logger.info(f"Fetching data from {start_date} to {end_date} for {len(tickers)} tickers")
        
        data = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            logger.error("No stock data fetched, aborting update")
            raise ValueError("No stock data fetched")
        
        logger.info(f"Fetched {len(data)} rows, latest date: {data['date'].max()}")
        grouped = data.groupby('ticker')
        inserted_rows = 0
        for ticker, group in grouped:
            try:
                group = calculate_indicators(group)
                for _, row in group.iterrows():
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
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        logger.info(f"Inserted or replaced {inserted_rows} rows into ohlcv table")
        
        latest_date = data['date'].max() if not data.empty else datetime.now().strftime('%Y-%m-%d')
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
