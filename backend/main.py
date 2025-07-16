from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
from typing import List, Dict
import gc
import os
import numpy as np
import time
import importlib

# Import S&P 500 tickers for initial population or fallback
SP500_TICKERS = []  # Replace with actual list or import from sp500_tickers.py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "/data/stocks.db"

BUFFER_DAYS = 30
BATCH_SIZE = 10
YF_RETRY_COUNT = 3
YF_RETRY_SLEEP = 5  # seconds
HISTORY_MONTHS = 6  # Limit to 6 months of data

def init_db():
    """
    Initializes the SQLite database with two tables:
    - 'ohlcv': Stores stock data (ticker, date, OHLCV, company name, indicators, signals).
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
                dmi_signal INTEGER DEFAULT 0,
                sto_signal INTEGER DEFAULT 0,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        # Check if dmi_signal and sto_signal columns exist, add if missing
        cursor.execute("PRAGMA table_info(ohlcv)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'dmi_signal' not in columns:
            cursor.execute("ALTER TABLE ohlcv ADD COLUMN dmi_signal INTEGER DEFAULT 0")
            logger.info("Added dmi_signal column to ohlcv table")
        if 'sto_signal' not in columns:
            cursor.execute("ALTER TABLE ohlcv ADD COLUMN sto_signal INTEGER DEFAULT 0")
            logger.info("Added sto_signal column to ohlcv table")
        
        # Create metadata table for tracking overall last update
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

def migrate_db():
    """
    Explicitly migrates the database to ensure dmi_signal and sto_signal columns exist.
    """
    logger.info("Running database migration")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check and add dmi_signal and sto_signal columns if missing
        cursor.execute("PRAGMA table_info(ohlcv)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'dmi_signal' not in columns:
            cursor.execute("ALTER TABLE ohlcv ADD COLUMN dmi_signal INTEGER DEFAULT 0")
            logger.info("Added dmi_signal column to ohlcv table")
        if 'sto_signal' not in columns:
            cursor.execute("ALTER TABLE ohlcv ADD COLUMN sto_signal INTEGER DEFAULT 0")
            logger.info("Added sto_signal column to ohlcv table")
        
        conn.commit()
        logger.info("Database migration completed successfully")
    except Exception as e:
        logger.error(f"Error during database migration: {e}")
        raise
    finally:
        conn.close()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for a DataFrame:
    - ADX (Average Directional Index, 9-day)
    - +DI (Plus Directional Indicator, 9-day)
    - -DI (Minus Directional Indicator, 9-day)
    - Stochastic %K (9-day, slow with 3,3 smoothing) and %D (3-day SMA)
    Uses Wilder's smoothing for DMI/ADX, simple moving average for stochastic.
    Converts outputs to float32 for memory efficiency.
    Expects lowercase columns: 'open', 'high', 'low', 'close', 'volume'.
    """
    logger.info(f"Calculating technical indicators for DF with {len(df)} rows")
    try:
        # Extract price series (lowercase columns)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Periods
        period_dmi_adx = 9
        period_stoch = 9
        period_slow = 3
        
        # DMI/ADX (9,9) with Wilder's smoothing
        delta_high = high.diff()
        delta_low = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        
        # Calculate +DM and -DM with mutual exclusivity
        plus_dm = delta_high.where((delta_high > 0) & (delta_high > delta_low.abs()), 0)
        minus_dm = delta_low.abs().where((delta_low > 0) & (delta_low.abs() > delta_high), 0)
        
        # Wilder's smoothing for +DM, -DM, and TR
        def wilder_smooth(series, period):
            smoothed = series.copy()
            smoothed[:period] = series[:period].mean()  # First value is simple mean
            for i in range(period, len(series)):
                smoothed.iloc[i] = (smoothed.iloc[i-1] * (period - 1) + series.iloc[i]) / period
            return smoothed
        
        smoothed_plus_dm = wilder_smooth(plus_dm, period_dmi_adx)
        smoothed_minus_dm = wilder_smooth(minus_dm, period_dmi_adx)
        smoothed_tr = wilder_smooth(tr, period_dmi_adx)
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period_dmi_adx)
        
        # Stochastic (9,3,3) - Slow Stochastic
        lowest_low = low.rolling(window=period_stoch).min()
        highest_high = high.rolling(window=period_stoch).max()
        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_slow = k_fast.rolling(window=period_slow).mean()  # Slow %K = SMA(Fast %K, 3)
        d = k_slow.rolling(window=period_slow).mean()  # %D = SMA(Slow %K, 3)
        
        # Assign indicators to DF, using float32 to save memory
        df['adx'] = adx.astype('float32') if adx.notnull().any() else np.nan
        df['pdi'] = plus_di.astype('float32') if plus_di.notnull().any() else np.nan
        df['mdi'] = minus_di.astype('float32') if minus_di.notnull().any() else np.nan
        df['k'] = k_slow.astype('float32') if k_slow.notnull().any() else np.nan  # Slow %K
        df['d'] = d.astype('float32') if d.notnull().any() else np.nan
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise

def compute_signals(df: pd.DataFrame) -> tuple[int, int]:
    """
    Computes DMI and Stochastic signals for the latest row based on last 3 days.
    Returns (dmi_signal, sto_signal) as 1/0.
    Assumes df is sorted by date, with at least 3 rows for full check.
    """
    if len(df) < 3:
        return 0, 0  # Insufficient data
    
    # Latest (day 0), previous (day -1), prev_prev (day -2)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev_prev = df.iloc[-3]
    
    # DMI Signal
    pdi = latest['pdi'] if pd.notna(latest['pdi']) else None
    mdi = latest['mdi'] if pd.notna(latest['mdi']) else None
    prev_pdi = prev['pdi'] if pd.notna(prev['pdi']) else None
    prev_mdi = prev['mdi'] if pd.notna(prev['mdi']) else None
    prev_prev_pdi = prev_prev['pdi'] if pd.notna(prev_prev['pdi']) else None
    prev_prev_mdi = prev_prev['mdi'] if pd.notna(prev_prev['mdi']) else None
    
    dmi_signal = 0
    if (pdi is not None and mdi is not None and prev_pdi is not None and prev_mdi is not None and
        prev_prev_pdi is not None and prev_prev_mdi is not None):
        cross_plus_di = (
            (prev_pdi > prev_mdi and prev_prev_pdi <= prev_prev_mdi) or
            (pdi > mdi and prev_pdi <= prev_mdi)
        )
        within_5pct = mdi > 0 and abs(pdi - mdi) / max(pdi, mdi) <= 0.05
        within_1pct = pdi <= mdi and mdi > 0 and abs(pdi - mdi) / max(pdi, mdi) <= 0.01
        no_minus_di_cross = not (
            (prev_mdi > prev_pdi and prev_prev_mdi <= prev_prev_pdi) or
            (mdi > pdi and prev_mdi <= prev_pdi)
        )
        dmi_signal = 1 if (cross_plus_di and within_5pct) or (within_1pct and no_minus_di_cross) else 0
    
    # Stochastic Signal
    k = latest['k'] if pd.notna(latest['k']) else None
    d_val = latest['d'] if pd.notna(latest['d']) else None
    prev_k = prev['k'] if pd.notna(prev['k']) else None
    prev_d = prev['d'] if pd.notna(prev['d']) else None
    prev_prev_k = prev_prev['k'] if pd.notna(prev_prev['k']) else None
    prev_prev_d = prev_prev['d'] if pd.notna(prev_prev['d']) else None
    
    sto_signal = 0
    if (k is not None and d_val is not None and prev_k is not None and prev_d is not None and
        prev_prev_k is not None and prev_prev_d is not None):
        cross_last3 = (
            (prev_k > prev_d and prev_prev_k <= prev_prev_d and (min(prev_k, prev_d) <= 22 or min(k, d_val) <= 22)) or
            (k > d_val and prev_k <= prev_d and (min(k, d_val) <= 22 or min(prev_k, prev_d) <= 22))
        )
        increasing_close = k > prev_k and abs(k - d_val) <= 3 and (k <= 21 or d_val <= 21)
        sto_signal = 1 if cross_last3 or increasing_close else 0
    
    return dmi_signal, sto_signal

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
            tickers = SP500_TICKERS
            logger.info(f"No tickers in DB, using fallback list with {len(tickers)} tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tracked tickers: {e}")
        return SP500_TICKERS

def get_last_date_for_ticker(conn, ticker):
    """
    Queries the MAX(date) for a specific ticker from ohlcv table.
    Returns the date as str, or None if no data.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()[0]
    return result if result else None

def trim_old_data(conn):
    """
    Deletes data older than 6 months for all tickers to limit history.
    """
    logger.info("Trimming old data to last 6 months")
    six_months_ago = (datetime.now() - timedelta(days=30 * HISTORY_MONTHS)).strftime('%Y-%m-%d')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ohlcv WHERE date < ?", (six_months_ago,))
    deleted = cursor.rowcount
    conn.commit()
    logger.info(f"Deleted {deleted} old rows")

def recalculate_indicators(conn, ticker):
    """
    Recalculates indicators and signals for a ticker's data and updates the DB.
    """
    logger.info(f"Recalculating indicators for {ticker}")
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date", conn, params=(ticker,))
    if df.empty:
        return
    df = calculate_indicators(df)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE ohlcv SET adx = ?, pdi = ?, mdi = ?, k = ?, d = ?
            WHERE ticker = ? AND date = ?
        ''', (
            row['adx'] if pd.notna(row['adx']) else None,
            row['pdi'] if pd.notna(row['pdi']) else None,
            row['mdi'] if pd.notna(row['mdi']) else None,
            row['k'] if pd.notna(row['k']) else None,
            row['d'] if pd.notna(row['d']) else None,
            ticker,
            row['date']
        ))
    # Compute signals for latest row (last 3 rows)
    if len(df) >= 3:
        last3 = df.iloc[-3:]
        dmi_signal, sto_signal = compute_signals(last3)
        latest_date = df.iloc[-1]['date']
        cursor.execute('''
            UPDATE ohlcv SET dmi_signal = ?, sto_signal = ?
            WHERE ticker = ? AND date = ?
        ''', (dmi_signal, sto_signal, ticker, latest_date))
    conn.commit()
    logger.info(f"Updated indicators and signals for {ticker}")

def update_data():
    """
    Updates the ohlcv table with new stock data, checking last date per ticker.
    - Fetches all tickers (471 per logs, from DB or TICKERS).
    - Queries last dates for all tickers in one go, identifies stale ones (last_date < current_date or no data).
    - Processes only stale tickers in batches.
    - Per ticker: Fetches buffer, new data, etc.
    - At end, updates metadata to MAX(MAX(date) per ticker) for frontend.
    """
    logger.info("Updating database with new stock data")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all tickers
        all_tickers = get_tracked_tickers()
        if not all_tickers:
            raise ValueError("No tickers available for update")
        
        # End date for all fetches (tomorrow to include today, since end is exclusive)
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get last dates for all tickers
        logger.info("Fetching last dates for all tickers")
        df_last = pd.read_sql("SELECT ticker, MAX(date) as last_date FROM ohlcv GROUP BY ticker", conn)
        df_all = pd.DataFrame({'ticker': all_tickers})
        df_last = df_all.merge(df_last, on='ticker', how='left')
        df_last['last_date'] = df_last['last_date'].fillna('1900-01-01')
        
        # Identify stale tickers (last_date < current_date)
        stale_tickers = df_last[df_last['last_date'] < current_date]['ticker'].tolist()
        if not stale_tickers:
            logger.info("All tickers are up to date; no updates needed")
            # Still update metadata even if no updates
            logger.info("Calculating max max date across all tickers for metadata")
            cursor.execute("""
                SELECT MAX(last_date) FROM (SELECT MAX(date) as last_date FROM ohlcv GROUP BY ticker)
            """)
            max_max_date = cursor.fetchone()[0]
            if max_max_date:
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata (key, last_update)
                    VALUES ('last_ohlcv_update', ?)
                ''', (max_max_date,))
                conn.commit()
                logger.info(f"Updated metadata with last_ohlcv_update: {max_max_date}")
            else:
                logger.warning("No max max date found; metadata not updated")
            return
        
        logger.info(f"Found {len(stale_tickers)} stale tickers to update: {', '.join(stale_tickers)}")
        
        # Track overall inserted rows
        inserted_rows = 0
        
        # Process stale tickers in batches
        for i in range(0, len(stale_tickers), BATCH_SIZE):
            batch_tickers = stale_tickers[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch_tickers)} tickers ({', '.join(batch_tickers)})")
            
            for ticker in batch_tickers:
                logger.info(f"Processing ticker {ticker} in batch")
                try:
                    # Get last date (from df_last)
                    last_date_str = df_last[df_last['ticker'] == ticker]['last_date'].values[0]
                    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                    api_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    db_start_date = (last_date - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')
                    logger.info(f"Incremental for {ticker}: DB buffer from {db_start_date}, API from {api_start_date} to {end_date}")
                    
                    # This should not happen since we filtered, but check
                    if api_start_date >= end_date:
                        logger.info(f"No new data needed for {ticker} (up to date as of {last_date_str})")
                        time.sleep(0.5)
                        continue
                    
                    # Step 1: Fetch buffer data from DB
                    buffer_df = pd.DataFrame()
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
                        time.sleep(0.5)
                        continue
                    
                    # Flatten if MultiIndex
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data = new_data.droplevel(1, axis=1)
                        logger.info(f"Droplevel MultiIndex columns for {ticker}")
                    
                    # Handle DF structure
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
                    
                    # Step 5: Insert new rows with signals
                    count = 0
                    ticker_max_date = last_date_str
                    for _, row in combined_df.iterrows():
                        if row['date'] > ticker_max_date:
                            # Compute signals for the new row (use last 3 rows including this one)
                            last3 = combined_df[combined_df['date'] <= row['date']].tail(3)
                            dmi_signal, sto_signal = compute_signals(last3) if len(last3) >= 3 else (0, 0)
                            cursor.execute('''
                                INSERT OR REPLACE INTO ohlcv (
                                    ticker, date, open, high, low, close, volume, 
                                    company_name, adx, pdi, mdi, k, d, dmi_signal, sto_signal
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                row['d'] if pd.notna(row['d']) else None,
                                dmi_signal,
                                sto_signal
                            ))
                            count += 1
                            logger.info(f"Added data point for {ticker} on {row['date']}")
                            if row['date'] > ticker_max_date:
                                ticker_max_date = row['date']
                    if count > 0:
                        logger.info(f"Added {count} new data points for {ticker}")
                    inserted_rows += count
                    
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
        
        # Trim old data after updates
        trim_old_data(conn)
        
        # Recalculate indicators and signals for all tickers
        for ticker in all_tickers:
            recalculate_indicators(conn, ticker)
        
        # Update metadata to MAX of all tickers' MAX(date) for frontend
        logger.info("Calculating max max date across all tickers for metadata")
        cursor.execute("""
            SELECT MAX(last_date) FROM (SELECT MAX(date) as last_date FROM ohlcv GROUP BY ticker)
        """)
        max_max_date = cursor.fetchone()[0]
        if max_max_date:
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, last_update)
                VALUES ('last_ohlcv_update', ?)
            ''', (max_max_date,))
            conn.commit()
            logger.info(f"Updated metadata with last_ohlcv_update: {max_max_date}")
        else:
            logger.warning("No max max date found; metadata not updated")
        
        logger.info(f"Total inserted or replaced {inserted_rows} rows into ohlcv table")
        logger.info("Database updated successfully")
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise
    finally:
        if conn:
            conn.close()

@app.get("/stocks/tickers")
async def get_tickers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM ohlcv")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

@app.get("/stocks/{ticker}")
async def get_stock_data(ticker: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date", conn, params=(ticker.upper(),))
    conn.close()
    if df.empty:
        return {"error": f"No data for {ticker}"}
    # Clean inf and NaN values to prevent JSON errors
    df = df.replace([np.inf, -np.inf, np.nan], None)
    return {"ohlcv": df.to_dict(orient="records")}

@app.post("/stocks/batch")
async def get_batch_stock_data(tickers: List[str]):
    conn = sqlite3.connect(DB_PATH)
    results = []
    for ticker in tickers:
        df = pd.read_sql_query("SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date DESC LIMIT 1", conn, params=(ticker.upper(),))
        if df.empty:
            results.append({"ticker": ticker, "latest_ohlcv": None})
        else:
            # Clean inf and NaN in latest row
            df = df.replace([np.inf, -np.inf, np.nan], None)
            latest = df.iloc[0].to_dict()
            results.append({"ticker": ticker, "latest_ohlcv": latest, "company_name": latest.get("company_name")})
    conn.close()
    return results

@app.get("/metadata")
async def get_metadata():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT last_update FROM metadata WHERE key = 'last_ohlcv_update'")
    result = cursor.fetchone()
    conn.close()
    return {"last_ohlcv_update": result[0] if result else None}

@app.get("/live-prices")
async def get_live_prices(tickers: str = Query(...)):
    logger.info(f"Received request for live prices: {tickers}")
    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")
    
    results = []
    est_tz = pytz.timezone('America/New_York')
    
    for ticker in ticker_list:
        try:
            tick = yf.Ticker(ticker)
            info = tick.info
            price = info.get('regularMarketPrice')
            timestamp_unix = info.get('regularMarketTime')
            volume = info.get('regularMarketVolume')
            
            timestamp = None
            if timestamp_unix:
                utc_dt = datetime.fromtimestamp(timestamp_unix)
                est_dt = utc_dt.astimezone(est_tz)
                timestamp = est_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            results.append({
                "ticker": ticker,
                "price": price,
                "timestamp": timestamp,
                "volume": volume
            })
        except Exception as e:
            logger.warning(f"Error fetching live price for {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "price": None,
                "timestamp": None,
                "volume": None
            })
    
    return results

@app.post("/update-db")
async def manual_update_db():
    try:
        update_data()
        return {"message": "Database update triggered successfully"}
    except Exception as e:
        logger.error(f"Error during manual database update: {e}")
        raise HTTPException(status_code=500, detail="Error updating database")

@app.post("/recalculate-indicators")
async def manual_recalculate_indicators():
    try:
        conn = sqlite3.connect(DB_PATH)
        all_tickers = get_tracked_tickers()
        for ticker in all_tickers:
            recalculate_indicators(conn, ticker)
        conn.close()
        return {"message": "Indicators recalculated for all tickers"}
    except Exception as e:
        logger.error(f"Error during manual recalculation: {e}")
        raise HTTPException(status_code=500, detail="Error recalculating indicators")

@app.post("/migrate-db")
async def manual_migrate_db():
    try:
        migrate_db()
        return {"message": "Database migration completed successfully"}
    except Exception as e:
        logger.error(f"Error during manual migration: {e}")
        raise HTTPException(status_code=500, detail="Error migrating database")