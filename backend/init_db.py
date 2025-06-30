import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import List

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
        
        # ADX, +DI, -DI
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
        
        # Stochastic Oscillator
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

def get_sp500_tickers():
    logger.info("Fetching S&P 500 tickers")
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = table['Symbol'].tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)
        if len(tickers) == 1:
            data['ticker'] = tickers[0]
            data = data.reset_index()[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        else:
            data = data.stack(level=0).reset_index().rename(columns={'Date': 'date', 'level_1': 'ticker'})
            data = data[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        data['company_name'] = None
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                data.loc[data['ticker'] == ticker, 'company_name'] = ticker_obj.info.get('longName', None)
            except Exception as e:
                logger.warning(f"Could not fetch company name for {ticker}: {e}")
                data.loc[data['ticker'] == ticker, 'company_name'] = None
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def update_data():
    logger.info("Updating database with new stock data")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        tickers = get_sp500_tickers()
        if not tickers:
            logger.error("No tickers found, aborting update")
            return
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        data = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            logger.error("No stock data fetched, aborting update")
            return
        
        grouped = data.groupby('ticker')
        for ticker, group in grouped:
            try:
                group = calculate_indicators(group)
                for _, row in group.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume, company_name, adx, pdi, mdi, k, d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        row['date'],
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        int(row['volume']),
                        row['company_name'],
                        row['adx'] if not pd.isna(row['adx']) else None,
                        row['pdi'] if not pd.isna(row['pdi']) else None,
                        row['mdi'] if not pd.isna(row['mdi']) else None,
                        row['k'] if not pd.isna(row['k']) else None,
                        row['d'] if not pd.isna(row['d']) else None
                    ))
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, last_update)
            VALUES ('last_ohlcv_update', ?)
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
        
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