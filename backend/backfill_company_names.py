import sqlite3
import yfinance as yf
from tqdm import tqdm
import time
import requests.exceptions

DB_PATH = "stocks.db"

def get_company_name(ticker: str) -> str:
    retries = 3
    delay = 2
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT company_name FROM company_names WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', f"{ticker} Inc.")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO company_names (ticker, company_name, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (ticker, company_name))
            conn.commit()
            conn.close()
            return company_name
        except (requests.exceptions.HTTPError, Exception) as e:
            if attempt < retries - 1:
                print(f"Retry {attempt + 1}/{retries} for {ticker}: {e}")
                time.sleep(delay * (2 ** attempt))
            else:
                print(f"Error fetching name for {ticker}: {e}")
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO company_names (ticker, company_name, timestamp)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (ticker, f"{ticker} Inc."))
                conn.commit()
                conn.close()
                return f"{ticker} Inc."

def backfill_company_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM ohlcv WHERE company_name IS NULL OR company_name = 'N/A' OR company_name LIKE '% Inc.'")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    for ticker in tqdm(tickers, desc="Backfilling company names"):
        company_name = get_company_name(ticker)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE ohlcv SET company_name = ? WHERE ticker = ?",
            (company_name, ticker)
        )
        conn.commit()
        conn.close()

if __name__ == "__main__":
    backfill_company_names()