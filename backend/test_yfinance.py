import yfinance as yf
import time
from datetime import datetime, timedelta
import requests.exceptions

def test_yfinance(ticker: str):
    print(f"\nTesting ticker: {ticker}")
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            # Fetch company name
            company_name = stock.info.get('longName', f"{ticker} Inc.")
            print(f"Company name: {company_name}")
            # Fetch OHLCV data (1 year)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                print(f"No data found for {ticker}")
                return
            print(f"OHLCV data (latest row):\n{df.tail(1)[['Open', 'High', 'Low', 'Close', 'Volume']]}")
            return
        except (requests.exceptions.HTTPError, Exception) as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                print(f"Failed to fetch data for {ticker}: {e}")

# Test tickers
tickers = ["MMM", "AOS", "ABT", "TSLA"]
for ticker in tickers:
    test_yfinance(ticker)