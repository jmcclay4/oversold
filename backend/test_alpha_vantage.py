import requests
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "T9YOQWD20T259F4I")
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
BATCH_SIZE = 5

def test_alpha_vantage_batch(tickers: list[str], batch_size: int = 5):
    logger.info(f"Testing Alpha Vantage batch with {len(tickers)} tickers, batch size {batch_size}")
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY":
        logger.error("Alpha Vantage API key is not set or invalid")
        return False
    try:
        results = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch: {batch}")
            for ticker in batch:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                logger.info(f"Requesting URL: {url}")
                response = requests.get(url, timeout=10)
                logger.info(f"Response status: {response.status_code}")
                logger.debug(f"Response content: {response.text}")
                response.raise_for_status()
                data = response.json()
                if "Global Quote" not in data:
                    logger.error(f"No 'Global Quote' in response for {ticker}: {data}")
                    continue
                quote = data["Global Quote"]
                price = quote.get("05. price")
                if not price:
                    logger.error(f"No price in response for {ticker}: {quote}")
                    continue
                try:
                    float_price = float(price)
                    results.append({"ticker": ticker, "price": float_price})
                    logger.info(f"Ticker: {ticker}, Price: {float_price}")
                except ValueError as e:
                    logger.error(f"Invalid price for {ticker}: {price}, error: {e}")
                time.sleep(12)  # Respect 5 calls/minute limit
        logger.info(f"Fetched prices for {len(results)} tickers")
        return len(results) == len(tickers)
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_alpha_vantage_batch(TEST_TICKERS, BATCH_SIZE)
    logger.info(f"Batch test {'succeeded' if success else 'failed'}")