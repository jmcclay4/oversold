from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import logging
import os
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from init_db import init_db, update_data, rebuild_database, fetch_live_prices
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/data/stocks.db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application")
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        if not os.path.exists(DB_PATH):
            logger.warning(f"Database file {DB_PATH} not found, initializing")
            init_db()
            logger.info(f"Database initialized at {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'")
        if not cursor.fetchone():
            logger.warning("Table 'ohlcv' does not exist, initializing database")
            conn.close()
            init_db()
            logger.info("Database tables created")
        else:
            logger.info("Database and ohlcv table exist, skipping initialization")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
    logger.info("Shutting down FastAPI application")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://oversold.jmcclay.com", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"]
)

class OHLCV(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    company_name: Optional[str]
    adx: Optional[float]
    pdi: Optional[float]
    mdi: Optional[float]
    k: Optional[float]
    d: Optional[float]

class StockDataResponse(BaseModel):
    ohlcv: List[OHLCV]

class BatchStockDataResponse(BaseModel):
    ticker: str
    company_name: Optional[str]
    latest_ohlcv: Optional[OHLCV]

class LivePriceResponse(BaseModel):
    ticker: str
    price: Optional[float]
    timestamp: Optional[str]
    volume: Optional[int]

class MetadataResponse(BaseModel):
    last_ohlcv_update: Optional[str]

@app.get("/stocks/tickers", response_model=List[str])
async def get_all_tickers():
    logger.info("Received request for all tickers")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        logger.info(f"Returning {len(tickers)} tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/stocks/{ticker}", response_model=StockDataResponse)
async def get_stock_data(ticker: str):
    logger.info(f"Received request for ticker: {ticker}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, open, high, low, close, volume, company_name, adx, pdi, mdi, k, d
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date
        """, (ticker.upper(),))
        rows = cursor.fetchall()
        if not rows:
            logger.warning(f"No data found for ticker {ticker}")
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        ohlcv_data = [
            OHLCV(
                date=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5],
                company_name=row[6],
                adx=row[7],
                pdi=row[8],
                mdi=row[9],
                k=row[10],
                d=row[11]
            )
            for row in rows
        ]
        logger.info(f"Returning {len(ohlcv_data)} data points for {ticker}")
        return StockDataResponse(ohlcv=ohlcv_data)
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/stocks/batch", response_model=List[BatchStockDataResponse])
async def get_batch_stock_data(tickers: List[str], response: Response):
    logger.info(f"Received batch request for {len(tickers)} tickers")
    response.headers["Cache-Control"] = "no-cache"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT ticker, date, open, high, low, close, volume, company_name, adx, pdi, mdi, k, d
            FROM ohlcv
            WHERE ticker IN ({placeholders})
            AND date = (
                SELECT MAX(date)
                FROM ohlcv
                WHERE ticker = ohlcv.ticker
            )
        """
        cursor.execute(query, [t.upper() for t in tickers])
        rows = cursor.fetchall()
        ticker_set = set(t.upper() for t in tickers)
        results = []
        for row in rows:
            results.append(BatchStockDataResponse(
                ticker=row[0],
                company_name=row[7],
                latest_ohlcv=OHLCV(
                    date=row[1],
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    volume=row[6],
                    company_name=row[7],
                    adx=row[8],
                    pdi=row[9],
                    mdi=row[10],
                    k=row[11],
                    d=row[12]
                )
            ))
        for ticker in ticker_set:
            if ticker not in {r.ticker for r in results}:
                results.append(BatchStockDataResponse(
                    ticker=ticker,
                    company_name=None,
                    latest_ohlcv=None
                ))
        logger.info(f"Returning data for {len(results)} tickers")
        return results
    except Exception as e:
        logger.error(f"Error fetching batch data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

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

@app.get("/metadata", response_model=MetadataResponse)
async def get_metadata(response: Response):
    logger.info("Received request for metadata")
    response.headers["Cache-Control"] = "no-cache"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT last_update FROM metadata WHERE key = 'last_ohlcv_update'")
        result = cursor.fetchone()
        last_update = result[0] if result else None
        logger.info(f"Returning last OHLCV update: {last_update}")
        if last_update is None:
            logger.warning("No last_ohlcv_update found in metadata table")
        return MetadataResponse(last_ohlcv_update=last_update)
    except sqlite3.Error as e:
        logger.error(f"Database error fetching metadata: {e}")
        return MetadataResponse(last_ohlcv_update=None)
    except Exception as e:
        logger.error(f"Unexpected error fetching metadata: {e}")
        return MetadataResponse(last_ohlcv_update=None)
    finally:
        conn.close()

@app.post("/update-db")
async def update_database():
    logger.info("Received request to update database")
    try:
        update_data()
        logger.info("Database update completed successfully")
        return {"status": "success", "message": "Database updated"}
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update database: {str(e)}")

@app.post("/rebuild-db")
async def rebuild_database_endpoint():
    logger.info("Received request to rebuild database")
    try:
        rebuild_database()
        logger.info("Database rebuild completed successfully")
        return {"status": "success", "message": "Database rebuilt with S&P 500 data"}
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild database: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
