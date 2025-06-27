import { OHLCV, StockAnalysisResult, IndicatorValues, BatchStockDataResponse } from '../types';
import { ADX_TREND_STRENGTH_THRESHOLD, DMI_CROSSOVER_PROXIMITY_PERCENTAGE } from '../constants';

const API_BASE_URL = 'https://oversold-backend.fly.dev';
const BATCH_SIZE = 50;

const fetchStockData = async (ticker: string): Promise<OHLCV[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/stocks/${ticker.toUpperCase()}`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    const ohlcv = data.ohlcv.map((d: any) => ({
      date: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
      companyName: d.company_name || undefined,
      adx: d.adx ?? null,
      pdi: d.pdi ?? null,
      mdi: d.mdi ?? null,
      k: d.k ?? null,
      d: d.d ?? null,
    }));
    console.log(`Fetched data for ${ticker}, latest:`, ohlcv[ohlcv.length - 1]);
    return ohlcv;
  } catch (err) {
    console.error(`Fetch error for ${ticker}:`, err);
    throw err;
  }
};

const fetchBatchStockData = async (tickers: string[]): Promise<BatchStockDataResponse[]> => {
  console.log(`Fetching batch data for ${tickers.length} tickers`);
  try {
    const response = await fetch(`${API_BASE_URL}/stocks/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tickers),
    });
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (err) {
    console.error('Batch fetch error:', err);
    throw err;
  }
};

export const fetchAllTickers = async (): Promise<string[]> => {
  console.log('fetchAllTickers: Fetching tickers from backend...');
  try {
    const response = await fetch(`${API_BASE_URL}/stocks/tickers`);
    console.log('fetchAllTickers: Response status:', response.status);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const tickers: string[] = await response.json();
    console.log('fetchAllTickers: Received', tickers.length, 'tickers');
    return tickers;
  } catch (error) {
    console.error('fetchAllTickers: Error:', (error as Error).message);
    return [];
  }
};

export const fetchMetadata = async (): Promise<{ last_ohlcv_update: string | null }> => {
  console.log('fetchMetadata: Fetching metadata...');
  try {
    const response = await fetch(`${API_BASE_URL}/metadata`);
    if (!response.ok) {
      console.error(`fetchMetadata: HTTP error ${response.status}`);
      return { last_ohlcv_update: null };
    }
    const data = await response.json();
    console.log('fetchMetadata: Received:', data);
    return data;
  } catch (error) {
    console.error('fetchMetadata: Error:', (error as Error).message);
    return { last_ohlcv_update: null };
  }
};

export const fetchLivePrices = async (tickers: string[]): Promise<{ ticker: string, price: number, timestamp: string }[]> => {
  console.log(`Fetching live prices for ${tickers.length} tickers`);
  try {
    const response = await fetch(`${API_BASE_URL}/live-prices?tickers=${tickers.join(',')}&batch_size=100`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    const data: { ticker: string, price: number, timestamp: string }[] = await response.json();
    console.log(`Received live prices for ${data.length} tickers`);
    return data;
  } catch (err) {
    console.error('Live prices fetch error:', err);
    return [];
  }
};

const getCompanyName = (ticker: string, ohlcvData: OHLCV[]): string => {
  if (ohlcvData.length > 0 && ohlcvData[0].companyName) {
    return ohlcvData[0].companyName;
  }
  const companyMap: { [key: string]: string } = {
    MMM: '3M Company',
    AOS: 'A.O. Smith Corporation',
    ABT: 'Abbott Laboratories',
    TSLA: 'Tesla, Inc.',
  };
  return companyMap[ticker] || `${ticker} Inc.`;
};

export const analyzeStockTicker = async (ticker: string): Promise<StockAnalysisResult> => {
  try {
    const ohlcvData = await fetchStockData(ticker);
    if (ohlcvData.length === 0) {
      return {
        ticker,
        statusTags: [],
        meetsCriteria: false,
        message: `No historical data found for ${ticker}.`,
        error: `No data for ${ticker}.`,
        companyName: getCompanyName(ticker, ohlcvData),
        ohlcv: [],
      };
    }
    if (ohlcvData.length < 2) {
      return {
        ticker,
        statusTags: [],
        meetsCriteria: false,
        message: `Not enough data points (${ohlcvData.length}) for ${ticker}.`,
        error: `Insufficient data for ${ticker}.`,
        companyName: getCompanyName(ticker, ohlcvData),
        ohlcv: ohlcvData,
      };
    }
    const latestOhlcvDataPoint = ohlcvData[ohlcvData.length - 1];
    const previousOhlcvDataPoint = ohlcvData[ohlcvData.length - 2];
    const latestPrice = latestOhlcvDataPoint.close;
    const percentChange = previousOhlcvDataPoint.close
      ? ((latestPrice - previousOhlcvDataPoint.close) / previousOhlcvDataPoint.close) * 100
      : undefined;
    const latestIndicators: IndicatorValues = {
      adx: latestOhlcvDataPoint.adx ?? null,
      pdi: latestOhlcvDataPoint.pdi ?? null,
      mdi: latestOhlcvDataPoint.mdi ?? null,
      k: latestOhlcvDataPoint.k ?? null,
      d: latestOhlcvDataPoint.d ?? null,
    };
    let previousIndicators: IndicatorValues | undefined = undefined;
    if (
      previousOhlcvDataPoint.adx != null &&
      previousOhlcvDataPoint.pdi != null &&
      previousOhlcvDataPoint.mdi != null &&
      previousOhlcvDataPoint.k != null &&
      previousOhlcvDataPoint.d != null
    ) {
      previousIndicators = {
        adx: previousOhlcvDataPoint.adx,
        pdi: previousOhlcvDataPoint.pdi,
        mdi: previousOhlcvDataPoint.mdi,
        k: previousOhlcvDataPoint.k,
        d: previousOhlcvDataPoint.d,
      };
    }
    const statusTags: string[] = [];
    let meetsCriteria = false;
    let message = "";
    const pdiCrossedAboveMdi =
      latestIndicators.pdi != null &&
      latestIndicators.mdi != null &&
      previousIndicators &&
      previousIndicators.pdi != null &&
      previousIndicators.mdi != null &&
      latestIndicators.pdi > latestIndicators.mdi &&
      previousIndicators.pdi <= previousIndicators.mdi;
    const pdiCurrentlyAboveMdi =
      latestIndicators.pdi != null && latestIndicators.mdi != null && latestIndicators.pdi > latestIndicators.mdi;
    const pdiNearingMdi =
      latestIndicators.pdi != null &&
      latestIndicators.mdi != null &&
      latestIndicators.mdi > 0 &&
      Math.abs(latestIndicators.pdi - latestIndicators.mdi) / latestIndicators.mdi <= DMI_CROSSOVER_PROXIMITY_PERCENTAGE;
    const isStrongTrend = latestIndicators.adx != null && latestIndicators.adx >= ADX_TREND_STRENGTH_THRESHOLD;
    const isStochasticClose = (k: number | null, d: number | null) => k != null && d != null && Math.abs(k - d) <= 8;
    const isStochasticOversold = (k: number | null, d: number | null) => (k != null && k <= 20) || (d != null && d <= 20);
    const isStochasticSignal = (k: number | null, d: number | null) => isStochasticClose(k, d) && isStochasticOversold(k, d);
    const hasStochasticSignal =
      isStochasticSignal(latestIndicators.k, latestIndicators.d) ||
      (previousIndicators && isStochasticSignal(previousIndicators.k, previousIndicators.d));
    if (pdiCrossedAboveMdi || pdiCurrentlyAboveMdi || pdiNearingMdi) {
      statusTags.push('DMI');
      meetsCriteria = true;
    }
    if (isStrongTrend) {
      statusTags.push('ADX');
    }
    if (hasStochasticSignal) {
      statusTags.push('STO');
      meetsCriteria = true;
    }
    if (statusTags.length === 0) {
      message = `No alerts. ADX: ${latestIndicators.adx?.toFixed(2) ?? 'N/A'}, +DI: ${latestIndicators.pdi?.toFixed(2) ?? 'N/A'}, -DI: ${latestIndicators.mdi?.toFixed(2) ?? 'N/A'}, %K: ${latestIndicators.k?.toFixed(2) ?? 'N/A'}, %D: ${latestIndicators.d?.toFixed(2) ?? 'N/A'}.`;
    } else {
      message = `Alerts: ${statusTags.join(', ')}.`;
    }
    return {
      ticker,
      companyName: getCompanyName(ticker, ohlcvData),
      latestPrice,
      percentChange,
      latestOhlcvDataPoint,
      latestIndicators,
      previousIndicators,
      historicalDates: ohlcvData.map(d => d.date),
      historicalClosePrices: ohlcvData.map(d => d.close),
      historicalAdx: ohlcvData.map(d => d.adx ?? null),
      historicalPdi: ohlcvData.map(d => d.pdi ?? null),
      historicalMdi: ohlcvData.map(d => d.mdi ?? null),
      historicalK: ohlcvData.map(d => d.k ?? null),
      historicalD: ohlcvData.map(d => d.d ?? null),
      ohlcv: ohlcvData,
      statusTags,
      meetsCriteria,
      message,
    };
  } catch (err) {
    return {
      ticker,
      statusTags: [],
      meetsCriteria: false,
      message: `Error analyzing ${ticker}: ${(err as Error).message}`,
      error: (err as Error).message,
      companyName: getCompanyName(ticker, []),
      ohlcv: [],
    };
  }
};

export const analyzeTrackedStocks = async (tickers: string[]): Promise<StockAnalysisResult[]> => {
  const results: StockAnalysisResult[] = [];
  const batches = [];
  for (let i = 0; i < tickers.length; i += BATCH_SIZE) {
    batches.push(tickers.slice(i, i + BATCH_SIZE));
  }
  for (const batch of batches) {
    try {
      const batchData = await fetchBatchStockData(batch);
      for (const stock of batchData) {
        const ticker = stock.ticker;
        const latestOhlcv = stock.latest_ohlcv;
        if (!latestOhlcv) {
          results.push({
            ticker,
            statusTags: [],
            meetsCriteria: false,
            message: `No data found for ${ticker}`,
            error: `No data for ${ticker}`,
            companyName: stock.company_name || `${ticker} Inc.`,
            ohlcv: [],
          });
          continue;
        }
        const latestIndicators: IndicatorValues = {
          adx: latestOhlcv.adx ?? null,
          pdi: latestOhlcv.pdi ?? null,
          mdi: latestOhlcv.mdi ?? null,
          k: latestOhlcv.k ?? null,
          d: latestOhlcv.d ?? null,
        };
        const statusTags: string[] = [];
        let meetsCriteria = false;
        let message = "";
        const pdiCurrentlyAboveMdi =
          latestIndicators.pdi != null && latestIndicators.mdi != null && latestIndicators.pdi > latestIndicators.mdi;
        const pdiNearingMdi =
          latestIndicators.pdi != null &&
          latestIndicators.mdi != null &&
          latestIndicators.mdi > 0 &&
          Math.abs(latestIndicators.pdi - latestIndicators.mdi) / latestIndicators.mdi <= DMI_CROSSOVER_PROXIMITY_PERCENTAGE;
        const isStrongTrend = latestIndicators.adx != null && latestIndicators.adx >= ADX_TREND_STRENGTH_THRESHOLD;
        const isStochasticClose = (k: number | null, d: number | null) => k != null && d != null && Math.abs(k - d) <= 8;
        const isStochasticOversold = (k: number | null, d: number | null) => (k != null && k <= 20) || (d != null && d <= 20);
        const isStochasticSignal = (k: number | null, d: number | null) => isStochasticClose(k, d) && isStochasticOversold(k, d);
        const hasStochasticSignal = isStochasticSignal(latestIndicators.k, latestIndicators.d);
        if (pdiCurrentlyAboveMdi || pdiNearingMdi) {
          statusTags.push('DMI');
          meetsCriteria = true;
        }
        if (isStrongTrend) {
          statusTags.push('ADX');
        }
        if (hasStochasticSignal) {
          statusTags.push('STO');
          meetsCriteria = true;
        }
        if (statusTags.length === 0) {
          message = `No alerts. ADX: ${latestIndicators.adx?.toFixed(2) ?? 'N/A'}, +DI: ${latestIndicators.pdi?.toFixed(2) ?? 'N/A'}, -DI: ${latestIndicators.mdi?.toFixed(2) ?? 'N/A'}, %K: ${latestIndicators.k?.toFixed(2) ?? 'N/A'}, %D: ${latestIndicators.d?.toFixed(2) ?? 'N/A'}.`;
        } else {
          message = `Alerts: ${statusTags.join(', ')}.`;
        }
        results.push({
          ticker,
          companyName: stock.company_name || `${ticker} Inc.`,
          latestPrice: latestOhlcv.close,
          percentChange: undefined, // Calculate on client-side if needed
          latestOhlcvDataPoint: latestOhlcv,
          latestIndicators,
          previousIndicators: undefined,
          historicalDates: [], // Empty for table data
          historicalClosePrices: [],
          historicalAdx: [],
          historicalPdi: [],
          historicalMdi: [],
          historicalK: [],
          historicalD: [],
          ohlcv: [latestOhlcv],
          statusTags,
          meetsCriteria,
          message,
        });
      }
    } catch (err) {
      console.error(`Batch error: ${err}`);
      batch.forEach(ticker => {
        results.push({
          ticker,
          statusTags: [],
          meetsCriteria: false,
          message: `Failed to fetch data for ${ticker}`,
          error: (err as Error).message,
          companyName: `${ticker} Inc.`,
          ohlcv: [],
        });
      });
    }
  }
  return results;
};