import { OHLCV, StockAnalysisResult, IndicatorValues } from '../types';
import { ADX_TREND_STRENGTH_THRESHOLD, DMI_CROSSOVER_PROXIMITY_PERCENTAGE } from '../constants';

const API_BASE_URL = 'http://localhost:8000'; // Update to https://your-app.onrender.com for deployment

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
    return data.ohlcv.map((d: any) => ({
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
  } catch (err) {
    console.error(`Fetch error for ${ticker}:`, err);
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
      adx: latestOhlcvDataPoint.adx ?? 0,
      pdi: latestOhlcvDataPoint.pdi ?? 0,
      mdi: latestOhlcvDataPoint.mdi ?? 0,
      k: latestOhlcvDataPoint.k ?? 0,
      d: latestOhlcvDataPoint.d ?? 0,
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
      latestIndicators.pdi > latestIndicators.mdi &&
      previousIndicators &&
      previousIndicators.pdi <= previousIndicators.mdi;
    const pdiCurrentlyAboveMdi = latestIndicators.pdi > latestIndicators.mdi;
    const pdiNearingMdi =
      latestIndicators.mdi > 0 &&
      Math.abs(latestIndicators.pdi - latestIndicators.mdi) / latestIndicators.mdi <= DMI_CROSSOVER_PROXIMITY_PERCENTAGE;
    const isStrongTrend = latestIndicators.adx >= ADX_TREND_STRENGTH_THRESHOLD;

    const isStochasticClose = (k: number, d: number) => Math.abs(k - d) <= 8;
    const isStochasticOversold = (k: number, d: number) => k <= 20 || d <= 20;
    const isStochasticSignal = (k: number, d: number) => isStochasticClose(k, d) && isStochasticOversold(k, d);

    const hasStochasticSignal =
      isStochasticSignal(latestIndicators.k, latestIndicators.d) ||
      (previousIndicators && isStochasticSignal(previousIndicators.k, previousIndicators.d));

    console.log(`STO alert for ${ticker}:`, {
      currentK: latestIndicators.k.toFixed(2),
      currentD: latestIndicators.d.toFixed(2),
      prevK: previousIndicators?.k.toFixed(2),
      prevD: previousIndicators?.d.toFixed(2),
      hasStochasticSignal,
    });

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
      message = `No alerts. ADX: ${latestIndicators.adx.toFixed(2)}, +DI: ${latestIndicators.pdi.toFixed(2)}, -DI: ${latestIndicators.mdi.toFixed(2)}, %K: ${latestIndicators.k.toFixed(2)}, %D: ${latestIndicators.d.toFixed(2)}.`;
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
  for (const ticker of tickers) {
    const stockData = await analyzeStockTicker(ticker);
    results.push(stockData);
  }
  return results;
};