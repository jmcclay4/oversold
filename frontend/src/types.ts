export interface OHLCV {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  companyName?: string;
  adx: number | null;
  pdi: number | null;
  mdi: number | null;
  k: number | null;
  d: number | null;
}

export interface IndicatorValues {
  adx: number | null;
  pdi: number | null;
  mdi: number | null;
  k: number | null;
  d: number | null;
}

export interface StockAnalysisResult {
  ticker: string;
  companyName: string;
  latestPrice?: number;
  livePrice?: number; 
  percentChange?: number;
  latestOhlcvDataPoint?: OHLCV;
  latestIndicators?: IndicatorValues;
  previousIndicators?: IndicatorValues;
  historicalDates?: string[];
  historicalClosePrices?: number[];
  historicalAdx?: (number | null)[];
  historicalPdi?: (number | null)[];
  historicalMdi?: (number | null)[];
  historicalK?: (number | null)[];
  historicalD?: (number | null)[];
  ohlcv: OHLCV[];
  statusTags: string[];
  meetsCriteria: boolean;
  message: string;
  error?: string;
}

export interface BatchStockDataResponse {
  ticker: string;
  company_name: string | null;
  latest_ohlcv: OHLCV | null;
}

export interface FilterCriteria {
  searchTerm?: string;
}