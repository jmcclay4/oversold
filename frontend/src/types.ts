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
  adx: number;
  pdi: number;
  mdi: number;
  k: number;
  d: number;
}

export interface StockAnalysisResult {
  ticker: string;
  companyName: string;
  latestPrice?: number;
  percentChange?: number; // Retained for ∆ column
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

export interface FilterCriteria {
  searchTerm?: string;
}