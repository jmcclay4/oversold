export interface OHLCV {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  company_name?: string;
  adx?: number;
  pdi?: number;
  mdi?: number;
  k?: number;
  d?: number;
}

export interface Indicators {
  adx?: number;
  pdi?: number;
  mdi?: number;
  k?: number;
  d?: number;
}

export interface StockAnalysisResult {
  ticker: string;
  companyName?: string;
  latestOhlcvDataPoint?: OHLCV;
  latestPrice?: number;
  percentChange?: number;
  latestIndicators?: Indicators;
  error?: string;
  statusTags: string[];
}

export interface FilterCriteria {
  searchTerm?: string;
}

export interface LivePrice {
  ticker: string;
  price: number;
  timestamp: string;
}

export interface StockAnalysisTableProps {
  results: StockAnalysisResult[];
  favoriteTickers: Set<string>;
  onToggleFavorite: (ticker: string) => void;
  onRowClick: (ticker: string) => void;
  selectedTickerForChart: string | null;
  livePrices: Record<string, LivePrice>;
}