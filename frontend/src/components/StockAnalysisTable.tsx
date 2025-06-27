import React, { useState, useEffect, useRef } from 'react';
import { StockAnalysisResult, OHLCV } from '../types';
import { ADX_TREND_STRENGTH_THRESHOLD, API_CALL_DELAY_MS } from '../constants';
import { fetchLivePrices } from '../services/stockDataService';

interface StockAnalysisTableProps {
  results: StockAnalysisResult[];
  favoriteTickers: Set<string>;
  onToggleFavorite: (ticker: string) => void;
  onRowClick: (ticker: string) => void;
  selectedTickerForChart?: string | null;
}

const getSignalColor = (tag: string, error?: string): string => {
  if (error) return 'bg-red-700 text-red-100';
  if (tag === 'DMI') return 'bg-teal-500 text-teal-100';
  if (tag === 'ADX') return 'bg-blue-500 text-blue-100';
  if (tag === 'STO') return 'bg-purple-500 text-purple-100';
  return 'bg-yellow-700 text-yellow-100';
};

const getPriceColor = (price?: number, close?: number): string => {
  if (!price || !close) return 'text-slate-500';
  const diff = ((price - close) / close) * 100;
  if (diff > 0.5) return 'text-green-400';
  if (diff < -0.5) return 'text-red-400';
  return 'text-slate-300';
};

const formatPrice = (price?: number) => price?.toFixed(2) ?? '-';
const formatPercent = (percent?: number) => percent !== undefined ? `${percent.toFixed(2)}%` : 'N/A';

const OhlcvDisplay: React.FC<{ ohlcv?: OHLCV }> = ({ ohlcv }) => {
  if (!ohlcv) return <span className="text-slate-500">N/A</span>;
  return (
    <div className="text-xs text-slate-400">
      <div>O: {ohlcv.open.toFixed(2)} H: {ohlcv.high.toFixed(2)}</div>
      <div>L: {ohlcv.low.toFixed(2)} C: {ohlcv.close.toFixed(2)}</div>
      <div>V: {ohlcv.volume.toLocaleString()}</div>
    </div>
  );
};

export const StockAnalysisTable: React.FC<StockAnalysisTableProps> = ({ 
  results, 
  favoriteTickers,
  onToggleFavorite,
  onRowClick,
  selectedTickerForChart
}) => {
  const [livePrices, setLivePrices] = useState<{ [ticker: string]: { price: number, timestamp: string } }>({});
  const [loadedTickers, setLoadedTickers] = useState<string[]>([]);
  const [hasMore, setHasMore] = useState<boolean>(true);
  const tableRef = useRef<HTMLDivElement>(null);

  const fetchLivePricesBatch = async (tickers: string[], reset: boolean = false): Promise<number> => {
    try {
      const data: { ticker: string, price: number, timestamp: string }[] = await fetchLivePrices(tickers);
      const prices = data.reduce((acc: { [ticker: string]: { price: number, timestamp: string } }, item) => {
        acc[item.ticker] = { price: item.price, timestamp: item.timestamp };
        return acc;
      }, {});
      setLivePrices(prev => (reset ? prices : { ...prev, ...prices }));
      return data.length;
    } catch (error) {
      console.error('Error fetching live prices:', error);
      return 0;
    }
  };

  useEffect(() => {
    if (results.length > 0) {
      const tickers = results.map(r => r.ticker);
      setLoadedTickers(tickers.slice(0, 100));
      const batches: string[][] = [];
      for (let i = 0; i < tickers.length; i += 100) {
        batches.push(tickers.slice(i, i + 100));
      }
      const fetchAllBatches = async () => {
        for (const batch of batches) {
          await fetchLivePricesBatch(batch);
          await new Promise(resolve => setTimeout(resolve, API_CALL_DELAY_MS));
        }
      };
      fetchAllBatches();
    }
  }, [results]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore) {
          const start = loadedTickers.length;
          const nextBatch = results.slice(start, start + 100).map(r => r.ticker);
          if (nextBatch.length === 0) {
            setHasMore(false);
            return;
          }
          fetchLivePricesBatch(nextBatch).then((count: number) => {
            if (count > 0) {
              setLoadedTickers(prev => [...prev, ...nextBatch]);
            } else {
              setHasMore(false);
            }
          });
        }
      },
      { threshold: 0.1 }
    );
    if (tableRef.current) {
      observer.observe(tableRef.current);
    }
    return () => {
      if (tableRef.current) {
        observer.unobserve(tableRef.current);
      }
    };
  }, [loadedTickers, hasMore, results]);

  if (results.length === 0) {
    return <p className="text-center text-slate-400 py-4">No analysis results to display.</p>;
  }
  
  const headerCellClass = "px-3 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider sticky top-0 z-10 bg-slate-900";

  return (
    <div className="bg-slate-800 shadow-2xl rounded-xl overflow-hidden mt-6">
      <div className="overflow-x-auto max-h-[60vh]">
        <table className="min-w-full divide-y divide-slate-700">
          <thead className="bg-slate-900">
            <tr>
              <th scope="col" className={`${headerCellClass} text-center`}>Fav</th>
              <th scope="col" className={headerCellClass}>Ticker</th>
              <th scope="col" className={headerCellClass}>
                <div className="flex items-center">
                  Price
                  <button
                    onClick={() => fetchLivePricesBatch(results.map(r => r.ticker), true)}
                    className="ml-2 p-1 bg-slate-900 hover:bg-slate-700 rounded-full text-white text-xs"
                    aria-label="Refresh live prices"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h5m-5 0a9 9 0 1114 0v-5h-5" />
                    </svg>
                  </button>
                </div>
              </th>
              <th scope="col" className={headerCellClass}>Close</th>
              <th scope="col" className={headerCellClass}>∆</th>
              <th scope="col" className={headerCellClass}>ADX</th>
              <th scope="col" className={headerCellClass}>+DI</th>
              <th scope="col" className={headerCellClass}>-DI</th>
              <th scope="col" className={headerCellClass}>%K</th>
              <th scope="col" className={headerCellClass}>%D</th>
              <th scope="col" className={`${headerCellClass} min-w-[150px]`}>Latest OHLCV</th>
              <th scope="col" className={`${headerCellClass} min-w-[200px]`}>Signals</th>
            </tr>
          </thead>
          <tbody className="bg-slate-800 divide-y divide-slate-700">
            {results.map((result) => (
              <tr 
                key={result.ticker} 
                className={`
                  ${result.error ? 'bg-red-900 bg-opacity-20' : ''} 
                  ${selectedTickerForChart === result.ticker ? 'bg-sky-700 bg-opacity-30' : 'hover:bg-slate-700'}
                  transition-colors group cursor-pointer
                `}
                onClick={() => onRowClick(result.ticker)}
              >
                <td className="px-3 py-3 whitespace-nowrap text-center">
                  <button 
                    onClick={(e) => { e.stopPropagation(); onToggleFavorite(result.ticker); }}
                    className={`text-2xl ${favoriteTickers.has(result.ticker) ? 'text-yellow-400' : 'text-slate-600 hover:text-yellow-500'} transition-colors`}
                    aria-label={favoriteTickers.has(result.ticker) ? `Unfavorite ${result.ticker}` : `Favorite ${result.ticker}`}
                  >
                    {favoriteTickers.has(result.ticker) ? '★' : '☆'}
                  </button>
                </td>
                <td className="px-3 py-3 whitespace-nowrap">
                  <div className={`text-sm font-semibold ${favoriteTickers.has(result.ticker) ? 'text-sky-300' : 'text-slate-100'}`}>{result.ticker}</div>
                  <div className="text-xs text-slate-400">{result.companyName || 'N/A'}</div>
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${getPriceColor(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close)}`}>
                  {formatPrice(livePrices[result.ticker]?.price)}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {formatPrice(result.latestPrice)}
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${
                  result.percentChange === undefined ? 'text-slate-300' :
                  result.percentChange > 0 ? 'text-green-400' : 
                  result.percentChange < 0 ? 'text-red-400' : 'text-slate-300'
                }`}>
                  {formatPercent(result.percentChange)}
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${result.latestIndicators && result.latestIndicators.adx != null && result.latestIndicators.adx > ADX_TREND_STRENGTH_THRESHOLD ? 'text-green-400 font-medium' : 'text-slate-300'}`}>
                  {result.latestIndicators?.adx != null ? result.latestIndicators.adx.toFixed(2) : 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.pdi != null ? result.latestIndicators.pdi.toFixed(2) : 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.mdi != null ? result.latestIndicators.mdi.toFixed(2) : 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.k != null ? result.latestIndicators.k.toFixed(2) : 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.d != null ? result.latestIndicators.d.toFixed(2) : 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap">
                  <OhlcvDisplay ohlcv={result.latestOhlcvDataPoint} />
                </td>
                <td className="px-3 py-3 whitespace-normal text-sm max-w-xs">
                  <div className="flex gap-2 flex-wrap">
                    {result.error ? (
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSignalColor('', result.error)}`}>
                        Error
                      </span>
                    ) : result.statusTags.length > 0 ? (
                      result.statusTags.map(tag => (
                        <span key={tag} className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSignalColor(tag)}`}>
                          {tag}
                        </span>
                      ))
                    ) : (
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSignalColor('')}`}>
                        -
                      </span>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div ref={tableRef} className="h-10"></div>
    </div>
  );
};