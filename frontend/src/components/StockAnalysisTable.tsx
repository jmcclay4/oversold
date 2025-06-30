
import React from 'react';
import { StockAnalysisResult, OHLCV, LivePrice } from '../types';
import { ADX_TREND_STRENGTH_THRESHOLD } from '../constants';

interface StockAnalysisTableProps {
  results: StockAnalysisResult[];
  favoriteTickers: Set<string>;
  onToggleFavorite: (ticker: string) => void;
  onRowClick: (ticker: string) => void;
  selectedTickerForChart?: string | null;
  livePrices: Record<string, LivePrice>;
  lastOhlcvUpdate: string | null;
  lastLiveUpdate: string | null;
  onRefreshLivePrices: () => void;
}

const getSignalColor = (tag: string, error?: string): string => {
  if (error) return 'bg-red-700 text-red-100';
  if (tag === 'DMI') return 'bg-teal-500 text-teal-100';
  if (tag === 'ADX') return 'bg-blue-500 text-blue-100';
  if (tag === 'STO') return 'bg-purple-500 text-purple-100';
  return 'bg-yellow-700 text-yellow-100';
};

const getPriceColor = (price: number | null | undefined, close: number | null | undefined): string => {
  if (price == null || close == null) return 'text-slate-500';
  const diff = ((price - close) / close) * 100;
  if (diff > 0.5) return 'text-green-400';
  if (diff < -0.5) return 'text-red-400';
  return 'text-slate-300';
};

const formatPrice = (price?: number | null) => price != null ? price.toFixed(2) : '-';
const formatPercent = (price?: number | null, close?: number | null) => {
  if (price == null || close == null) return '-';
  return `${((price - close) / close * 100).toFixed(2)}%`;
};

const OhlcvDisplay: React.FC<{ ohlcv?: OHLCV }> = ({ ohlcv }) => {
  if (!ohlcv) return <span className="text-slate-500">-</span>;
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
  selectedTickerForChart,
  livePrices,
  lastOhlcvUpdate,
  lastLiveUpdate,
  onRefreshLivePrices
}) => {
  if (results.length === 0) {
    return <p className="text-center text-slate-400 py-4">No analysis results to display.</p>;
  }
  
  const headerCellClass = "px-3 py-3 pb-0 text-left text-xs font-medium text-slate-300 uppercase tracking-wider sticky top-0 z-10 bg-slate-900";
  const updateCellClass = "px-3 py-1 text-[10px] text-left font-medium text-slate-300 uppercase tracking-wider mt-0 bg-slate-900";

  return (
    <div className="bg-slate-800 shadow-2xl rounded-xl overflow-hidden mt-6">
      <div className="overflow-x-auto max-h-[60vh]">
        <table className="min-w-full divide-y divide-slate-700">
          <thead className="bg-slate-900">
            <tr>
              <th scope="col" className={`${headerCellClass} text-center`}>Fav</th>
              <th scope="col" className={headerCellClass}>Ticker</th>
              <th scope="col" className={`${headerCellClass} bg-slate-700 flex justify-start items-center`}>
                <button
                  onClick={(e) => { e.stopPropagation(); onRefreshLivePrices(); }}
                  className="px-1 py-0.5 bg-slate-800 text-slate-300 text-sx rounded hover:bg-slate-600"
                  aria-label="Refresh live prices"
                >
                  PRICE ⟳
                </button>
              </th>
              <th scope="col" className={`${headerCellClass} bg-slate-700`}>∆</th>
              <th scope="col" className={headerCellClass}>Close</th>
              <th scope="col" className={headerCellClass}>ADX</th>
              <th scope="col" className={headerCellClass}>+DI</th>
              <th scope="col" className={headerCellClass}>-DI</th>
              <th scope="col" className={headerCellClass}>%K</th>
              <th scope="col" className={headerCellClass}>%D</th>
              <th scope="col" className={`${headerCellClass} min-w-[150px]`}>Latest OHLCV</th>
              <th scope="col" className={`${headerCellClass} min-w-[200px]`}>Signals</th>
            </tr>
            <tr className="bg-slate-900">
              <td className={`${updateCellClass} text-center`}></td>
              <td className={updateCellClass}></td>
              <td className={`${updateCellClass} bg-slate-700 text-slate-400`}>{lastLiveUpdate || '-'}</td>
              <td className={`${updateCellClass} bg-slate-700 text-slate-400`}></td>
              <td className={updateCellClass}>{lastOhlcvUpdate || '-'}</td>
              <td className={updateCellClass}></td>
              <td className={updateCellClass}></td>
              <td className={updateCellClass}></td>
              <td className={updateCellClass}></td>
              <td className={updateCellClass}></td>
              <td className={`${updateCellClass} min-w-[150px]`}></td>
              <td className={`${updateCellClass} min-w-[200px]`}></td>
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
                  <div className="text-xs text-slate-400">{result.companyName || '-'}</div>
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm bg-slate-700 ${getPriceColor(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close)}`}>
                  {formatPrice(livePrices[result.ticker]?.price)}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm bg-slate-700 text-slate-300">
                  {formatPercent(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close)}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {formatPrice(result.latestOhlcvDataPoint?.close)}
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${result.latestIndicators && result.latestIndicators.adx != null && result.latestIndicators.adx > ADX_TREND_STRENGTH_THRESHOLD ? 'text-green-400 font-medium' : 'text-slate-300'}`}>
                  {result.latestIndicators?.adx != null ? result.latestIndicators.adx.toFixed(2) : '-'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.pdi != null ? result.latestIndicators.pdi.toFixed(2) : '-'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.mdi != null ? result.latestIndicators.mdi.toFixed(2) : '-'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.k != null ? result.latestIndicators.k.toFixed(2) : '-'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.d != null ? result.latestIndicators.d.toFixed(2) : '-'}
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
    </div>
  );
};
