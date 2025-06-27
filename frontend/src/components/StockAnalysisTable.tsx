import React from 'react';
import { StockAnalysisResult, StockAnalysisTableProps, LivePrice } from '../types';

const getSignalColor = (tag: string, error?: string): string => {
  if (error) return 'bg-red-700 text-red-100';
  if (tag === 'Oversold' || tag === 'Overbought') return 'bg-purple-500 text-purple-100';
  if (tag === 'Trending') return 'bg-blue-500 text-blue-100';
  if (tag === 'Bullish' || tag === 'Bearish') return 'bg-teal-500 text-teal-100';
  return 'bg-yellow-700 text-yellow-100';
};

const getPriceColor = (price?: number, close?: number): string => {
  if (!price || !close) return 'text-slate-500';
  const diff = ((price - close) / close) * 100;
  if (diff > 0.5) return 'text-green-400';
  if (diff < -0.5) return 'text-red-400';
  return 'text-slate-300';
};

const formatPrice = (price?: number) => price?.toFixed(2) ?? 'N/A';

export const StockAnalysisTable: React.FC<StockAnalysisTableProps> = ({
  results,
  favoriteTickers,
  onToggleFavorite,
  onRowClick,
  selectedTickerForChart,
  livePrices
}) => {
  const headerCellClass = "px-3 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider sticky top-0 z-10 bg-slate-900";

  if (results.length === 0) {
    return <p className="text-center text-slate-400 py-4">No analysis results to display.</p>;
  }

  return (
    <div className="bg-slate-800 shadow-2xl rounded-xl overflow-hidden mt-6">
      <div className="overflow-x-auto max-h-[60vh]">
        <table className="min-w-full divide-y divide-slate-700">
          <thead className="bg-slate-900">
            <tr>
              <th scope="col" className={`${headerCellClass} text-center`}>Fav</th>
              <th scope="col" className={headerCellClass}>Ticker</th>
              <th scope="col" className={headerCellClass}>Company</th>
              <th scope="col" className={headerCellClass}>Price</th>
              <th scope="col" className={headerCellClass}>ADX</th>
              <th scope="col" className={headerCellClass}>+DI</th>
              <th scope="col" className={headerCellClass}>-DI</th>
              <th scope="col" className={headerCellClass}>%K</th>
              <th scope="col" className={headerCellClass}>%D</th>
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
                  transition-colors cursor-pointer
                `}
                onClick={() => onRowClick(result.ticker)}
              >
                <td className="px-3 py-3 whitespace-nowrap text-center">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleFavorite(result.ticker);
                    }}
                    className={`text-2xl ${favoriteTickers.has(result.ticker) ? 'text-yellow-400' : 'text-slate-600 hover:text-yellow-500'} transition-colors`}
                    aria-label={favoriteTickers.has(result.ticker) ? `Unfavorite ${result.ticker}` : `Favorite ${result.ticker}`}
                  >
                    {favoriteTickers.has(result.ticker) ? '★' : '☆'}
                  </button>
                </td>
                <td className="px-3 py-3 whitespace-nowrap">
                  <div className={`text-sm font-semibold ${favoriteTickers.has(result.ticker) ? 'text-sky-300' : 'text-slate-100'}`}>
                    {result.ticker}
                  </div>
                  <div className="text-xs text-slate-400">{result.companyName || 'N/A'}</div>
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.companyName || 'N/A'}
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${getPriceColor(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close)}`}>
                  {formatPrice(livePrices[result.ticker]?.price)}
                </td>
                <td className={`px-3 py-3 whitespace-nowrap text-sm ${result.latestIndicators?.adx && result.latestIndicators.adx > 25 ? 'text-green-400 font-medium' : 'text-slate-300'}`}>
                  {result.latestIndicators?.adx?.toFixed(2) || 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.pdi?.toFixed(2) || 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.mdi?.toFixed(2) || 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.k?.toFixed(2) || 'N/A'}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-sm text-slate-300">
                  {result.latestIndicators?.d?.toFixed(2) || 'N/A'}
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