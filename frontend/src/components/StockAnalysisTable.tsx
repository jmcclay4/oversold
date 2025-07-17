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

const getSignalStyle = (tag: string, error?: string): React.CSSProperties => {
  if (error) return { backgroundColor: '#FF7373', color: '#D1D5DB' };
  if (tag === 'DMI') return { backgroundColor: '#3CBABA', color: '#D1D5DB' };
  if (tag === 'ADX') return { backgroundColor: '#3CBABA', color: '#D1D5DB' };
  if (tag === 'STO') return { backgroundColor: '#FF7373', color: '#D1D5DB' };
  return { backgroundColor: '#333333', color: '#D1D5DB' };
};

const getPriceStyle = (price: number | null | undefined, close: number | null | undefined): React.CSSProperties => {
  if (price == null || close == null) return { color: '#6B7280' };
  const diff = ((price - close) / close) * 100;
  if (diff > 0.5) return { color: 'green' };
  if (diff < -0.5) return { color: 'red' };
  return { color: '#D1D5DB' };
};

const formatPrice = (price?: number | null) => price != null ? price.toFixed(2) : '-';
const formatPercent = (price?: number | null, close?: number | null) => {
  if (price == null || close == null) return '-';
  return `${((price - close) / close * 100).toFixed(2)}%`;
};

const OhlcvDisplay: React.FC<{ ohlcv?: OHLCV }> = ({ ohlcv }) => {
  if (!ohlcv) return <span style={{ color: '#6B7280' }}>-</span>;
  return (
    <div style={{ fontSize: '12px', color: '#6B7280' }}>
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
    return <p style={{ textAlign: 'center', color: '#6B7280', padding: '16px' }}>No analysis results to display.</p>;
  }
  
  const headerCellStyle: React.CSSProperties = { padding: '12px 12px 0 12px', textAlign: 'left', fontSize: '12px', fontWeight: 500, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', position: 'sticky' as 'sticky', top: 0, zIndex: 10, backgroundColor: '#1a1a1a' };
  const updateCellStyle: React.CSSProperties = { padding: '4px 12px', fontSize: '10px', textAlign: 'left', fontWeight: 500, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 0, backgroundColor: '#1a1a1a' };

  return (
    <div style={{ backgroundColor: '#1a1a1a', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', borderRadius: '12px', overflow: 'hidden', marginTop: '24px' }}>
      <div style={{ overflowX: 'auto', maxHeight: '60vh' }}>
        <table style={{ minWidth: '100%', borderCollapse: 'separate', borderSpacing: 0 }}>
          <thead style={{ backgroundColor: '#1a1a1a' }}>
            <tr>
              <th scope="col" style={{...headerCellStyle, textAlign: 'center'}}>Fav</th>
              <th scope="col" style={headerCellStyle}>Ticker</th>
              <th scope="col" style={{...headerCellStyle, backgroundColor: '#1a1a1a', display: 'flex', justifyContent: 'flex-start', alignItems: 'center' }}>
                <button
                  onClick={(e) => { e.stopPropagation(); onRefreshLivePrices(); }}
                  style={{ padding: '2px 4px', backgroundColor: '#1a1a1a', color: '#D1D5DB', fontSize: '10px', borderRadius: '4px', cursor: 'pointer' }}
                  aria-label="Refresh live prices"
                >
                  PRICE ⟳
                </button>
              </th>
              <th scope="col" style={{...headerCellStyle, backgroundColor: '#1a1a1a'}}>∆</th>
              <th scope="col" style={headerCellStyle}>Close</th>
              <th scope="col" style={headerCellStyle}>ADX</th>
              <th scope="col" style={headerCellStyle}>+DI</th>
              <th scope="col" style={headerCellStyle}>-DI</th>
              <th scope="col" style={headerCellStyle}>%K</th>
              <th scope="col" style={headerCellStyle}>%D</th>
              <th scope="col" style={{...headerCellStyle, minWidth: '150px'}}>Latest OHLCV</th>
              <th scope="col" style={{...headerCellStyle, minWidth: '200px'}}>Signals</th>
            </tr>
            <tr style={{ backgroundColor: '#1a1a1a' }}>
              <td style={{...updateCellStyle, textAlign: 'center'}}></td>
              <td style={updateCellStyle}></td>
              <td style={{...updateCellStyle, backgroundColor: '#1a1a1a', color: '#6B7280'}}>{lastLiveUpdate || '-'}</td>
              <td style={{...updateCellStyle, backgroundColor: '#1a1a1a', color: '#6B7280'}}></td>
              <td style={updateCellStyle}>{lastOhlcvUpdate || '-'}</td>
              <td style={updateCellStyle}></td>
              <td style={updateCellStyle}></td>
              <td style={updateCellStyle}></td>
              <td style={updateCellStyle}></td>
              <td style={updateCellStyle}></td>
              <td style={{...updateCellStyle, minWidth: '150px'}}></td>
              <td style={{...updateCellStyle, minWidth: '200px'}}></td>
            </tr>
          </thead>
          <tbody style={{ backgroundColor: '#1a1a1a', borderTop: '1px solid #333333' }}>
            {results.map((result) => (
              <tr 
                key={result.ticker} 
                style={{
                  backgroundColor: result.error ? 'rgba(255, 0, 0, 0.1)' : '',
                  transition: 'background-color 0.3s',
                  cursor: 'pointer',
                }}
                onClick={() => onRowClick(result.ticker)}
              >
                <td style={{ padding: '12px', whiteSpace: 'nowrap', textAlign: 'center' }}>
                  <button 
                    onClick={(e) => { e.stopPropagation(); onToggleFavorite(result.ticker); onRefreshLivePrices();}}
                    style={{ fontSize: '24px', color: favoriteTickers.has(result.ticker) ? '#E5E7EB' : '#6B7280', transition: 'color 0.3s' }}
                    aria-label={favoriteTickers.has(result.ticker) ? `Unfavorite ${result.ticker}` : `Favorite ${result.ticker}`}
                  >
                    {favoriteTickers.has(result.ticker) ? '★' : '☆'}
                  </button>
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap' }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: favoriteTickers.has(result.ticker) ? '#D1D5DB' : '#D1D5DB' }}>{result.ticker}</div>
                  <div style={{ fontSize: '12px', color: '#6B7280' }}>{result.companyName || '-'}</div>
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', backgroundColor: '#1a1a1a', ...getPriceStyle(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close) }}>
                  {formatPrice(livePrices[result.ticker]?.price)}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', backgroundColor: '#1a1a1a', color: '#D1D5DB' }}>
                  {formatPercent(livePrices[result.ticker]?.price, result.latestOhlcvDataPoint?.close)}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: '#D1D5DB' }}>
                  {formatPrice(result.latestOhlcvDataPoint?.close)}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: result.latestIndicators && result.latestIndicators.adx != null && result.latestIndicators.adx > ADX_TREND_STRENGTH_THRESHOLD ? 'green' : '#D1D5DB' }}>
                  {result.latestIndicators?.adx != null ? result.latestIndicators.adx.toFixed(2) : '-'}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: '#D1D5DB' }}>
                  {result.latestIndicators?.pdi != null ? result.latestIndicators.pdi.toFixed(2) : '-'}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: '#D1D5DB' }}>
                  {result.latestIndicators?.mdi != null ? result.latestIndicators.mdi.toFixed(2) : '-'}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: '#D1D5DB' }}>
                  {result.latestIndicators?.k != null ? result.latestIndicators.k.toFixed(2) : '-'}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap', fontSize: '14px', color: '#D1D5DB' }}>
                  {result.latestIndicators?.d != null ? result.latestIndicators.d.toFixed(2) : '-'}
                </td>
                <td style={{ padding: '12px', whiteSpace: 'nowrap' }}>
                  <OhlcvDisplay ohlcv={result.latestOhlcvDataPoint} />
                </td>
                <td style={{ padding: '12px', whiteSpace: 'normal', fontSize: '14px', maxWidth: '10rem' }}>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {result.error ? (
                      <span style={{ padding: '4px 8px', display: 'inline-flex', fontSize: '12px', lineHeight: '20px', fontWeight: '600', borderRadius: '9999px', backgroundColor: '#FF7373', color: '#D1D5DB' }}>
                        Error
                      </span>
                    ) : result.statusTags.length > 0 ? (
                      result.statusTags.map(tag => (
                        <span key={tag} style={{ padding: '4px 8px', display: 'inline-flex', fontSize: '12px', lineHeight: '20px', fontWeight: '600', borderRadius: '9999px', ...getSignalStyle(tag) }}>
                          {tag}
                        </span>
                      ))
                    ) : (
                      <span style={{ padding: '4px 8px', display: 'inline-flex', fontSize: '12px', lineHeight: '20px', fontWeight: '600', borderRadius: '9999px', backgroundColor: '#333333', color: '#D1D5DB' }}>
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
