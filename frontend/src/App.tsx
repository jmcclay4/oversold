import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { StockAnalysisTable } from './components/StockAnalysisTable';
import { StockChart } from './components/StockChart';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { analyzeTrackedStocks, fetchAllTickers, fetchMetadata, analyzeStockTicker, fetchLivePrices } from './services/stockDataService';
import { StockAnalysisResult, FilterCriteria, LivePrice, StockAnalysisTableProps } from './types';

const App: React.FC = () => {
  console.log('App component rendering...');
  const [allAnalysisResults, setAllAnalysisResults] = useState<StockAnalysisResult[]>([]);
  const [displayedAnalysisResults, setDisplayedAnalysisResults] = useState<StockAnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [filterCriteria, setFilterCriteria] = useState<FilterCriteria>({ searchTerm: '' });
  const [favoriteTickers, setFavoriteTickers] = useState<Set<string>>(new Set());
  const [selectedTickerForChart, setSelectedTickerForChart] = useState<string | null>(null);
  const [customTickers, setCustomTickers] = useState<string[]>([]);
  const [showMenu, setShowMenu] = useState(false);
  const [newTicker, setNewTicker] = useState('');
  const [lastOhlcvUpdate, setLastOhlcvUpdate] = useState<string | null>(null);
  const [lastLiveUpdate, setLastLiveUpdate] = useState<string | null>(null);
  const [livePrices, setLivePrices] = useState<Record<string, LivePrice>>({});

  useEffect(() => {
    console.log('Loading stored favorites and custom tickers...');
    const storedFavorites = localStorage.getItem('favoriteStockTickers');
    if (storedFavorites) {
      setFavoriteTickers(new Set(JSON.parse(storedFavorites)));
    }
    const storedCustomTickers = localStorage.getItem('customTickers');
    if (storedCustomTickers) {
      setCustomTickers(JSON.parse(storedCustomTickers));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('favoriteStockTickers', JSON.stringify(Array.from(favoriteTickers)));
  }, [favoriteTickers]);

  useEffect(() => {
    localStorage.setItem('customTickers', JSON.stringify(customTickers));
  }, [customTickers]);

  const toggleFavorite = useCallback((ticker: string) => {
    setFavoriteTickers(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(ticker) ) {
        newFavorites.delete(ticker);
      } else {
        if (newFavorites.size >= 10) {
          alert('Maximum 10 favorite tickers allowed.');
          return prev;
        }
        newFavorites.add(ticker);
      }
      return newFavorites;
    });
    fetchLivePricesForFavorites(); // Fetch live prices when favorites change
  }, []);

  const handleSelectStockForChart = useCallback(async (ticker: string) => {
    console.log('Selected ticker for chart:', ticker);
    setSelectedTickerForChart(ticker);
    try {
      const stockData = await analyzeStockTicker(ticker);
      if (!stockData.error) {
        setAllAnalysisResults(prev => {
          const updated = prev.map(r => r.ticker === ticker ? stockData : r);
          return updated;
        });
      } else {
        console.error(`Error fetching chart data for ${ticker}:`, stockData.error);
      }
    } catch (err) {
      console.error(`Error fetching chart data for ${ticker}:`, err);
    }
  }, []);

  const addTicker = async () => {
    const ticker = newTicker.trim().toUpperCase();
    if (ticker && !customTickers.includes(ticker)) {
      try {
        console.log('Adding ticker:', ticker);
        const result = await analyzeStockTicker(ticker);
        if (!result.error) {
          setCustomTickers(prev => [...prev, ticker]);
          setNewTicker('');
        } else {
          alert(`Invalid ticker: ${ticker}`);
        }
      } catch {
        alert(`Error adding ticker: ${ticker}`);
      }
    }
  };

  const removeTicker = (ticker: string) => {
    setCustomTickers(prev => prev.filter(t => t !== ticker));
    if (selectedTickerForChart === ticker) {
      setSelectedTickerForChart(null);
    }
  };

  const fetchLivePricesForFavorites = async () => {
    if (favoriteTickers.size === 0) return;
    const favoriteTickerList = Array.from(favoriteTickers);
    console.log(`Fetching live prices for favorite tickers: ${favoriteTickerList}`);
    try {
      const livePricesData = await fetchLivePrices(favoriteTickerList);
      const livePricesMap = livePricesData.reduce((acc, item) => {
        acc[item.ticker] = item;
        return acc;
      }, {} as Record<string, LivePrice>);
      setLivePrices(livePricesMap);
      // Set latest live update time (most recent timestamp in EST)
      const latestTimestamp = livePricesData
        .map(p => p.timestamp)
        .filter((t): t is string => t != null)
        .sort((a, b) => new Date(b).getTime() - new Date(a).getTime())[0];
      if (latestTimestamp) {
        const date = new Date(latestTimestamp);
        setLastLiveUpdate(date.toLocaleString('en-US', {
          timeZone: 'America/Chicago',
          hour: '2-digit',
          minute: '2-digit',
          hour12: true
        }));
      } else {
        setLastLiveUpdate(null);
      }
    } catch (error) {
      console.error(`Error fetching live prices: ${(error as Error).message}`);
      setLivePrices(prev => {
        const newPrices = { ...prev };
        favoriteTickerList.forEach(ticker => {
          newPrices[ticker] = { ticker, price: null, timestamp: null, volume: null };
        });
        return newPrices;
      });
      setLastLiveUpdate(null);
    }
  };

  const fetchInitialData = async () => {
    console.log('Loading initial data...');
    setIsLoading(true);
    setGlobalError(null);
    try {
      const tickers = await fetchAllTickers();
      if (tickers.length === 0) {
        throw new Error('No tickers available');
      }
      console.log('Fetched tickers:', tickers.length);
      const allTickers = [...new Set([...tickers, ...customTickers])];
      const results = await analyzeTrackedStocks(allTickers);
      console.log('Loaded stocks:', results.length, results.slice(0, 3));
      setAllAnalysisResults(results);
      const metadata = await fetchMetadata();
      // Format last_ohlcv_update to MM/DD
      setLastOhlcvUpdate(metadata.last_ohlcv_update ? metadata.last_ohlcv_update.split(' ')[0].replace(/(\d{4})-(\d{2})-(\d{2})/, '$2/$3') : null);
      await fetchLivePricesForFavorites(); // Fetch live prices on initial load
    } catch (error) {
      const errMsg = `Error analyzing stocks: ${(error as Error).message}`;
      console.error(errMsg);
      setGlobalError(errMsg);
      setAllAnalysisResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchInitialData();
  }, [customTickers]);

  useEffect(() => {
    console.log('Filtering results, allAnalysisResults length:', allAnalysisResults.length);
    let filtered = [...allAnalysisResults];

    const searchTerm = filterCriteria.searchTerm?.trim();
    if (searchTerm) {
      filtered = filtered.filter(stock =>
        stock.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (stock.companyName && stock.companyName.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    filtered.sort((a, b) => {
      const aIsFav = favoriteTickers.has(a.ticker);
      const bIsFav = favoriteTickers.has(b.ticker);
      if (aIsFav && !bIsFav) return -1;
      if (!aIsFav && bIsFav) return 1;

      const aTagCount = a.error ? -1 : a.statusTags.length;
      const bTagCount = b.error ? -1 : b.statusTags.length;
      if (aTagCount !== bTagCount) {
        return bTagCount - aTagCount;
      }

      return a.ticker.localeCompare(b.ticker);
    });
    
    setDisplayedAnalysisResults(filtered);
  }, [allAnalysisResults, filterCriteria, favoriteTickers]);

  const selectedStockDataForChart = useMemo(() => {
    if (!selectedTickerForChart) return null;
    const stock = allAnalysisResults.find(r => r.ticker === selectedTickerForChart);
    console.log('Chart data for:', selectedTickerForChart, stock);
    return stock || null;
  }, [selectedTickerForChart, allAnalysisResults]);

  const stockAnalysisTableProps: StockAnalysisTableProps = {
    results: displayedAnalysisResults,
    favoriteTickers,
    onToggleFavorite: toggleFavorite,
    onRowClick: handleSelectStockForChart,
    selectedTickerForChart,
    livePrices,
    lastOhlcvUpdate,
    lastLiveUpdate,
    onRefreshLivePrices: fetchLivePricesForFavorites
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0f0f0f', color: '#D1D5DB', padding: '16px', fontSize: '16px', display: 'flex', flexDirection: 'column' }}>
      <header style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '20px', fontWeight: '800', color: 'transparent', backgroundClip: 'text', backgroundImage: 'linear-gradient(to right, #9CA3AF, #6B7280, #4B5563)', textTransform: 'uppercase', textAlign: 'left', letterSpacing: '0.1em', fontFamily: '"Special Gothic Expanded One", sans-serif' }}>
          OVERSOLD
        </h1>
        <div style={{ textAlign: 'right', fontSize: '12px', color: '#6B7280' }}>
          {/* Timestamps removed */}
        </div>
      </header>

      <main style={{ flexGrow: 1, maxWidth: '100%', margin: '0 auto', width: '100%', paddingLeft: '8px', paddingRight: '8px' }}>
        <div style={{ marginBottom: '24px', backgroundColor: '#1a1a1a', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', borderRadius: '12px', padding: '16px' }}>
          <StockChart stockData={selectedStockDataForChart} />
        </div>

        <div style={{ margin: '16px 0', padding: '16px', backgroundColor: '#1a1a1a', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', borderRadius: '12px', display: 'flex', flexDirection: 'row', gap: '16px', alignItems: 'center', justifyContent: 'space-between' }}>
          <input
            type="text"
            placeholder="Search by ticker or company..."
            style={{ flexGrow: 1, padding: '8px', border: '1px solid #333333', backgroundColor: '#1a1a1a', color: '#D1D5DB', borderRadius: '8px', outline: 'none', width: '100%' }}
            value={filterCriteria.searchTerm ?? ''}
            onChange={(e) => setFilterCriteria({ searchTerm: e.target.value })}
          />
          <button
            onClick={() => setShowMenu(!showMenu)}
            style={{ padding: '8px 16px', backgroundColor: '#1a1a1a', color: '#D1D5DB', borderRadius: '8px', cursor: 'pointer' }}
          >
            Manage Stocks
          </button>
          <button
            onClick={fetchInitialData}
            style={{ padding: '8px 16px', backgroundColor: '#333333', color: '#D1D5DB', borderRadius: '8px', cursor: 'pointer' }}
            aria-label="Refresh data"
          >
            Refresh
          </button>
        </div>

        {showMenu && (
          <div style={{ marginBottom: '16px', padding: '16px', backgroundColor: '#1a1a1a', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', borderRadius: '12px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: '#D1D5DB', marginBottom: '8px' }}>Manage Stocks</h2>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <input
                type="text"
                placeholder="Add ticker (e.g., TSLA)"
                style={{ flexGrow: 1, padding: '8px', border: '1px solid #333333', backgroundColor: '#1a1a1a', color: '#D1D5DB', borderRadius: '8px' }}
                value={newTicker}
                onChange={(e) => setNewTicker(e.target.value)}
              />
              <button
                onClick={addTicker}
                style={{ padding: '8px 16px', backgroundColor: '#333333', color: '#D1D5DB', borderRadius: '8px' }}
              >
                Add
              </button>
            </div>
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {customTickers.map(ticker => (
                <div key={ticker} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #333333' }}>
                  <span style={{ color: '#D1D5DB' }}>{ticker}</span>
                  <button
                    onClick={() => removeTicker(ticker)}
                    style={{ color: '#FF7373', cursor: 'pointer' }}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {globalError && <ErrorMessage message={globalError} />}
        {isLoading && allAnalysisResults.length === 0 && <LoadingSpinner />}
        
        {!isLoading && allAnalysisResults.length === 0 && !globalError && (
          <div style={{ textAlign: 'center', padding: '40px 24px', backgroundColor: '#1a1a1a', borderRadius: '12px' }}>
            <svg style={{ margin: '0 auto', height: '48px', width: '48px', color: '#6B7280' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2z" />
            </svg>
            <h3 style={{ marginTop: '8px', fontSize: '18px', fontWeight: 'medium', color: '#D1D5DB' }}>No Data Loaded</h3>
            <p style={{ marginTop: '4px', fontSize: '14px', color: '#6B7280' }}>
              The application automatically loads tracked stocks.
            </p>
          </div>
        )}

        {!isLoading && allAnalysisResults.length > 0 && (
          <StockAnalysisTable {...stockAnalysisTableProps} />
        )}
        
      </main>
      <footer style={{ textAlign: 'center', marginTop: '32px', padding: '16px 0', borderTop: '1px solid #333333' }}>
        <p style={{ fontSize: '14px', color: '#6B7280' }}>
          OVERSOLD Dashboard. For educational purposes. Not financial advice.
        </p>
        <p style={{ fontSize: '12px', color: '#4B5563', marginTop: '4px' }}>
          Data provided by Yahoo Finance and Alpaca Markets.
        </p>
      </footer>
    </div>
  );
};

export default App;
