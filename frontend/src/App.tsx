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
  const [livePriceUpdate, setLivePriceUpdate] = useState<string | null>(null);
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
      if (newFavorites.has(ticker)) {
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
      if (livePricesData.length > 0 && livePricesData[0].timestamp) {
        setLivePriceUpdate(livePricesData[0].timestamp);
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
      setLastOhlcvUpdate(metadata.last_ohlcv_update || 'Unknown');
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
    if (allAnalysisResults.length > 0 && favoriteTickers.size > 0) {
      fetchLivePricesForFavorites();
      const interval = setInterval(fetchLivePricesForFavorites, 30000); // Fetch every 30 seconds
      return () => clearInterval(interval);
    }
  }, [allAnalysisResults, favoriteTickers]);

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
    livePrices
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-gray-200 p-4 sm:p-6 flex flex-col">
      <header className="mb-6 flex justify-between items-center">
        <h1 className="text-xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-sky-400 to-cyan-500 uppercase text-left tracking-wider" style={{ fontFamily: '"Special Gothic Expanded One", sans-serif' }}>
          OVERSOLD
        </h1>
        <div className="text-right text-xs text-slate-500">
          <p>OHLCV Data: {lastOhlcvUpdate || '...'}</p>
          <p>Live Price: {livePriceUpdate || '...'}</p>
        </div>
      </header>

      <main className="flex-grow max-w-full mx-auto w-full px-2">
        <div className="mb-6 bg-slate-800 shadow-xl rounded-xl p-4">
          <StockChart stockData={selectedStockDataForChart} />
        </div>

        <div className="my-4 p-4 bg-slate-800 shadow-xl rounded-xl flex flex-col sm:flex-row gap-4 items-center justify-between">
          <input
            type="text"
            placeholder="Search by ticker or company..."
            className="flex-grow p-2 border border-slate-600 bg-slate-700 text-slate-200 rounded-lg focus:ring-2 focus:ring-sky-500 w-full sm:w-auto"
            value={filterCriteria.searchTerm ?? ''}
            onChange={(e) => setFilterCriteria({ searchTerm: e.target.value })}
          />
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-200"
          >
            Manage Stocks
          </button>
          <button
            onClick={fetchInitialData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
            aria-label="Refresh data"
          >
            Refresh
          </button>
        </div>

        {showMenu && (
          <div className="mb-4 p-4 bg-slate-800 shadow-xl rounded-xl">
            <h2 className="text-lg font-bold text-white mb-2">Manage Stocks</h2>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                placeholder="Add ticker (e.g., TSLA)"
                className="flex-grow p-2 border border-gray-600 bg-gray-700 text-white rounded-lg"
                value={newTicker}
                onChange={(e) => setNewTicker(e.target.value)}
              />
              <button
                onClick={addTicker}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
              >
                Add
              </button>
            </div>
            <div className="max-h-[200px] overflow-y-auto">
              {customTickers.map(ticker => (
                <div key={ticker} className="flex justify-between items-center py-2 border-b border-gray-600">
                  <span className="text-white">{ticker}</span>
                  <button
                    onClick={() => removeTicker(ticker)}
                    className="text-red-400 hover:text-red-300"
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
          <div className="text-center py-10 px-6 bg-gray-800 rounded-xl">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2z" />
            </svg>
            <h3 className="mt-2 text-lg font-medium text-white">No Data Loaded</h3>
            <p className="mt-1 text-sm text-gray-400">
              The application automatically loads tracked stocks.
            </p>
          </div>
        )}

        {!isLoading && allAnalysisResults.length > 0 && (
          <StockAnalysisTable {...stockAnalysisTableProps} />
        )}
        
      </main>
      <footer className="text-center mt-8 py-4 border-t border-gray-600">
        <p className="text-sm text-gray-400">
          OVERSOLD Dashboard. For educational purposes. Not financial advice.
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Data provided by Yahoo Finance and Alpaca Markets.
        </p>
      </footer>
    </div>
  );
};

export default App;