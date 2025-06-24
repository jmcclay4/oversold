# Stock Technical Analyzer Dashboard - README

## 1. Application Overview

This application is a frontend technical analysis tool designed to help users identify potential trading signals in stocks based on the ADX (Average Directional Index) and DMI (Directional Movement Index) indicators.

**Core Functionality:**

*   **Tracked Stocks:** The application automatically loads and analyzes a predefined list of "tracked" stock tickers upon startup.
*   **Data Simulation:** It simulates fetching historical OHLCV (Open, High, Low, Close, Volume) data for these stocks. In a real-world scenario, this would be replaced by a live data feed (e.g., from Yahoo Finance, Alpha Vantage, or a paid provider) via a backend service.
*   **Local Indicator Calculation:** ADX, +DI (PDI), and -DI (MDI) indicators are calculated locally in the browser using the fetched OHLCV data.
*   **Signal Identification:** The application identifies stocks that meet specific criteria:
    *   A strong trend (ADX above a defined threshold).
    *   A bullish DMI setup (+DI crossing above -DI, +DI already above -DI, or +DI nearing -DI from below).
*   **Interactive Charting:** Users can click on any stock in the results table to view an interactive chart displaying:
    *   The stock's historical closing prices.
    *   Overlaid lines for its ADX, +DI, and -DI values over time.
*   **Table Display:** Analysis results are presented in a paginated table with columns for:
    *   Ticker symbol
    *   Latest Price
    *   Day's Percentage Change
    *   Latest ADX, +DI, -DI values
    *   Latest OHLCV data point
    *   Signal status message (e.g., "Meets Criteria," "No Signal," "Error")
*   **Sorting & Filtering:**
    *   The table is automatically sorted: Favorites first, then by signal status (Meets Criteria > No Signal > Error), and finally alphabetically by ticker.
    *   Users can search for stocks by ticker.
    *   Users can filter the table by signal status and by their favorited stocks.
*   **Favoriting:** Users can mark stocks as "favorites," and this preference is saved in the browser's local storage.

## 2. File Structure and Purpose

```
.
├── public/
│   └── (Static assets, if any, not currently used extensively)
├── src/
│   ├── components/
│   │   ├── App.tsx                    # Main application component, orchestrates state and UI.
│   │   ├── ErrorMessage.tsx           # Component to display error messages.
│   │   ├── LoadingSpinner.tsx         # Component for loading animations.
│   │   ├── StockAnalysisTable.tsx     # Component to display stock analysis results in a table.
│   │   └── StockChart.tsx             # Component for rendering stock price and indicator charts using Chart.js.
│   │
│   ├── services/
│   │   └── stockDataService.ts        # Handles fetching (simulated) stock data and performing core analysis logic.
│   │
│   ├── utils/
│   │   └── indicatorCalculations.ts   # Contains functions for calculating ADX, DMI, and other technical indicators.
│   │
│   ├── App.css                      # (If global non-Tailwind CSS is needed - currently minimal)
│   ├── constants.ts                 # Global constants (e.g., indicator periods, API settings, tracked stock list).
│   ├── index.css                    # (Global CSS, often used by Tailwind setup - not explicitly listed but assumed part of Tailwind)
│   ├── index.tsx                    # Entry point for the React application.
│   └── types.ts                     # TypeScript type definitions and interfaces for the application.
│
├── .env                             # (Not committed) For environment variables like API keys (though current version simulates data).
├── .gitignore                     # Specifies intentionally untracked files that Git should ignore.
├── index.html                     # Main HTML file, loads Tailwind CSS, Chart.js, and the React app.
├── metadata.json                  # Metadata for the application (name, description, permissions).
├── package.json                   # Project metadata, dependencies, and scripts.
├── postcss.config.js              # Configuration for PostCSS (used by Tailwind CSS).
├── readme.txt                     # This file.
├── tailwind.config.js             # Configuration for Tailwind CSS.
└── tsconfig.json                  # TypeScript compiler configuration.
```

**Key File Explanations:**

*   **`index.html`**: The main page that the browser loads. It includes CDN links for Tailwind CSS and Chart.js, sets up the import map for ES modules, and contains the root `<div>` where the React application is mounted.
*   **`index.tsx`**: The JavaScript/TypeScript entry point. It imports the main `App` component and renders it into the `root` div in `index.html`.
*   **`App.tsx`**: The heart of the application. It manages:
    *   Overall application state (list of all analyzed stocks, currently displayed stocks, loading/error states, selected stock for chart, filter criteria, favorites).
    *   Fetching initial stock data via `stockDataService.ts`.
    *   Implementing filtering, sorting, and pagination logic.
    *   Rendering the `StockChart` and `StockAnalysisTable` components.
*   **`stockDataService.ts`**:
    *   Simulates fetching OHLCV data for stock tickers (would be replaced by actual API calls in a production app).
    *   For each stock, it calculates derived data like latest price, percentage change.
    *   Calls `indicatorCalculations.ts` to get ADX/DMI values.
    *   Determines if a stock meets the predefined signal criteria.
    *   Formats the data into `StockAnalysisResult` objects.
*   **`indicatorCalculations.ts`**: Contains the mathematical logic for calculating ADX, +DI, and -DI from OHLCV data. Uses Wilder's Smoothing.
*   **`StockChart.tsx`**: Uses the `Chart.js` library to render line charts for a selected stock's price and its ADX/DMI indicators.
*   **`StockAnalysisTable.tsx`**: Renders the main table of stock analysis results, including interactive elements like favoriting and row selection for the chart.
*   **`constants.ts`**: Centralizes application-wide constants, such as DMI/ADX periods, trend strength thresholds, the list of tracked stocks (`TOP_N_TICKERS_LIST`), and pagination settings.
*   **`types.ts`**: Defines TypeScript interfaces for data structures used throughout the application (e.g., `OHLCV`, `StockAnalysisResult`, `IndicatorValues`).

## 3. How It Works - Data Flow & Logic

1.  **Initialization (`App.tsx`, `index.tsx`):**
    *   The application starts, and `App.tsx` is mounted.
    *   An initial loading state is set.
    *   Favorites are loaded from `localStorage`.

2.  **Data Fetching & Analysis (Initial Load in `App.tsx` -> `stockDataService.ts`):**
    *   `App.tsx` calls `analyzeTrackedStocks` from `stockDataService.ts`, passing the `TOP_N_TICKERS_LIST` from `constants.ts`.
    *   `analyzeTrackedStocks` iterates through each ticker:
        *   It calls `fetchStockData` (in `stockDataService.ts`) which *simulates* fetching historical OHLCV data for the ticker.
        *   If data is successfully "fetched":
            *   It calculates the latest price and daily percentage change.
            *   It calls `calculateADXDMI` (from `utils/indicatorCalculations.ts`) with the OHLCV data to get the historical series for ADX, +DI, and -DI.
            *   It extracts the latest indicator values.
            *   It determines if the stock `meetsCriteria` based on ADX strength and DMI crossover/proximity rules (defined in `constants.ts`).
            *   It constructs a `StockAnalysisResult` object containing all this information, including the historical data needed for charting.
        *   If an error occurs or no data is found, an appropriate error `StockAnalysisResult` is created.
    *   The `App.tsx` component receives the array of `StockAnalysisResult` objects and updates its state (`allAnalysisResults`). The loading state is then set to false.

3.  **UI Rendering & Updates (`App.tsx` and child components):**
    *   Based on `allAnalysisResults` and current filter/search/pagination/favorite states, `App.tsx` derives `displayedAnalysisResults`.
    *   **Chart (`StockChart.tsx`):**
        *   If a `selectedTickerForChart` is set (by clicking a table row), the corresponding `StockAnalysisResult` is passed to `StockChart.tsx`.
        *   `StockChart.tsx` uses `Chart.js` to render the historical close prices, ADX, +DI, and -DI from the provided stock data.
        *   If no stock is selected, or if data is insufficient, it displays a placeholder message.
    *   **Table (`StockAnalysisTable.tsx`):**
        *   Receives `displayedAnalysisResults`, favorite tickers, and callbacks.
        *   Renders the table rows, applying appropriate styling for signal status, price changes, and favorites.
        *   Handles click events for favoriting (updates state in `App.tsx`) and row selection (updates `selectedTickerForChart` in `App.tsx`).
    *   **Controls (in `App.tsx`):**
        *   Search input, filter dropdowns, and pagination buttons update their respective state variables in `App.tsx`.
        *   Changes to these states trigger a re-calculation of `displayedAnalysisResults`, which in turn re-renders the table.

4.  **User Interactions:**
    *   **Selecting a Stock for Chart:** Clicking a table row updates `selectedTickerForChart` in `App.tsx`, causing `StockChart.tsx` to re-render with the new stock's data.
    *   **Favoriting:** Clicking a star icon updates the `favoriteTickers` set in `App.tsx`, which is persisted to `localStorage` and affects the table's sort order and filtering.
    *   **Searching/Filtering:** Modifying search terms or filter criteria updates state in `App.tsx`, leading to a re-filtered and re-sorted list of displayed stocks and resets pagination to page 1.
    *   **Pagination:** Clicking "Next" or "Previous" updates the `currentPage` state in `App.tsx`, causing the table to display a different slice of the (filtered and sorted) data.

This setup allows for a responsive user interface where technical analysis and visualization are performed directly in the user's browser using simulated data.
