# OVERSOLD - Stock Analyzer Dashboard

**OVERSOLD** is a web-based stock analysis dashboard that provides technical analysis for stocks using indicators such as ADX, DMI, and Stochastic oscillators. The dashboard allows users to view stock data, track favorite stocks, and receive alerts based on predefined technical conditions. It is designed for educational purposes and is not intended for financial advice.

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Technologies Used](#technologies-used)
4. [Frontend and Backend Interaction](#frontend-and-backend-interaction)
5. [Setup and Installation](#setup-and-installation)
6. [Running the Project](#running-the-project)
7. [Known Issues and Improvements](#known-issues-and-improvements)

---

## Project Overview

**OVERSOLD** is a stock analysis tool that:
- Displays technical indicators (ADX, DMI, Stochastic) for S&P 500 stocks and user-added tickers.
- Provides visual alerts (status tags) for specific conditions:
  - **DMI**: When DMI+ crosses above DMI- or is nearing a crossover.
  - **ADX**: When ADX ≥ 25, indicating a strong trend.
  - **STO**: When Stochastic %K or %D ≤ 20 or %K is nearing %D.
- Allows users to favorite stocks, add custom tickers, and view interactive charts.
- Uses a dark-themed, modern UI with a custom logo and responsive design.

### Target Audience
- Beginner to intermediate stock traders or enthusiasts interested in technical analysis.
- Developers or students learning about full-stack development with React and FastAPI.

---

## File Structure

The project is divided into two main directories: `frontend` and `backend`. Below is a detailed breakdown of each directory and the purpose of each file.

### Frontend (`/frontend`)

- **`/src`**:
  - **`App.tsx`**: The main React component that renders the entire application. It handles state management for stock data, favorites, filters, and the selected stock for charting.
  - **`main.tsx`**: Entry point for the React application. Renders `App.tsx` into the DOM.
  - **`/components`**:
    - **`StockAnalysisTable.tsx`**: Renders the table of stock data with columns for favorites, ticker, price, indicators, and status tags. Handles row clicks to select stocks for charting.
    - **`StockChart.tsx`**: Renders interactive charts for the selected stock using Chart.js. Displays historical close prices and technical indicators.
    - **`LoadingSpinner.tsx`**: A simple loading spinner component shown while data is being fetched.
    - **`ErrorMessage.tsx`**: Displays error messages when data fetching or analysis fails.
  - **`/services`**:
    - **`stockDataService.ts`**: Contains functions to fetch stock data from the backend, analyze stocks, and calculate technical indicators (ADX, DMI, Stochastic).
  - **`/types`**:
    - **`types.ts`**: Defines TypeScript interfaces for stock data, indicators, and filter criteria.
  - **`/constants`**:
    - **`constants.ts`**: Holds constants like `TOP_N_TICKERS_LIST`, thresholds for indicators, and API delay settings.
  - **`/utils`**:
    - **`indicatorCalculations.ts`**: Utility functions for calculating ADX, DMI, and Stochastic indicators.

- **`/public`**:
  - **`index.html`**: The main HTML file that loads the React application. Includes links to Google Fonts for custom typography.

- **`/package.json`**: Manages frontend dependencies and scripts for building and running the React application.

### Backend (`/backend`)

- **`main.py`**: The entry point for the FastAPI backend. Defines API endpoints for fetching stock data, refreshing analysis, and checking cache status. Handles database initialization and caching.
- **`backfill_company_names.py`**: A utility script to backfill company names for stocks using `yfinance`.
- **`requirements.txt`**: Lists Python dependencies for the backend, including `fastapi`, `uvicorn`, `yfinance`, `pandas`, and `tqdm`.
- **`stocks.db`**: SQLite database file storing OHLCV data and cached analysis results.

---

## Technologies Used

### Frontend
- **React**: JavaScript library for building user interfaces.
- **TypeScript**: Typed superset of JavaScript for better code quality and maintainability.
- **Chart.js**: For rendering interactive stock charts.
- **Tailwind CSS**: Utility-first CSS framework for styling.
- **Vite**: Build tool for faster development and production builds.

### Backend
- **FastAPI**: Modern Python web framework for building APIs.
- **SQLite**: Lightweight database for storing stock data and analysis cache.
- **yfinance**: Python library to fetch stock data from Yahoo Finance.
- **pandas**: Data manipulation library for handling stock data.
- **uvicorn**: ASGI server for running the FastAPI application.

---

## Frontend and Backend Interaction

The frontend communicates with the backend via RESTful API endpoints provided by FastAPI. Below are the key endpoints and their purposes:

- **`GET /stocks/{ticker}`**: Fetches cached or live stock data for a given ticker. Returns OHLCV data and company name.
- **`GET /refresh_analysis/{ticker}`**: Refreshes the stock data for a ticker by fetching only the missing dates from Yahoo Finance.
- **`GET /cache_status`**: Returns the timestamp of the latest cache refresh.

### Data Flow
1. The frontend requests stock data for multiple tickers via `analyzeTrackedStocks`, which calls `/stocks/{ticker}` for each ticker.
2. If cached data is valid (within 24 hours), it is served from the `analysis_cache` table.
3. If the cache is invalid or missing, the backend fetches fresh data from `yfinance`, stores it in `ohlcv`, and caches the result.
4. The frontend processes the data to calculate technical indicators and display them in the table and charts.

---

## Setup and Installation

### Prerequisites
- **Node.js** (v16 or later) and **npm** for the frontend.
- **Python** (v3.8 or later) and **pip** for the backend.
- **SQLite** (pre-installed on most systems).

### Backend Setup
1. Navigate to the backend directory:
   ```
   cd /path/to/stock-analyzer/backend
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Initialize the database:
   - Run `python3 main.py` to start the server and initialize the database with example tickers (MMM, AOS, ABT).
5. (Optional) Run the backfill script to update company names:
   ```
   python3 backfill_company_names.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd /path/to/stock-analyzer/frontend
   ```
2. Install dependencies:
   ```
   npm install
   ```
3. Ensure Google Fonts are linked in `/public/index.html` for the custom font (Special Gothic Expanded One).

---

## Running the Project

### Backend
- Start the FastAPI server:
  ```
  cd /path/to/stock-analyzer/backend
  python3 main.py
  ```
- The server will run on `http://0.0.0.0:8000`.

### Frontend
- Start the React development server:
  ```
  cd /path/to/stock-analyzer/frontend
  npm run dev
  ```
- The frontend will be available at `http://localhost:5173`.

### Interacting with the Application
- **View Stocks**: The dashboard displays a table of stocks with technical indicators and status tags.
- **Add/Remove Stocks**: Use the "Manage Stocks" menu to add or remove custom tickers.
- **Favorites**: Click the star icon to favorite stocks, which are saved in `localStorage`.
- **Refresh Data**: Click "Refresh Data" to update stock data for all tickers.

---

## Known Issues and Improvements

### Known Issues
1. **401 Errors when Fetching Company Names**: Some tickers fail to retrieve company names due to `yfinance` API restrictions. Retries and user-agent headers are in place, but some tickers may still fail.
2. **Slow Cache Retrieval**: SQLite performance can be slow for large datasets. Consider migrating to a faster database (e.g., PostgreSQL) or in-memory cache (e.g., Redis) for production.
3. **Chart Rendering**: Ensure Chart.js is correctly installed and configured. If charts are missing, check for errors in `StockChart.tsx` or missing dependencies.

### Suggested Improvements
1. **Server-Side Persistence for Favorites and Custom Tickers**: Currently stored in `localStorage`; implement user accounts or session-based storage for cross-device persistence.
2. **Batch API Requests**: Optimize frontend requests by batching multiple tickers in a single API call.
3. **Unit Tests**: Add tests for critical components (`calculateStochastic`, `calculateADXDMI`, API endpoints) using Jest and pytest.
4. **UI/UX Enhancements**: Improve mobile responsiveness, add tooltips for status tags, and enhance chart interactivity.
5. **Deployment**: Deploy the frontend to Vercel/Netlify and the backend to Render/Heroku with a managed database.

---

This README provides a comprehensive overview of the **OVERSOLD** stock analyzer project, including its purpose, file structure, technologies, and setup instructions. It serves as a detailed guide for understanding the codebase and will enable an AI service (or any developer) to assist with improving the code and adding new features.