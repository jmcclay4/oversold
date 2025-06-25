# Oversold

Oversold is a web application for analyzing S&P 500 stocks based on technical indicators such as Average Directional Index (ADX), Directional Movement Index (DMI), and Stochastic Oscillator. It provides a user-friendly interface to view stock data, including price trends and technical signals, helping investors identify potential oversold or overbought conditions. The project is hosted at [https://oversold.jmcclay.com](https://oversold.jmcclay.com) and is open-source, available at [https://github.com/jmcclay4/oversold](https://github.com/jmcclay4/oversold).

## Table of Contents

- [Project Overview](#project-overview)
  - [Features](#features)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Technologies and Libraries](#technologies-and-libraries)
  - [Backend](#backend-1)
  - [Frontend](#frontend-1)
  - [Deployment](#deployment)
- [Setup and Development](#setup-and-development)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Deployment](#deployment-1)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The Oversold application fetches historical stock data for S&P 500 companies, calculates technical indicators, and stores the results in a SQLite database. The backend serves this data via a REST API, while the frontend displays it in a responsive table and interactive charts. The application is designed to be efficient, scalable, and cost-effective, leveraging free tiers of Fly.io for backend deployment and Vercel for frontend hosting.

### Features
- **Stock Analysis**: Displays daily OHLCV (Open, High, Low, Close, Volume) data for S&P 500 stocks with calculated indicators (ADX, +DI, -DI, %K, %D).
- **Batch Processing**: Optimizes data retrieval by fetching multiple tickers in a single API call.
- **Persistent Database**: Stores data in a SQLite database on a 1 GB volume for persistence.
- **Responsive UI**: A clean, modern interface with a stock table and interactive charts, styled with Tailwind CSS.
- **Real-Time Updates**: Automatically rebuilds the database daily to ensure data is current (up to June 24, 2025, due to API constraints).
- **CORS Support**: Enables seamless communication between the frontend and backend.

## How It Works

1. **Backend (Fly.io)**:
   - Fetches historical stock data using the `yfinance` library for S&P 500 tickers.
   - Calculates technical indicators (ADX, DMI, Stochastic) and stores them in `/data/stocks.db`.
   - Serves data via FastAPI endpoints (`/stocks/tickers`, `/stocks/{ticker}`, `/stocks/batch`, `/metadata`).
   - Rebuilds the database daily or on-demand via `/rebuild-db` to keep data current.
   - Hosted on Fly.io with a 1 GB persistent volume for `stocks.db`.

2. **Frontend (Vercel)**:
   - Built with React and TypeScript, displaying a table of stocks with indicators in `StockAnalysisTable.tsx`.
   - Fetches data from the backend using batch requests for efficiency.
   - Uses Chart.js for interactive stock price and indicator charts.
   - Styled with Tailwind CSS and Google Fonts (Special Gothic Expanded One for the logo).
   - Deployed on Vercel for fast, scalable hosting.

3. **Data Flow**:
   - On startup, the backend checks `stocks.db` and rebuilds it if outdated or missing.
   - The frontend fetches a list of tickers (`/stocks/tickers`), then retrieves batch data (`/stocks/batch`) for the table.
   - Users can select a stock to view detailed charts, fetching data via `/stocks/{ticker}`.

## File Structure

### Backend
```
backend/
├── Dockerfile            # Defines the Docker image for Fly.io deployment
├── fly.toml              # Fly.io configuration for deployment and volume
├── main.py               # FastAPI app with API endpoints and CORS
├── init_db.py            # Database initialization and data fetching logic
├── sp500_tickers.py      # List of S&P 500 ticker symbols
├── requirements.txt      # Python dependencies
└── /data/stocks.db       # SQLite database (generated on Fly.io)
```

- **Dockerfile**: Configures a Python 3.9-slim environment, installs dependencies, and runs `uvicorn main:app`.
- **fly.toml**: Specifies Fly.io settings, including the app name (`oversold-backend`), port (8000), and 1 GB volume for `/data`.
- **main.py**: Defines FastAPI endpoints (`/stocks/tickers`, `/stocks/{ticker}`, `/stocks/batch`, `/metadata`, `/update-db`, `/rebuild-db`), handles CORS, and manages database lifecycle.
- **init_db.py**: Initializes `stocks.db`, fetches data via `yfinance`, calculates indicators, and updates the database.
- **sp500_tickers.py**: Contains a list of S&P 500 ticker symbols as strings.
- **requirements.txt**: Lists Python dependencies (e.g., `fastapi`, `uvicorn`, `yfinance`, `pandas`, `numpy`, `sqlite3`).
- **/data/stocks.db**: SQLite database storing OHLCV data, company names, and metadata (persistent on Fly.io).

### Frontend
```
frontend/
├── public/
│   └── index.html        # HTML entry point with Google Fonts
├── src/
│   ├── App.tsx           # Main React component with header and layout
│   ├── StockAnalysisTable.tsx # Table component for stock data
│   ├── stockDataService.ts # API client for fetching backend data
│   ├── components/        # Reusable React components
│   ├── styles/            # Tailwind CSS configuration
│   └── assets/            # Static assets (e.g., favicon)
├── package.json          # Node.js dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite build configuration
└── tailwind.config.js    # Tailwind CSS configuration
```

- **index.html**: HTML template with Google Fonts (Special Gothic Expanded One) for the "OVERSOLD" logo.
- **App.tsx**: Root component with the header (styled as a logo) and routes to `StockAnalysisTable`.
- **StockAnalysisTable.tsx**: Displays a table of stocks with indicators, fetches batch data, and renders charts with Chart.js.
- **stockDataService.ts**: Handles API requests to the backend (`https://oversold-backend.fly.dev`).
- **package.json**: Lists frontend dependencies (e.g., `react`, `typescript`, `tailwindcss`, `chart.js`).
- **tsconfig.json**: Configures TypeScript for React and strict type checking.
- **vite.config.ts**: Vite configuration for fast development and production builds.
- **tailwind.config.js**: Customizes Tailwind CSS with colors, fonts, and responsive design.

## Technologies and Libraries

### Backend
- **Python 3.9**: Core programming language.
- **FastAPI**: High-performance web framework for building APIs.
- **Uvicorn**: ASGI server for running FastAPI.
- **yfinance**: Fetches historical stock data from Yahoo Finance.
- **pandas**: Data manipulation for calculating indicators.
- **numpy**: Numerical computations for indicator calculations.
- **sqlite3**: Manages the SQLite database (`stocks.db`).
- **logging**: Logs application events for debugging.

### Frontend
- **React 18**: JavaScript library for building the UI.
- **TypeScript**: Static typing for safer code.
- **Tailwind CSS**: Utility-first CSS framework for styling.
- **Chart.js**: Interactive charts for stock data visualization.
- **Vite**: Fast build tool for development and production.
- **Axios**: HTTP client for API requests (in `stockDataService.ts`).
- **Google Fonts**: Special Gothic Expanded One for the logo.

### Deployment
- **Fly.io**: Hosts the backend (`oversold-backend`) with a 1 GB persistent volume for `stocks.db`.
- **Vercel**: Hosts the frontend (`oversold.jmcclay.com`) with automatic scaling and CDN.
- **Docker**: Containerizes the backend for consistent deployment.

## Setup and Development

### Prerequisites
- **Backend**: Python 3.9, Docker, Fly.io CLI.
- **Frontend**: Node.js 18+, npm, Vercel CLI.
- **Git**: For cloning the repository.

### Local Development

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jmcclay4/oversold
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```
   - Runs the FastAPI server at `http://localhost:8000`.
   - Initializes `stocks.db` in the local directory.

3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   - Runs the React app at `http://localhost:5173`.
   - Ensure the backend is running or update `stockDataService.ts` to point to `https://oversold-backend.fly.dev`.

4. **Database Rebuild** (Optional):
   ```bash
   curl -X POST http://localhost:8000/rebuild-db
   ```
   - Populates `stocks.db` with S&P 500 data (takes ~10–15 minutes).

### Deployment

1. **Backend (Fly.io)**:
   ```bash
   cd backend
   flyctl auth login
   flyctl deploy
   ```
   - Deploys to `https://oversold-backend.fly.dev`.
   - Ensure `fly.toml` specifies the 1 GB volume:
     ```toml
     [[mounts]]
       source = "stocks_data"
       destination = "/data"
     ```

2. **Frontend (Vercel)**:
   ```bash
   cd frontend
   vercel login
   vercel
   ```
   - Deploys to `https://oversold.jmcclay.com`.
   - Configure `API_BASE_URL` in `stockDataService.ts` to `https://oversold-backend.fly.dev`.

3. **Post-Deployment**:
   - Trigger database rebuild:
     ```bash
     curl -X POST https://oversold-backend.fly.dev/rebuild-db
     ```
   - Verify endpoints:
     ```bash
     curl https://oversold-backend.fly.dev/metadata
     curl https://oversold-backend.fly.dev/stocks/ABBV
     ```

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Report issues or suggest features via [GitHub Issues](https://github.com/jmcclay4/oversold/issues).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with inspiration from stock analysis tools like TradingView.
- Thanks to Yahoo Finance for providing stock data via `yfinance`.
- Powered by Fly.io and Vercel for seamless deployment.
