import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import AnnotationPlugin from 'chartjs-plugin-annotation';
import 'chartjs-chart-financial'; // Import the financial chart plugin
import { Chart } from 'react-chartjs-2';
import { StockAnalysisResult } from '../types';
import { useState } from 'react';
import { DateTime } from 'luxon';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, AnnotationPlugin);

ChartJS.defaults.color = '#D1D5DB';
ChartJS.defaults.font.family = 'Inter, sans-serif';

interface StockChartProps {
  stockData: StockAnalysisResult | null;
}

export const StockChart: React.FC<StockChartProps> = ({ stockData }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'1m' | '3m' | '6m'>('6m');

  if (!stockData || !stockData.historicalDates || !stockData.historicalClosePrices) {
    return <div className="text-gray-400 text-center">No chart data available.</div>;
  }

  // Filter data based on selected period
  const now = DateTime.now();
  const minDate = now.minus({ months: selectedPeriod === '1m' ? 1 : selectedPeriod === '3m' ? 3 : 6 }).toISODate();

  const filteredIndices = stockData.historicalDates.map((date, index) => (date >= minDate ? index : -1)).filter(i => i >= 0);

  const filteredDates = filteredIndices.map(i => stockData.historicalDates![i]);
  const filteredOpen = filteredIndices.map(i => stockData.ohlcv[i].open);
  const filteredHigh = filteredIndices.map(i => stockData.ohlcv[i].high);
  const filteredLow = filteredIndices.map(i => stockData.ohlcv[i].low);
  const filteredClose = filteredIndices.map(i => stockData.historicalClosePrices![i]);
  const filteredAdx = filteredIndices.map(i => stockData.historicalAdx![i]);
  const filteredPdi = filteredIndices.map(i => stockData.historicalPdi![i]);
  const filteredMdi = filteredIndices.map(i => stockData.historicalMdi![i]);
  const filteredK = filteredIndices.map(i => stockData.historicalK![i]);
  const filteredD = filteredIndices.map(i => stockData.historicalD![i]);

  const priceChartData = {
    labels: filteredDates,
    datasets: [
      {
        type: 'candlestick' as const,
        label: 'Price',
        data: filteredIndices.map(i => ({
          x: filteredDates[i],
          o: filteredOpen[i],
          h: filteredHigh[i],
          l: filteredLow[i],
          c: filteredClose[i],
        })),
        borderColor: 'transparent',
        color: {
          up: 'white',
          down: 'black',
        },
        yAxisID: 'y-price',
      },
    ],
  };

  const adxDmiChartData = {
    labels: filteredDates,
    datasets: [
      {
        label: '+DI',
        data: filteredPdi,
        borderColor: '#9CA3AF', // gray-400
        borderWidth: 1, // Thinner lines (50% of default 2)
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '-DI',
        data: filteredMdi,
        borderColor: '#6B7280', // gray-500
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: 'ADX',
        data: filteredAdx,
        borderColor: '#4B5563', // gray-600
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
    ],
  };

  const stochasticChartData = {
    labels: filteredDates,
    datasets: [
      {
        label: '%K',
        data: filteredK,
        borderColor: '#9CA3AF', // gray-400
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-stochastic',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '%D',
        data: filteredD,
        borderColor: '#6B7280', // gray-500
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-stochastic',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
    ],
  };

  const xAxisConfig = {
    type: 'category' as const,
    ticks: {
      autoSkip: true,
      maxTicksLimit: 10,
      align: 'center' as const,
      padding: 5,
    },
    grid: {
      display: false,
    },
  };

  const axisConfig = {
    type: 'linear' as const,
    grid: {
      drawOnChartArea: true,
      color: 'rgb(75, 85, 99)', // gray-600
    },
    ticks: {
      stepSize: 20,
      callback: (value: number | string): string => Number(value).toFixed(0),
    },
  };

  const hoverLine = hoverIndex !== null ? {
    type: 'line' as const,
    xMin: hoverIndex,
    xMax: hoverIndex,
    borderColor: 'rgba(255, 255, 255, 0.5)',
    borderWidth: 1,
    drawOnChartArea: true,
  } : undefined;

  return (
    <div className="w-full max-w-[1200px] mx-auto">
      {/* Period Selector */}
      <div className="mb-4 flex justify-end">
        <select
          value={selectedPeriod}
          onChange={(e) => setSelectedPeriod(e.target.value as '1m' | '3m' | '6m')}
          className="p-2 bg-gray-700 text-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500"
        >
          <option value="1m">1 Month</option>
          <option value="3m">3 Months</option>
          <option value="6m">6 Months</option>
        </select>
      </div>

      {/* Price Chart */}
      <div className="mb-4" style={{ height: '400px', width: '100%' }}>
        <Chart
          type="candlestick"
          data={priceChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            onHover: (event: any, elements: any[]) => {
              if (elements.length > 0) {
                setHoverIndex(elements[0].index);
              } else {
                setHoverIndex(null);
              }
            },
            scales: {
              x: xAxisConfig,
              'y-price': {
                ...axisConfig,
                position: 'left' as const,
                title: { display: true, text: 'Price' },
                ticks: {
                  ...axisConfig.ticks,
                  stepSize: undefined,
                  callback: (value: number | string): string => Number(value).toFixed(2),
                },
                grid: {
                  ...axisConfig.grid,
                  drawOnChartArea: false, // Only right axis draws grid
                },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                titleColor: '#D1D5DB',
                bodyColor: '#D1D5DB',
              },
              legend: {
                labels: {
                  color: '#D1D5DB',
                },
              },
              annotation: {
                annotations: hoverLine ? [hoverLine] : [],
              },
            },
          }}
        />
      </div>

      {/* DMI/ADX Chart */}
      <div className="mb-4" style={{ height: '200px', width: '100%' }}>
        <Chart
          type="line"
          data={adxDmiChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            onHover: (event: any, elements: any[]) => {
              if (elements.length > 0) {
                setHoverIndex(elements[0].index);
              } else {
                setHoverIndex(null);
              }
            },
            scales: {
              x: xAxisConfig,
              'y-indicators': {
                ...axisConfig,
                position: 'right' as const,
                min: 0,
                max: 100,
                title: { display: true, text: 'Indicators' },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                titleColor: '#D1D5DB',
                bodyColor: '#D1D5DB',
              },
              legend: {
                labels: {
                  color: '#D1D5DB',
                },
              },
              annotation: {
                annotations: hoverLine ? [hoverLine] : [],
              },
            },
          }}
        />
      </div>

      {/* Stochastic Chart */}
      <div style={{ height: '200px', width: '100%' }}>
        <Chart
          type="line"
          data={stochasticChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            onHover: (event: any, elements: any[]) => {
              if (elements.length > 0) {
                setHoverIndex(elements[0].index);
              } else {
                setHoverIndex(null);
              }
            },
            scales: {
              x: xAxisConfig,
              'y-stochastic': {
                ...axisConfig,
                position: 'left' as const,
                min: 0,
                max: 100,
                title: { display: true, text: 'Stochastic' },
                grid: {
                  ...axisConfig.grid,
                  drawOnChartArea: false,
                },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                titleColor: '#D1D5DB',
                bodyColor: '#D1D5DB',
              },
              legend: {
                labels: {
                  color: '#D1D5DB',
                },
              },
              annotation: {
                annotations: hoverLine ? [hoverLine] : [],
              },
            },
          }}
        />
      </div>
    </div>
  );
};

export default StockChart;
