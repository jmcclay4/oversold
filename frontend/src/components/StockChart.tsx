import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, LineController } from 'chart.js';
import AnnotationPlugin from 'chartjs-plugin-annotation';
import 'chartjs-chart-financial';
import { CandlestickController, CandlestickElement } from 'chartjs-chart-financial';
import { Chart } from 'react-chartjs-2';
import { StockAnalysisResult } from '../types';
import { useState } from 'react';
import { DateTime } from 'luxon';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, AnnotationPlugin, LineController, CandlestickController, CandlestickElement);

interface StockChartProps {
  stockData: StockAnalysisResult | null;
}

export const StockChart: React.FC<StockChartProps> = ({ stockData }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'1m' | '3m' | '6m'>('6m');

  if (!stockData || !stockData.historicalDates || !stockData.historicalClosePrices) {
    return <div style={{color: '#D1D5DB', textAlign: 'center'}}>No chart data available.</div>;
  }

  // Filter data based on selected period
  const now = DateTime.now();
  const minDate = now.minus({ months: selectedPeriod === '1m' ? 1 : selectedPeriod === '3m' ? 3 : 6 }).toISODate();

  const filteredIndices = stockData.historicalDates.map((date, index) => (date >= minDate ? index : -1)).filter(i => i >= 0);

  if (filteredIndices.length === 0) {
    return <div style={{color: '#D1D5DB', textAlign: 'center'}}>No chart data available for selected period.</div>;
  }

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

  // Calculate min/max for price y-axis
  const priceMin = Math.min(...filteredLow) * 0.95;
  const priceMax = Math.max(...filteredHigh) * 1.05;

  // Dynamic barPercentage based on selected period
  const barPercentage = selectedPeriod === '1m' ? 0.4 : selectedPeriod === '3m' ? 0.3 : 0.2;

  const priceChartData = {
    labels: filteredDates,
    datasets: [
      {
        type: 'candlestick' as const,
        label: 'Price',
        data: filteredDates.map((date, idx) => ({
          x: date,
          o: filteredOpen[idx],
          h: filteredHigh[idx],
          l: filteredLow[idx],
          c: filteredClose[idx],
        })),
        borderColor: 'transparent',
        color: {
          up: 'green',
          down: 'red',
        },
        barPercentage: barPercentage,
        categoryPercentage: barPercentage, // Adjust categoryPercentage similarly for consistency
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
        borderColor: '#3CBABA',
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '-DI',
        data: filteredMdi,
        borderColor: '#FF7373',
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: 'ADX',
        data: filteredAdx,
        borderColor: '#FFFFFF',
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
        borderColor: '#3CBABA',
        borderWidth: 1,
        fill: false,
        yAxisID: 'y-stochastic',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '%D',
        data: filteredD,
        borderColor: '#FF7373',
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
      color: '#333333',
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
    <div style={{ width: '100%', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Price Chart with Overlay Selector */}
      <div style={{ marginBottom: '16px', position: 'relative', height: '300px', width: '100%' }}>
        <div style={{ position: 'absolute', top: '8px', left: '50%', transform: 'translateX(-50%)', zIndex: 10, color: '#D1D5DB', fontSize: '14px', fontWeight: '200' }}>
          <span style={{ fontWeight: 'bold' }}>{stockData.ticker || 'Ticker'}</span> {stockData.companyName || 'Full Stock Name'}
        </div>
        <div style={{ position: 'absolute', top: '8px', right: '8px', zIndex: 10, fontSize: '10px' }}>
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as '1m' | '3m' | '6m')}
            style={{ padding: '4px', backgroundColor: '#1a1a1a', color: '#D1D5DB', borderRadius: '4px', border: '1px solid #1a1a1a' }}
          >
            <option value="1m">1 MONTH</option>
            <option value="3m">3 MONTHS</option>
            <option value="6m">6 MONTHS</option>
          </select>
        </div>
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
                min: priceMin,
                max: priceMax,
                grid: {
                  ...axisConfig.grid,
                  drawOnChartArea: false,
                },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: '#1a1a1a',
                titleColor: '#D1D5DB',
                bodyColor: '#D1D5DB',
              },
              legend: {
                display: false,
              },
              annotation: {
                annotations: hoverLine ? [hoverLine] : [],
              },
            },
          }}
        />
      </div>

      {/* DMI/ADX Chart */}
      <div style={{ marginBottom: '16px', height: '250px', width: '100%' }}>
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
                position: 'left' as const,
                min: 0,
                max: 100,
                title: { display: true, text: 'Indicators' },
                grid: {
                  ...axisConfig.grid,
                  drawOnChartArea: true,
                },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: '#1a1a1a',
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
      <div style={{ height: '250px', width: '100%' }}>
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
                  drawOnChartArea: true,
                },
              },
            },
            plugins: {
              tooltip: {
                enabled: true,
                backgroundColor: '#1a1a1a',
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