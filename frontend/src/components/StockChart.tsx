import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import AnnotationPlugin from 'chartjs-plugin-annotation';
import { StockAnalysisResult } from '../types';
import { useState } from 'react';

ChartJS.defaults.color = '#D1D5DB';
ChartJS.defaults.font.family = 'Inter, sans-serif';
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, AnnotationPlugin);

interface StockChartProps {
  stockData: StockAnalysisResult | null;
}

export const StockChart: React.FC<StockChartProps> = ({ stockData }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!stockData || !stockData.historicalDates || !stockData.historicalClosePrices) {
    return <div className="text-slate-400 text-center">No chart data available.</div>;
  }

  const dates = stockData.historicalDates;
  const minDate = dates[0];
  const maxDate = dates[dates.length - 1];

  const adxDmiChartData = {
    labels: dates,
    datasets: [
      {
        label: 'Close Price',
        data: stockData.historicalClosePrices,
        borderColor: '#f59e0b',
        fill: false,
        yAxisID: 'y-price',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: 'ADX',
        data: stockData.historicalAdx || [],
        borderColor: '#a855f7',
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '+DI',
        data: stockData.historicalPdi || [],
        borderColor: '#3b82f6',
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '-DI',
        data: stockData.historicalMdi || [],
        borderColor: '#ef4444',
        fill: false,
        yAxisID: 'y-indicators',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
    ],
  };

  const stochasticChartData = {
    labels: dates,
    datasets: [
      {
        label: '%K',
        data: stockData.historicalK || [],
        borderColor: '#3b82f6',
        fill: false,
        yAxisID: 'y-stochastic',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: '%D',
        data: stockData.historicalD || [],
        borderColor: '#ef4444',
        fill: false,
        yAxisID: 'y-stochastic',
        pointRadius: 0,
        pointHoverRadius: 5,
      },
    ],
  };

  const xAxisConfig = {
    type: 'category' as const,
    min: minDate,
    max: maxDate,
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
      color: 'rgb(51, 65, 85)', // slate-700
    },
    ticks: {
      stepSize: 20,
      callback: (value: number) => value.toFixed(0),
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
      <div className="mb-4" style={{ height: '400px', width: '100%' }}>
        <Line
          data={adxDmiChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            onHover: (event, elements) => {
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
                },
                grid: {
                  ...axisConfig.grid,
                  drawOnChartArea: false, // Only right axis draws grid
                },
              },
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
      <div style={{ height: '200px', width: '100%' }}>
        <Line
          data={stochasticChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            onHover: (event, elements) => {
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
              'y-d': {
                ...axisConfig,
                position: 'right' as const,
                min: 0,
                max: 100,
                title: { display: true, text: ' ' },
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