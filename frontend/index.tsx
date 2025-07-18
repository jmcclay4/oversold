import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './src/App';

console.log('index.tsx: Rendering App...');

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);