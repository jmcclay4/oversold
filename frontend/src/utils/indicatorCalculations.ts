
import { OHLCV } from '../types';
import { ADX_PERIOD, DMI_PERIOD } from '../constants';

// Wilder's Smoothing (similar to EMA but with 1/N factor)
const wildersSmoothing = (data: (number | null)[], period: number): (number | null)[] => {
  const smoothedArray: (number | null)[] = new Array(data.length).fill(null);
  if (data.length < period) return smoothedArray;

  let sum = 0;
  let validPointsForInitialAvg = 0;
  for (let i = 0; i < period; i++) {
    if (data[i] !== null && !isNaN(data[i] as number)) {
      sum += data[i] as number;
      validPointsForInitialAvg++;
    }
  }

  if (validPointsForInitialAvg < period / 2 && validPointsForInitialAvg === 0) { // Heuristic: if less than half points are valid, or no points valid
    // Not enough data to start smoothing if initial values are mostly NaN
    return smoothedArray;
  }
  
  // Find first valid starting point for SMA
  let firstValidSmaIndex = -1;
  for(let i = 0; i <= data.length - period; i++) {
    sum = 0;
    validPointsForInitialAvg = 0;
    let tempSum = 0;
    for(let k=0; k < period; k++) {
        if(data[i+k] !== null && !isNaN(data[i+k] as number)) {
            tempSum += data[i+k] as number;
            validPointsForInitialAvg++;
        }
    }
    if (validPointsForInitialAvg >= period / 2 && validPointsForInitialAvg > 0) { // Allow if at least half are valid
        smoothedArray[i + period - 1] = tempSum / validPointsForInitialAvg; // Use actual count of valid points
        firstValidSmaIndex = i + period - 1;
        break;
    }
  }

  if (firstValidSmaIndex === -1) return smoothedArray; // Cannot start smoothing

  for (let i = firstValidSmaIndex + 1; i < data.length; i++) {
    if (data[i] === null || isNaN(data[i] as number)) {
      smoothedArray[i] = smoothedArray[i-1]; // Carry forward if current data is NaN
    } else if (smoothedArray[i-1] === null || isNaN(smoothedArray[i-1] as number) ) {
        // If previous smoothed is NaN, try to restart SMA with available data for 'period' points
        // This is a complex recovery, for now, we might accept NaN propagation
        // Or, find the last non-NaN smoothed value and use that.
        // For simplicity, if prev smoothed is NaN, current might also become NaN unless we re-initialize.
        // Let's try to re-initialize if possible
        let reSum = 0;
        let reValidCount = 0;
        if (i >= period -1) {
            for (let j = 0; j < period; j++) {
                if (data[i-j] !== null && !isNaN(data[i-j] as number)) {
                    reSum += data[i-j] as number;
                    reValidCount++;
                }
            }
            if (reValidCount > 0) smoothedArray[i] = reSum / reValidCount; // Simplified re-init
            else smoothedArray[i] = null;
        } else {
            smoothedArray[i] = null;
        }

    } else {
      smoothedArray[i] = ((smoothedArray[i-1] as number) * (period - 1) + (data[i] as number)) / period;
    }
  }
  return smoothedArray;
}

export const calculateADXDMI = (
  ohlcvData: OHLCV[],
  dmiPeriod: number = DMI_PERIOD,
  adxPeriod: number = ADX_PERIOD
): { pdi: (number | null)[]; mdi: (number | null)[]; adx: (number | null)[] } => {
  const high = ohlcvData.map(d => d.high);
  const low = ohlcvData.map(d => d.low);
  const close = ohlcvData.map(d => d.close);
  const length = ohlcvData.length;

  const nullArray = () => new Array(length).fill(null);

  if (length < Math.max(dmiPeriod, adxPeriod) + 1) { // Adjusted minimum length check
    return { pdi: nullArray(), mdi: nullArray(), adx: nullArray() };
  }

  const trArray: (number | null)[] = new Array(length).fill(null);
  const pdmArray: (number | null)[] = new Array(length).fill(null); // Plus Directional Movement
  const mdmArray: (number | null)[] = new Array(length).fill(null); // Minus Directional Movement

  for (let i = 1; i < length; i++) {
    const h = high[i];
    const l = low[i];
    const pc = close[i-1]; // previous close
    const ph = high[i-1]; // previous high
    const pl = low[i-1]; // previous low

    if (h == null || l == null || pc == null || ph == null || pl == null) continue;

    trArray[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));

    const upMove = h - ph;
    const downMove = pl - l;

    pdmArray[i] = (upMove > downMove && upMove > 0) ? upMove : 0;
    mdmArray[i] = (downMove > upMove && downMove > 0) ? downMove : 0;
  }
  
  // Pass full arrays to wildersSmoothing, it will handle leading NaNs/nulls
  const smoothedTR = wildersSmoothing(trArray, dmiPeriod);
  const smoothedPDM = wildersSmoothing(pdmArray, dmiPeriod);
  const smoothedMDM = wildersSmoothing(mdmArray, dmiPeriod);
  
  const pdiArray: (number | null)[] = nullArray();
  const mdiArray: (number | null)[] = nullArray();
  const dxArray: (number | null)[] = nullArray();

  for (let i = dmiPeriod -1; i < length; i++) { // Start after initial smoothing period consideration by wilder's
    if (smoothedTR[i] === null || smoothedTR[i] === 0 || smoothedPDM[i] === null || smoothedMDM[i] === null) {
        pdiArray[i] = null;
        mdiArray[i] = null;
    } else {
        pdiArray[i] = 100 * (smoothedPDM[i]! / smoothedTR[i]!);
        mdiArray[i] = 100 * (smoothedMDM[i]! / smoothedTR[i]!);
    }
    
    if (pdiArray[i] === null || mdiArray[i] === null) {
        dxArray[i] = null;
        continue;
    }

    const sumDI = pdiArray[i]! + mdiArray[i]!;
    if (sumDI === 0) { 
        dxArray[i] = 0; // Or null, depending on how ADX handles 0 DX. Typically results in low ADX.
    } else {
        dxArray[i] = 100 * (Math.abs(pdiArray[i]! - mdiArray[i]!) / sumDI);
    }
  }

  const adxArray = wildersSmoothing(dxArray, adxPeriod);
  
  return { 
    pdi: pdiArray.map(v => v === null || isNaN(v) ? null : parseFloat(v.toFixed(2))),
    mdi: mdiArray.map(v => v === null || isNaN(v) ? null : parseFloat(v.toFixed(2))),
    adx: adxArray.map(v => v === null || isNaN(v) ? null : parseFloat(v.toFixed(2)))
  };
};