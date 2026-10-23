"""API para indicadores de velas Binance. Ejecución: uvicorn main:app --reload."""
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Indicadores BTC Binance', version='1.0.0', description='Calcula EMA20, EMA50, RSI14 y MACD(12,26,9) y retorna las últimas diez velas.')

class DatosRequest(BaseModel):
    candles: list[Any] = Field(..., description='Velas Binance como arreglos o diccionarios con claves 0..11')
    symbol: str = 'BTCUSDT'
    interval: str = '5m'
    only_closed: bool = True


def ema_series(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    alpha = 2 / (period + 1)
    for i in range(period, len(values)):
        current = (values[i] - current) * alpha + current
        result[i] = current
    return result


def rsi_series(values: list[float], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    gain = sum(max(x, 0) for x in diffs[:period]) / period
    loss = sum(max(-x, 0) for x in diffs[:period]) / period
    def convert(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100 - 100 / (1 + g / l)
    result[period] = convert(gain, loss)
    for i in range(period + 1, len(values)):
        change = diffs[i - 1]
        gain = (gain * (period - 1) + max(change, 0)) / period
        loss = (loss * (period - 1) + max(-change, 0)) / period
        result[i] = convert(gain, loss)
    return result


def macd_series(values: list[float]):
    fast = ema_series(values, 12)
    slow = ema_series(values, 26)
    line: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        if fast[i] is not None and slow[i] is not None:
            line[i] = fast[i] - slow[i]
    valid = [(i, x) for i, x in enumerate(line) if x is not None]
    signal: list[float | None] = [None] * len(values)
    if len(valid) >= 9:
        alpha = 2 / 10
        s = sum(x for _, x in valid[:9]) / 9
        signal[valid[8][0]] = s
        for idx, x in valid[9:]:
            s = (x - s) * alpha + s
            signal[idx] = s
    hist = [None if a is None or b is None else a - b for a, b in zip(line, signal)]
    return line, signal, hist


def read_candle(record: Any) -> dict:
    if isinstance(record, dict):
        if 'json' in record and isinstance(record['json'], (list, dict)):
            record = record['json']
        if isinstance(record, dict):
            if 'data' in record and isinstance(record['data'], list):
                record = record['data']
            elif '0' in record or 0 in record:
                record = [record.get(str(i), record.get(i)) for i in range(12)]
            elif 'open_time' in record:
                try:
                    return {'open_time': int(record['open_time']), 'close_time': int(record['close_time']), 'open': float(record['open']), 'high': float(record['high']), 'low': float(record['low']), 'close': float(record['close']), 'volume': float(record['volume'])}
                except (KeyError, TypeError, ValueError) as ex:
                    raise ValueError(f'Campos OHLCV incorrectos: {ex}') from ex
    if not isinstance(record, (list, tuple)) or len(record) < 7:
        raise ValueError('Cada vela debe ser un arreglo kline de al menos 7 posiciones o un objeto equivalente')
    try:
        return {'open_time': int(record[0]), 'open': float(record[1]), 'high': float(record[2]), 'low': float(record[3]), 'close': float(record[4]), 'volume': float(record[5]), 'close_time': int(record[6])}
    except (TypeError, ValueError) as ex:
        raise ValueError(f'Valores de vela inválidos: {ex}') from ex


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/datos')
def datos(request: DatosRequest):
    try:
        candles = [read_candle(c) for c in request.candles]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if request.only_closed:
            candles = [c for c in candles if c['close_time'] < now_ms]
        candles.sort(key=lambda c: c['open_time'])
        # Exigir velas distintas: duplicados distorsionan indicadores.
        times = [c['open_time'] for c in candles]
        if len(times) != len(set(times)):
            raise ValueError('Hay velas duplicadas por open_time')
        if any(c['high'] < max(c['open'], c['close'], c['low']) or c['low'] > min(c['open'], c['close'], c['high']) or c['volume'] < 0 for c in candles):
            raise ValueError('Datos OHLCV inconsistentes')
        if len(candles) < 50:
            raise ValueError('Se requieren al menos 50 velas cerradas para calcular indicadores estables')
        prices = [c['close'] for c in candles]
        e20 = ema_series(prices, 20)
        e50 = ema_series(prices, 50)
        r14 = rsi_series(prices, 14)
        macd, signal, hist = macd_series(prices)
        def rounded(v, decimals=8):
            return None if v is None else round(v, decimals)
        output = []
        for i in range(max(0, len(candles) - 10), len(candles)):
            c = candles[i]
            output.append({
                'open_time': c['open_time'],
                'close_time': c['close_time'],
                'fecha_cierre_utc': datetime.fromtimestamp(c['close_time'] / 1000, timezone.utc).isoformat(),
                'close': c['close'],
                'ema20': rounded(e20[i]),
                'ema50': rounded(e50[i]),
                'rsi14': rounded(r14[i], 3),
                'macd': rounded(macd[i]),
                'macd_signal': rounded(signal[i]),
                'macd_histogram': rounded(hist[i]),
            })
        return {'symbol': request.symbol, 'interval': request.interval, 'velas_procesadas': len(candles), 'registros': output}
    except (ValueError, OverflowError) as ex:
        raise HTTPException(status_code=422, detail=str(ex)) from ex
