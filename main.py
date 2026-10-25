"""API de velas e indicadores Binance Spot. Fechas de entrada en UTC (YYYY-MM-DD o ISO-8601)."""
from datetime import datetime, timedelta, timezone
import re
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import pandas_ta_classic as ta
import requests

app = FastAPI(title="Trading Indicators API", description="API de análisis técnico con datos de Binance.", version="2.1.0")
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVALOS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
COLUMNAS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
NUMERICAS = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
WARMUP = 200


def validar_parametros(symbol: str, interval: str, fecha_inicio: Optional[str], fecha_fin: Optional[str]):
    symbol = symbol.upper().strip()
    if not re.fullmatch(r"[A-Z0-9]{3,20}", symbol):
        raise HTTPException(status_code=400, detail="Símbolo inválido")
    if interval not in INTERVALOS:
        raise HTTPException(status_code=400, detail="Intervalo no soportado por Binance")
    if bool(fecha_inicio) != bool(fecha_fin):
        raise HTTPException(status_code=400, detail="Debe proporcionar fecha_inicio y fecha_fin conjuntamente")
    inicio = parsear_fecha(fecha_inicio) if fecha_inicio else None
    fin = parsear_fecha(fecha_fin, fin_dia=True) if fecha_fin else None
    if inicio and (inicio >= fin):
        raise HTTPException(status_code=400, detail="fecha_inicio debe ser anterior a fecha_fin")
    if inicio and inicio > datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="fecha_inicio no puede estar en el futuro")
    return symbol, inicio, fin


def parsear_fecha(valor: str, fin_dia: bool = False) -> datetime:
    try:
        # Fechas YYYY-MM-DD representan el día completo para fecha_fin.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", valor):
            fecha = datetime.strptime(valor, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return fecha + timedelta(days=1) if fin_dia else fecha
        fecha = datetime.fromisoformat(valor.replace("Z", "+00:00"))
        return fecha.replace(tzinfo=timezone.utc) if fecha.tzinfo is None else fecha.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Fecha inválida: {valor}. Use YYYY-MM-DD o ISO-8601")


def consultar_binance(symbol: str, interval: str, limit: int, inicio_ms: Optional[int] = None, fin_ms: Optional[int] = None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if inicio_ms is not None:
        params["startTime"] = inicio_ms
    if fin_ms is not None:
        params["endTime"] = fin_ms
    try:
        respuesta = requests.get(BINANCE_URL, params=params, timeout=20)
        respuesta.raise_for_status()
        velas = respuesta.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Binance: {exc}")
    if not isinstance(velas, list) or not velas:
        raise HTTPException(status_code=502, detail="Respuesta de Binance inválida")
    return velas


def descargar_velas(symbol: str, interval: str, periodos: int, inicio: Optional[datetime], fin: Optional[datetime], warmup: int):
    if inicio is None:
        return consultar_binance(symbol, interval, min(periodos + warmup, 1000))

    inicio_ms = int(inicio.timestamp() * 1000)
    fin_ms = min(int(fin.timestamp() * 1000), int(datetime.now(timezone.utc).timestamp() * 1000))
    if fin_ms <= inicio_ms:
        raise HTTPException(status_code=422, detail="No hay velas cerradas en el rango solicitado")

    # Para los indicadores, obtener las 200 velas anteriores a fecha_inicio.
    anteriores = []
    if warmup:
        anteriores = consultar_binance(symbol, interval, warmup, fin_ms=inicio_ms - 1)

    # Consulta ascendente, paginada por tiempo. No se limita a 1000 velas en total.
    registros = []
    cursor = inicio_ms
    while cursor < fin_ms:
        bloque = consultar_binance(symbol, interval, 1000, inicio_ms=cursor, fin_ms=fin_ms - 1)
        registros.extend(bloque)
        siguiente = int(bloque[-1][0]) + 1
        if siguiente <= cursor:
            raise HTTPException(status_code=502, detail="Paginación de Binance sin avance")
        cursor = siguiente
        if len(bloque) < 1000:
            break

    # periodos representa máximo de velas de salida; se conservan antecedentes.
    # Se depura y recorta después para evitar descartar datos válidos.
    return anteriores + registros


def depurar_velas(velas: list):
    validas = [v[:12] for v in velas if isinstance(v, list) and len(v) >= 12]
    if not validas:
        raise HTTPException(status_code=422, detail="Binance no devolvió velas válidas")
    df = pd.DataFrame(validas, columns=COLUMNAS)
    original_count = len(df)
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERICAS).sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    now_ms = pd.Timestamp.now(tz="UTC").value // 10**6
    df = df[(df["close_time"] < now_ms) & (df["close_time"] > df["open_time"])].copy()
    valid = ((df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) &
             (df["volume"] >= 0) & (df["high"] >= df["low"]) & (df["high"] >= df["open"]) &
             (df["high"] >= df["close"]) & (df["low"] <= df["open"]) & (df["low"] <= df["close"]))
    df = df[valid].reset_index(drop=True)
    # Las columnas secundarias se convierten para no fallar ante datos no numéricos.
    for col in ["quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, original_count


def calcular_indicadores(df: pd.DataFrame):
    df = df.copy()
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_histogram"] = macd["MACDh_12_26_9"]
    adx = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=14)
    df["adx14"] = adx["ADX_14"]
    df["atr14"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14, mamode="rma")
    df["atr_percent"] = (df["atr14"] / df["close"]) * 100
    return df


def numero(value, decimales: int = 8):
    return None if pd.isna(value) else round(float(value), decimales)


def formatear_registros(df: pd.DataFrame, indicadores: bool):
    registros = []
    for _, row in df.iterrows():
        item = {
            "open_date": pd.to_datetime(row["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
            "open": numero(row["open"]), "high": numero(row["high"]),
            "low": numero(row["low"]), "close": numero(row["close"]),
            "volume": numero(row["volume"]), "quote_volume": numero(row["quote_volume"]),
            "trades": numero(row["trades"]), "taker_buy_base": numero(row["taker_buy_base"]),
            "taker_buy_quote": numero(row["taker_buy_quote"]),
        }
        if indicadores:
            item.update({
                "ema20": numero(row["ema20"]), "ema50": numero(row["ema50"]),
                "rsi14": numero(row["rsi14"], 4), "macd": numero(row["macd"]),
                "macd_signal": numero(row["macd_signal"]), "macd_histogram": numero(row["macd_histogram"]),
                "adx14": numero(row["adx14"], 4), "atr14": numero(row["atr14"]),
                "atr_percent": numero(row["atr_percent"], 4),
                "pg": numero(row["close"]) - numero(row["open"]),
            })
        registros.append(item)
    return registros


def ejecutar_consulta(symbol: str, interval: str, periodos: int, fecha_inicio: Optional[str], fecha_fin: Optional[str], indicadores: bool):
    symbol, inicio, fin = validar_parametros(symbol, interval, fecha_inicio, fecha_fin)
    velas = descargar_velas(symbol, interval, periodos, inicio, fin, WARMUP if indicadores else 0)
    df, total_original = depurar_velas(velas)
    total_validas = len(df)

    if indicadores and total_validas < 51:
        raise HTTPException(status_code=422, detail="Se requieren más de 50 velas válidas")

    if inicio is not None:
        # El filtrado por fecha se hace después del cálculo para conservar warmup.
        desde, hasta = int(inicio.timestamp() * 1000), int(fin.timestamp() * 1000)
        salida = df[(df["open_time"] >= desde) & (df["open_time"] < hasta)]
    else:
        salida = df

    if len(salida) < periodos:
        # Con fechas explícitas, periodos es tope máximo, no cantidad obligatoria.
        if inicio is None:
            raise HTTPException(status_code=422, detail=f"Solo hay {len(salida)} velas válidas; se solicitaron {periodos}")
        if salida.empty:
            raise HTTPException(status_code=422, detail="No hay velas válidas en el rango solicitado")

    if indicadores:
        df = calcular_indicadores(df)
        if inicio is not None:
            salida = df[(df["open_time"] >= desde) & (df["open_time"] < hasta)]
        else:
            salida = df

    salida = salida.tail(periodos) if inicio is None else salida.head(periodos)
    registros = formatear_registros(salida, indicadores)
    return {
        "symbol": symbol, "timeframe": interval, "periodos": periodos, "timezone": "UTC",
        "velas_recibidas": total_original, "velas_validas": total_validas,
        "velas_eliminadas": total_original - total_validas,
        "total_registros": len(registros), "registros": registros,
    }


@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/indicadores", tags=["Análisis Técnico"], summary="Indicadores actuales (compatible con versión 2.0)")
def indicadores(symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
                interval: str = Query("5m"), periodos: int = Query(10, gt=1, le=800)):
    return ejecutar_consulta(symbol, interval, periodos, None, None, True)


@app.get("/datos", tags=["Mercado"], summary="Velas históricas OHLCV")
def datos(symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
          fecha_inicio: Optional[str] = Query(None, description="YYYY-MM-DD o ISO-8601, UTC"),
          fecha_fin: Optional[str] = Query(None, description="YYYY-MM-DD inclusiva o ISO-8601, UTC"),
          interval: str = Query("15m"), periodos: int = Query(100, gt=1, le=10000)):
    return ejecutar_consulta(symbol, interval, periodos, fecha_inicio, fecha_fin, False)


@app.get("/indicadores2", tags=["Análisis Técnico"], summary="Velas históricas con indicadores")
def indicadores2(symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
                 fecha_inicio: Optional[str] = Query(None, description="YYYY-MM-DD o ISO-8601, UTC"),
                 fecha_fin: Optional[str] = Query(None, description="YYYY-MM-DD inclusiva o ISO-8601, UTC"),
                 interval: str = Query("15m"), periodos: int = Query(100, gt=1, le=10000)):
    return ejecutar_consulta(symbol, interval, periodos, fecha_inicio, fecha_fin, True)
