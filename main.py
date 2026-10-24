
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

import pandas as pd
import pandas_ta_classic as ta
import requests
import re

app = FastAPI(
    title="Trading Indicators API",
    description="API de análisis técnico de criptomonedas utilizando datos de Binance.",
    version="2.0.0"
)


# BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

INTERVALOS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h",
    "8h", "12h", "1d", "3d",
    "1w", "1M"
}

COLUMNAS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore"
]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0"
    }


@app.get("/indicadores",tags=["Análisis Técnico"], summary="Consultar velas e indicadores técnicos", 
         description="Retorna los nPeriodos para el symbol con sus respectivos indicadores tecnicos.")
def indicadores(
    symbol: str = Query(
        default="BTCUSDT",
        min_length=3,
        max_length=20,
        description=(
            "Instrumento"
        ),
        examples=["BTCUSDT","SOlUSDT"]
    ),
    interval: str = Query(
        default="5m",
        examples=["5m","15m","30m","1h","1d"]
    ),
    periodos: int = Query(
        default=10,
        gt=1,
        le=3000,
        description=(
            "Numero de velas a retornar"
        )
    )
):

    # =====================================
    # 1. VALIDAR PARÁMETROS
    # =====================================

    symbol = symbol.upper().strip()

    if not re.fullmatch(r"[A-Z0-9]{3,20}", symbol):
        raise HTTPException(
            status_code=400,
            detail="Símbolo inválido"
        )

    if interval not in INTERVALOS:
        raise HTTPException(
            status_code=400,
            detail="Intervalo no soportado por Binance"
        )

    # Historial adicional para inicializar
    # los indicadores EMA, RSI, MACD y ADX.
    warmup = 200
    limit = min(periodos + warmup, 2500)

    # =====================================
    # 2. CONSULTAR BINANCE
    # =====================================

    try:
        response = requests.get(
            BINANCE_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=20
        )

        response.raise_for_status()
        velas = response.json()

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error consultando Binance: {exc}"
        )

    if not isinstance(velas, list) or not velas:
        raise HTTPException(
            status_code=502,
            detail="Respuesta de Binance inválida"
        )

    # =====================================
    # 3. CONSTRUIR DATAFRAME
    # =====================================

    # Eliminar elementos sin 12 posiciones
    velas = [
        v for v in velas
        if isinstance(v, list) and len(v) >= 12
    ]

    if not velas:
        raise HTTPException(
            status_code=422,
            detail="Binance no devolvió velas válidas"
        )

    df = pd.DataFrame(
        [v[:12] for v in velas],
        columns=COLUMNAS
    )

    # Convertir campos numéricos
    numeric_cols = [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # =====================================
    # 4. DEPURACIÓN DE DATOS
    # =====================================

    original_count = len(df)

    # Eliminar valores nulos o no numéricos
    df = df.dropna(subset=numeric_cols).copy()

    # Orden cronológico
    df = df.sort_values("open_time")

    # Cada vela es única por su apertura
    df = df.drop_duplicates(
        subset=["open_time"],
        keep="last"
    )

    # Solo velas cerradas
    now_ms = pd.Timestamp.now(tz="UTC").value // 10**6

    df = df[
        (df["close_time"] < now_ms) &
        (df["close_time"] > df["open_time"])
    ].copy()

    # Validar coherencia de OHLCV
    valid = (
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["volume"] >= 0) &

        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &

        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    )

    df = df[valid].copy()

    df = df.reset_index(drop=True)

    removed_count = original_count - len(df)

    if len(df) < 51:
        raise HTTPException(
            status_code=422,
            detail="Se requieren más de 50 velas válidas"
        )

    if len(df) < periodos:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Solo hay {len(df)} velas válidas; "
                f"se solicitaron {periodos}"
            )
        )

    # =====================================
    # 5. CALCULAR INDICADORES TÉCNICOS
    # =====================================

    # EMA
    df["ema20"] = ta.ema(
        df["close"],
        length=20
    )

    df["ema50"] = ta.ema(
        df["close"],
        length=50
    )

    # RSI
    df["rsi14"] = ta.rsi(
        df["close"],
        length=14
    )

    # MACD (12,26,9)
    macd = ta.macd(
        df["close"],
        fast=12,
        slow=26,
        signal=9
    )

    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_histogram"] = macd["MACDh_12_26_9"]

    # ADX14 y DI
    adx = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=14
    )

    df["adx14"] = adx["ADX_14"]

    # =====================================
    # 6. FORMATEAR FECHAS
    # =====================================

    df["open_datetime"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_datetime"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # =====================================
    # 7. SELECCIONAR ÚLTIMAS N VELAS
    # =====================================

    result = df.tail(periodos).copy()

    registros = []

    for _, row in result.iterrows():

        def numero(value, decimals=8):
            if pd.isna(value):
                return None
            return round(float(value), decimals)

        registros.append({
            "open_date": row["open_datetime"].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "open": numero(row["open"]),
            "high": numero(row["high"]),
            "low": numero(row["low"]),
            "close": numero(row["close"]),
            "volume": numero(row["volume"]),
            "quote_volume": numero(row["quote_volume"]),
            "trades": numero(row["trades"]),
            "taker_buy_base": numero(row["taker_buy_base"]),
            "taker_buy_quote": numero(row["taker_buy_quote"]),
            "ema20": numero(row["ema20"]),
            "ema50": numero(row["ema50"]),
            "rsi14": numero(row["rsi14"], 4),

            "macd": numero(row["macd"]),
            "macd_signal": numero(row["macd_signal"]),
            "macd_histogram": numero(
                row["macd_histogram"]
            ),

            "adx14": numero(row["adx14"], 4)
        })

    # =====================================
    # 8. RESPUESTA API
    # =====================================

    return JSONResponse(
        content={
            "symbol": symbol,
            "timeframe": interval,
            "periodos": periodos,
            "timezone": "UTC",
            "velas_recibidas": original_count,
            "velas_validas": len(df),
            "velas_eliminadas": removed_count,
            "total_registros": len(registros),
            "registros": registros
        }
    )
