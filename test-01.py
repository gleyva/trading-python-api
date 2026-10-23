
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Trading API",
    version="1.0.0"
)

class Indicadores(BaseModel):
    symbol: str
    price: float
    ema20: float
    ema50: float
    rsi14: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(data: Indicadores):

    signal = "HOLD"

    if (
        data.ema20 > data.ema50
        and 45 <= data.rsi14 <= 65
    ):
        signal = "BUY"

    return {
        "symbol": data.symbol,
        "signal": signal,
        "price": data.price
    }
