"""
Fetches 5-minute BTC/USDT OHLCV over a fixed 2024-01-01 -> yesterday window, to test whether
finer-frequency return sampling reduces the measurement noise in `volatility_7d` enough to raise
the achievable R2 ceiling (scripts/frequency_test.py runs the actual comparison).

A shorter window than the full project history is used deliberately: 5-minute bars are ~12x the
row count of hourly bars for the same calendar span, and this test only needs enough data for a
fair, apples-to-apples 80/20 split comparison against an hourly arm covering the identical dates
(frequency_test.py derives the hourly arm by resampling this same 5-minute data, rather than
reusing the separately-fetched hourly CSV, to remove any chance of a data-source mismatch
confounding the comparison).

Run from the project root:
    venv/bin/python3 scripts/fetch_5min_data.py
"""
import time
import ccxt
import pandas as pd

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
WINDOW_START = "2024-01-01T00:00:00Z"
WINDOW_END = "2026-08-26T00:00:00Z"
OUT_CSV = "data/btc_5min_ohlcv.csv"


def fetch_ohlcv_history(exchange, symbol, timeframe="5m", since_ms=None, until_ms=None, limit=1000):
    all_candles = []
    since = since_ms
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if until_ms is not None and since >= until_ms:
            break
        if len(candles) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)
    return all_candles


if __name__ == "__main__":
    exchange = ccxt.binance()
    since_ms = exchange.parse8601(WINDOW_START)
    until_ms = exchange.parse8601(WINDOW_END)
    candles = fetch_ohlcv_history(exchange, SYMBOL, timeframe=TIMEFRAME, since_ms=since_ms, until_ms=until_ms)
    df = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.drop(columns=["Timestamp"]).sort_values("Date").reset_index(drop=True)
    until_ts = pd.Timestamp(WINDOW_END).tz_localize(None)
    df = df[df["Date"] < until_ts].reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Fetched {len(df)} 5-minute candles ({df['Date'].min()} to {df['Date'].max()}), saved to {OUT_CSV}")
