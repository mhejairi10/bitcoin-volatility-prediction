"""
Extends data/btc_5min_ohlcv.csv backward to cover the project's full training history
(2017-08-17, Binance's earliest BTC/USDT data, through 2024-01-01, where the existing 5-minute
pull already picks up), so the scoped HAR-RV-X 5-minute migration can use the identical
2017-08-17 -> 2026-01-01 training window as the rest of the project.

Run from the project root:
    venv/bin/python3 scripts/fetch_5min_extend.py
"""
import time
import ccxt
import pandas as pd

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
WINDOW_START = "2017-08-17T00:00:00Z"
WINDOW_END = "2024-01-01T00:00:00Z"
EXISTING_CSV = "data/btc_5min_ohlcv.csv"
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
    df_new = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df_new["Date"] = pd.to_datetime(df_new["Timestamp"], unit="ms")
    df_new = df_new.drop(columns=["Timestamp"]).sort_values("Date").reset_index(drop=True)
    until_ts = pd.Timestamp(WINDOW_END).tz_localize(None)
    df_new = df_new[df_new["Date"] < until_ts].reset_index(drop=True)
    print(f"Fetched {len(df_new)} new candles ({df_new['Date'].min()} to {df_new['Date'].max()})")

    df_existing = pd.read_csv(EXISTING_CSV, parse_dates=["Date"])
    combined = (
        pd.concat([df_new, df_existing], ignore_index=True)
        .drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    )
    combined.to_csv(OUT_CSV, index=False)
    print(f"Merged: {len(combined)} total candles ({combined['Date'].min()} to {combined['Date'].max()}), saved to {OUT_CSV}")
