"""
Fetches ETH/USDT hourly OHLCV data over the exact same date windows used for BTC
(notebooks/capstone-btc-volatility.ipynb, cell 7), so the ETH generalization test is scored on
a directly comparable historical + live-test period.

Run from the project root:
    venv/bin/python3 scripts/fetch_eth_data.py
"""
import time
import ccxt
import pandas as pd

SYMBOL = "ETH/USDT"
TIMEFRAME = "1h"
TRAIN_START = "2010-01-01T00:00:00Z"
TRAIN_END = "2026-01-01T00:00:00Z"
LIVE_TEST_START = "2026-01-02T00:00:00Z"
LIVE_TEST_END = "2026-08-26T00:00:00Z"

TRAIN_CSV = "data/eth_hourly_ohlcv.csv"
LIVE_CSV = "data/eth_hourly_live.csv"


def fetch_ohlcv_history(exchange, symbol, timeframe="1h", since_ms=None, until_ms=None, limit=1000):
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


def fetch_and_save(exchange, since_str, until_str, out_path, label):
    since_ms = exchange.parse8601(since_str)
    until_ms = exchange.parse8601(until_str)
    candles = fetch_ohlcv_history(exchange, SYMBOL, timeframe=TIMEFRAME, since_ms=since_ms, until_ms=until_ms)
    df = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.drop(columns=["Timestamp"]).sort_values("Date").reset_index(drop=True)
    until_ts = pd.Timestamp(until_str).tz_localize(None)
    df = df[df["Date"] < until_ts].reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"{label}: fetched {len(df)} candles ({df['Date'].min()} to {df['Date'].max()}), saved to {out_path}")


if __name__ == "__main__":
    exchange = ccxt.binance()
    fetch_and_save(exchange, TRAIN_START, TRAIN_END, TRAIN_CSV, "Training-period ETH data")
    fetch_and_save(exchange, LIVE_TEST_START, LIVE_TEST_END, LIVE_CSV, "Live-test-period ETH data")
