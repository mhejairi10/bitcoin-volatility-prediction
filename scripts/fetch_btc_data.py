"""
Fetches BTC/USDT hourly OHLCV data via ccxt (Binance) and appends it to the combined
dataset used throughout the capstone notebooks (data/btc_hourly_combined.csv).

If the CSV already exists, only candles newer than the last saved timestamp are fetched
and appended - existing rows are never touched or duplicated. If the CSV doesn't exist
yet, the full history from 2017-01-01 to now is fetched and a new file is created.

Run from the project root:
    python3 scripts/fetch_btc_data.py
"""
import os
import time
import ccxt
import pandas as pd

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
HISTORY_START = "2017-01-01T00:00:00Z"
OUT_CSV = "data/btc_hourly_combined.csv"
COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Date"]


def fetch_ohlcv_history(exchange, symbol, timeframe="1h", since_ms=None, limit=1000):
    all_candles = []
    since = since_ms
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)
    return all_candles


def candles_to_df(candles):
    df = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.drop(columns=["Timestamp"])[COLUMNS]
    return df.sort_values("Date").reset_index(drop=True)


def main():
    exchange = ccxt.binance()

    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV, parse_dates=["Date"])
        last_date = existing["Date"].max()
        since_ms = int(last_date.timestamp() * 1000) + 1
        print(f"Existing file found - last row is {last_date}. Fetching candles after that...")
    else:
        existing = pd.DataFrame(columns=COLUMNS)
        since_ms = pd.Timestamp(HISTORY_START).value // 10**6
        print(f"No existing file - fetching full history from {HISTORY_START}...")

    candles = fetch_ohlcv_history(exchange, SYMBOL, timeframe=TIMEFRAME, since_ms=since_ms)

    if not candles:
        print("No new candles available - file is already up to date.")
        return

    new_df = candles_to_df(candles)

    # Drop the most recent candle if it's still in progress (its hour hasn't fully closed yet)
    now = pd.Timestamp.utcnow().tz_localize(None)
    new_df = new_df[new_df["Date"] < now.floor("h")].reset_index(drop=True)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    combined.to_csv(OUT_CSV, index=False)
    print(f"Added {len(new_df)} new rows. File now has {len(combined)} rows, "
          f"from {combined['Date'].min()} to {combined['Date'].max()}.")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
