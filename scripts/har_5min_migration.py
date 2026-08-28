"""
Scoped 5-minute migration: HAR-RV-X only (not GARCH — already established to fail at this
horizon regardless of frequency, and its forecast cost scales too badly to be worth rerunning at
this resolution; not EWMA — out of scope for this pass, can be added the same way later).

Uses the exact same date boundaries as the main project (TRAIN_END / LIVE_TEST_START /
LIVE_TEST_END) so the historical and live-test results are directly comparable to the existing
hourly-resolution HAR-RV-X numbers already in the report/deck. The realized-volatility target
uses the corrected sqrt(sum of squared returns) definition validated in frequency_test.py, not
the project's original rolling-std() convention — this only matters when comparing across
frequencies (see frequency_test.py's docstring for why); it does not retroactively change the
already-delivered hourly results.

Run from the project root (after data/btc_5min_ohlcv.csv covers 2017-08-17 -> present):
    venv/bin/python3 scripts/har_5min_migration.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import vol_lib as vl

FIVE_MIN_CSV = "data/btc_5min_ohlcv.csv"
TRAIN_END = "2026-01-01T00:00:00Z"
LIVE_TEST_START = "2026-01-02T00:00:00Z"
LIVE_TEST_END = "2026-08-26T00:00:00Z"  # matches the cutoff already used in the project's reported HAR-RV-X live numbers

BARS_PER_HOUR = 12
ROLL = 168 * BARS_PER_HOUR
HORIZON_BARS = 168 * BARS_PER_HOUR
ATR_PERIOD = 14 * BARS_PER_HOUR
LAG_1H = 1 * BARS_PER_HOUR
DAY = 24 * BARS_PER_HOUR


def build_features(df_raw):
    df = df_raw.sort_values("Date").reset_index(drop=True)
    ret = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["ret"] = ret

    # Corrected realized-volatility definition (see frequency_test.py) — frequency-invariant,
    # unlike a raw rolling std() of the per-bar return series.
    df["volatility_7d"] = np.sqrt((df["ret"] ** 2).rolling(ROLL).sum())

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    df["RV_day"] = df["volatility_7d"].shift(DAY)
    df["RV_week"] = df["volatility_7d"].shift(DAY).rolling(DAY * 7).mean()
    df["RV_month"] = df["volatility_7d"].shift(DAY).rolling(DAY * 30).mean()
    df["ATR_X"] = df["atr_pct"].shift(LAG_1H)
    df["target"] = df["volatility_7d"].shift(-HORIZON_BARS)
    df["Naive"] = df["volatility_7d"].shift(LAG_1H)
    return df


def main():
    vl.print_header("SCOPED 5-MINUTE MIGRATION — HAR-RV-X Only")
    df_raw = pd.read_csv(FIVE_MIN_CSV, parse_dates=["Date"])
    print(f"Loaded {len(df_raw):,} 5-minute candles: {df_raw['Date'].min()} to {df_raw['Date'].max()}")

    df = build_features(df_raw)
    train_end_ts = pd.Timestamp(TRAIN_END).tz_localize(None)
    live_start_ts = pd.Timestamp(LIVE_TEST_START).tz_localize(None)
    live_end_ts = pd.Timestamp(LIVE_TEST_END).tz_localize(None)

    # --- Historical split: fit + tune on the 80% train portion of the TRAIN_END-bounded range ---
    train_range = df[df["Date"] < train_end_ts].dropna(
        subset=["RV_day", "RV_week", "RV_month", "ATR_X", "target", "Naive"]
    ).reset_index(drop=True)
    split_idx = int(len(train_range) * 0.8)
    har_train, har_test = train_range.iloc[:split_idx], train_range.iloc[split_idx:]
    print(f"\nHistorical split: {len(har_train):,} train rows, {len(har_test):,} test rows "
          f"({har_train['Date'].iloc[0]} to {har_train['Date'].iloc[-1]} train; "
          f"{har_test['Date'].iloc[0]} to {har_test['Date'].iloc[-1]} test)")

    ridge_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(Ridge(random_state=42), ridge_grid, cv=tscv,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(har_train[["RV_day", "RV_week", "RV_month", "ATR_X"]], har_train["target"])
    model = search.best_estimator_
    print(f"Best Ridge alpha: {search.best_params_['alpha']}")

    hist_pred = model.predict(har_test[["RV_day", "RV_week", "RV_month", "ATR_X"]])
    hist_df = pd.DataFrame({
        "Date": har_test["Date"].values, "y_true": har_test["target"].values,
        "Naive": har_test["Naive"].values, "HAR": hist_pred,
    })

    # --- Live test: same trained model, scored on the genuinely later, untouched live window ---
    live_range = df[(df["Date"] >= live_start_ts) & (df["Date"] < live_end_ts)].dropna(
        subset=["RV_day", "RV_week", "RV_month", "ATR_X", "target", "Naive"]
    ).reset_index(drop=True)
    live_pred = model.predict(live_range[["RV_day", "RV_week", "RV_month", "ATR_X"]])
    live_df = pd.DataFrame({
        "Date": live_range["Date"].values, "y_true": live_range["target"].values,
        "Naive": live_range["Naive"].values, "HAR": live_pred,
    })
    print(f"Live-test rows scoreable at 5-min resolution: {len(live_df):,}")

    def report(name, d):
        vl.print_header(f"{name} (5-MINUTE, HAR-RV-X only)")
        naive_m = vl.rmse_mae_r2(d["y_true"], d["Naive"])
        naive_q = vl.qlike_mean(d["y_true"], d["Naive"])
        har_m = vl.rmse_mae_r2(d["y_true"], d["HAR"])
        har_q = vl.qlike_mean(d["y_true"], d["HAR"])
        print(f"  Naive     RMSE={naive_m['RMSE']:.6f}  MAE={naive_m['MAE']:.6f}  R2={naive_m['R2']:.4f}  QLIKE={naive_q:.4f}")
        print(f"  HAR-RV-X  RMSE={har_m['RMSE']:.6f}  MAE={har_m['MAE']:.6f}  R2={har_m['R2']:.4f}  QLIKE={har_q:.4f}   "
              f"({vl.pct_reduction(naive_m['RMSE'], har_m['RMSE']):+.1f}% RMSE vs naive)")
        dm = vl.diebold_mariano(d["y_true"], d["Naive"], d["HAR"], h=HORIZON_BARS, loss="squared", name1="Naive", name2="HAR")
        dm_q = vl.diebold_mariano(d["y_true"], d["Naive"], d["HAR"], h=HORIZON_BARS, loss="qlike", name1="Naive", name2="HAR")
        print(f"  DM (squared) vs naive: stat={dm['dm_stat']:+.3f}  p={dm['p_value']:.4f}  n={dm['n_obs']}")
        print(f"  DM (QLIKE)   vs naive: stat={dm_q['dm_stat']:+.3f}  p={dm_q['p_value']:.4f}  n={dm_q['n_obs']}")
        mz = vl.mincer_zarnowitz(d["y_true"], d["HAR"], h=HORIZON_BARS)
        print(f"  Mincer-Zarnowitz: alpha={mz['alpha']:+.6f}  beta={mz['beta']:+.4f}  R2={mz['r2']:.4f}  "
              f"joint-calib p={mz['joint_calibration_pvalue']:.4f}")
        return har_m

    hist_metrics = report("HISTORICAL TEST SET", hist_df)
    live_metrics = report("LIVE TEST SET", live_df)

    vl.print_header("COMPARISON — 5-MINUTE vs. THE PROJECT'S EXISTING HOURLY HAR-RV-X NUMBERS")
    print("                    Historical test              Live test")
    print("                    RMSE       R2                RMSE       R2")
    print(f"Hourly (reported)   0.001418   0.1840            0.001502   0.0610")
    print(f"5-minute (this run) {hist_metrics['RMSE']:.6f}   {hist_metrics['R2']:.4f}"
          f"            {live_metrics['RMSE']:.6f}   {live_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
