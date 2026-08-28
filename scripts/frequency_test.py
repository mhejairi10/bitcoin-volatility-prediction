"""
Frequency Test: does computing volatility_7d (and HAR-RV-X's regressors) from 5-minute returns,
instead of hourly returns, raise the achievable R2 ceiling?

Hypothesis being tested: volatility_7d estimated from only 168 hourly returns carries real
measurement/estimation noise (a well-documented property of realized-volatility estimators —
Andersen & Bollerslev, 1998). R2 is capped by how noisy the TARGET itself is, independent of how
good the model is. Estimating volatility_7d from 2,016 five-minute returns instead should be a
less noisy proxy for the same underlying 7-day volatility — IF microstructure noise (bid-ask
bounce, tick discreteness) doesn't dominate at 5-minute sampling for this asset/exchange, which
is the real risk the "volatility signature" section below checks directly rather than assuming.

Both frequency arms are derived from the SAME raw 5-minute pull (data/btc_5min_ohlcv.csv,
fetched by scripts/fetch_5min_data.py) — the hourly arm is built by resampling the 5-minute bars
up to hourly, not by reusing the separately-fetched project CSV — so any difference in the result
is attributable to sampling frequency alone, not to a data-source mismatch between two separate
fetches. Every window/lag is expressed in real hours so a fair comparison holds the real-world
meaning of "7-day volatility," "168 hours ahead," and "14-hour ATR" fixed across both arms —
only how finely each is MEASURED changes.

Run from the project root (after scripts/fetch_5min_data.py has produced data/btc_5min_ohlcv.csv):
    venv/bin/python3 scripts/frequency_test.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import vol_lib as vl

FIVE_MIN_CSV = "data/btc_5min_ohlcv.csv"


def resample_ohlcv(df_5min, rule):
    d = df_5min.set_index("Date").sort_index()
    out = pd.DataFrame({
        "Open": d["Open"].resample(rule).first(),
        "High": d["High"].resample(rule).max(),
        "Low": d["Low"].resample(rule).min(),
        "Close": d["Close"].resample(rule).last(),
        "Volume": d["Volume"].resample(rule).sum(),
    }).dropna().reset_index()
    return out


def build_har_dataset(df_raw, bars_per_hour):
    """Build HAR-RV-X's dataset (RV_day/week/month, ATR_X, target, naive baseline) at an
    arbitrary bar frequency, with every window/lag expressed in real hours, so a 5-min run and
    an hourly run represent identical real-world spans — only the measurement fineness changes.
    """
    df = df_raw.sort_values("Date").reset_index(drop=True)
    ret = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["ret"] = ret

    roll = 168 * bars_per_hour        # 7-day rolling window
    horizon = 168 * bars_per_hour     # 168 hours ahead
    atr_period = 14 * bars_per_hour   # 14-hour ATR, matching the project's ATR-14
    lag1h = 1 * bars_per_hour         # "1 hour ago" — matches ATR_X's shift(1) in the hourly pipeline
    day = 24 * bars_per_hour

    # Realized volatility = sqrt(sum of squared returns) over the window, NOT rolling std() of
    # the per-bar return series. Those differ by a factor of ~sqrt(bars_in_window): std() alone
    # measures "typical size of one bar's move," which shrinks mechanically as bars get finer
    # and has nothing to do with genuine 7-day volatility. Using rolling std() here would make
    # every cross-frequency comparison meaningless (confirmed by a signature-plot sanity check:
    # the naive std() version was inflated at coarser frequencies by almost exactly
    # sqrt(bars_5min / bars_hourly) = sqrt(12) ~= 3.46x, a pure bar-count artifact).
    df["volatility_7d"] = np.sqrt((df["ret"] ** 2).rolling(roll).sum())

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(atr_period).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    df["RV_day"] = df["volatility_7d"].shift(day)
    df["RV_week"] = df["volatility_7d"].shift(day).rolling(day * 7).mean()
    df["RV_month"] = df["volatility_7d"].shift(day).rolling(day * 30).mean()
    df["ATR_X"] = df["atr_pct"].shift(lag1h)
    df["target"] = df["volatility_7d"].shift(-horizon)
    df["Naive"] = df["volatility_7d"].shift(lag1h)

    df = df.dropna(subset=["RV_day", "RV_week", "RV_month", "ATR_X", "target", "Naive"]).reset_index(drop=True)
    return df


def evaluate_frequency(df_raw, bars_per_hour, label):
    ds = build_har_dataset(df_raw, bars_per_hour)
    split_idx = int(len(ds) * 0.8)
    train, test = ds.iloc[:split_idx].copy(), ds.iloc[split_idx:].copy()

    ridge_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(Ridge(random_state=42), ridge_grid, cv=tscv,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(train[["RV_day", "RV_week", "RV_month", "ATR_X"]], train["target"])
    model = search.best_estimator_
    pred = model.predict(test[["RV_day", "RV_week", "RV_month", "ATR_X"]])

    m = vl.rmse_mae_r2(test["target"].values, pred)
    q = vl.qlike_mean(test["target"].values, pred)
    naive_m = vl.rmse_mae_r2(test["target"].values, test["Naive"].values)
    naive_q = vl.qlike_mean(test["target"].values, test["Naive"].values)
    dm = vl.diebold_mariano(test["target"].values, test["Naive"].values, pred,
                             h=horizon_for(bars_per_hour), loss="squared", name1="Naive", name2="HAR-RV-X")

    print(f"\n--- {label}  (bars/hour={bars_per_hour}, n_train={len(train):,}, n_test={len(test):,}, "
          f"best_alpha={search.best_params_['alpha']}) ---")
    print(f"  Target std. dev. (total variance to explain): {test['target'].std():.6f}")
    print(f"  Naive       RMSE={naive_m['RMSE']:.6f}  R2={naive_m['R2']:.4f}  QLIKE={naive_q:.4f}")
    print(f"  HAR-RV-X    RMSE={m['RMSE']:.6f}  R2={m['R2']:.4f}  QLIKE={q:.4f}   "
          f"({vl.pct_reduction(naive_m['RMSE'], m['RMSE']):+.1f}% RMSE vs naive)")
    print(f"  DM test (HAR vs naive): stat={dm['dm_stat']:+.3f}  p={dm['p_value']:.4f}  n={dm['n_obs']}")
    return {"R2": m["R2"], "RMSE": m["RMSE"], "QLIKE": q, "n_test": len(test)}


def horizon_for(bars_per_hour):
    return 168 * bars_per_hour


def volatility_signature(df_5min):
    """Andersen-Bollerslev-Diebold-Labys (2001) style signature plot: realized volatility of
    the SAME calendar days, estimated at increasingly coarse sampling frequencies. If mean RV
    is inflated at the finest frequency and flattens out as sampling coarsens, that inflation is
    microstructure noise (bid-ask bounce, tick discreteness), not genuine volatility — meaning
    the finer frequency would make the target WORSE, not better. If RV is roughly flat across
    frequencies, finer sampling has no bias problem, just proportionally less estimation noise.
    """
    print("\n--- Volatility signature plot (mean 7-day realized vol at each sampling frequency) ---")
    print("(Flat across frequencies = no microstructure bias. Rising sharply at the finest")
    print(" frequency = microstructure noise is inflating that estimate, not adding real signal.)")
    for rule, bars_per_hour, label in [("5min", 12, "5-minute"), ("15min", 4, "15-minute"),
                                        ("30min", 2, "30-minute"), ("1h", 1, "1-hour")]:
        resampled = resample_ohlcv(df_5min, rule)
        ret = resampled["Close"].pct_change().replace([np.inf, -np.inf], np.nan)
        vol7d = np.sqrt((ret ** 2).rolling(168 * bars_per_hour).sum()).dropna()
        print(f"  {label:10s}  mean volatility_7d = {vol7d.mean():.6f}   (n={len(vol7d):,})")


def main():
    vl.print_header("FREQUENCY TEST — Does 5-Minute Sampling Raise the R2 Ceiling?")
    df_5min = pd.read_csv(FIVE_MIN_CSV, parse_dates=["Date"])
    print(f"Loaded {len(df_5min):,} 5-minute candles: {df_5min['Date'].min()} to {df_5min['Date'].max()}")

    volatility_signature(df_5min)

    df_hourly = resample_ohlcv(df_5min, "1h")
    print(f"\nResampled to {len(df_hourly):,} hourly candles from the same underlying 5-minute data "
          f"(this is the hourly arm — NOT the project's separately-fetched hourly CSV, to remove any "
          f"data-source mismatch from the comparison).")

    hourly_result = evaluate_frequency(df_hourly, bars_per_hour=1, label="HOURLY (current pipeline resolution)")
    five_min_result = evaluate_frequency(df_5min, bars_per_hour=12, label="5-MINUTE")

    vl.print_header("VERDICT")
    r2_delta = five_min_result["R2"] - hourly_result["R2"]
    print(f"R2:    hourly={hourly_result['R2']:.4f}   5-min={five_min_result['R2']:.4f}   delta={r2_delta:+.4f}")
    print(f"RMSE:  hourly={hourly_result['RMSE']:.6f}   5-min={five_min_result['RMSE']:.6f}")
    print(f"QLIKE: hourly={hourly_result['QLIKE']:.4f}   5-min={five_min_result['QLIKE']:.4f}")
    if r2_delta > 0.02:
        print("\n-> 5-minute sampling gives a materially higher R2 on the SAME calendar window and model.")
        print("   Combined with the signature plot above: if RV was roughly flat across frequencies,")
        print("   this is genuine noise reduction in the target, not a new artifact.")
    elif r2_delta < -0.02:
        print("\n-> 5-minute sampling gives a WORSE R2 — check the signature plot above for microstructure")
        print("   noise inflating the 5-minute realized-volatility estimate.")
    else:
        print("\n-> No material difference. Target measurement noise at hourly resolution is apparently")
        print("   NOT the binding constraint on R2 for this problem — the ceiling is elsewhere (missing")
        print("   information sources, not measurement precision).")


if __name__ == "__main__":
    main()
