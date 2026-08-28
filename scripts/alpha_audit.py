"""
Alpha Audit: independent quantitative validation of the BTC 7-day realized-volatility
forecasting models (GARCH, HAR-RV-X, EWMA) against naive-persistence and mean-reversion
baselines.

This script does NOT trust the notebook's own printed metrics. It independently reproduces
every prediction from raw data (data/btc_hourly_ohlcv.csv, data/btc_hourly_live.csv) plus the
saved model artifacts (models/*.pkl), replicating the notebook's exact feature-engineering and
split logic. If this script's numbers disagree with the notebook's, that disagreement is itself
a leakage/bug signal.

Shared evaluation machinery (feature engineering, DM test, QLIKE, Mincer-Zarnowitz, directional
accuracy) lives in scripts/vol_lib.py, so scripts/eth_generalization_test.py can reuse the exact
same methodology on a second asset.

Run from the project root:
    venv/bin/python3 scripts/alpha_audit.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from arch import arch_model

import vol_lib as vl

HORIZON = vl.HORIZON
FEATURE_COLS = vl.FEATURE_COLS


def build_historical_split():
    df_raw = pd.read_csv("data/btc_hourly_ohlcv.csv", parse_dates=["Date"])
    df = vl.engineer(df_raw)
    df_clean = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    lag_source_cols = [c for c in FEATURE_COLS if c != "Date"]
    df_model = df_clean.copy()
    for col in lag_source_cols:
        for lag in vl.LAGS:
            df_model[f"{col}_lag{lag}"] = df_model[col].shift(lag)
    df_model["target"] = df_model["volatility_7d"].shift(-HORIZON)
    df_model = df_model.dropna().reset_index(drop=True)

    split_idx = int(len(df_model) * 0.8)
    dates_test = df_model["Date"].iloc[split_idx:].reset_index(drop=True)
    y_test = df_model["target"].iloc[split_idx:].reset_index(drop=True)
    naive_pred = df_model["volatility_7d_lag1"].iloc[split_idx:].reset_index(drop=True)
    train_end_date = df_model["Date"].iloc[split_idx - 1]
    test_start_date = df_model["Date"].iloc[split_idx]
    test_end_date = df_model["Date"].iloc[-1]
    train_mean_vol = df_model["volatility_7d"].iloc[:split_idx].mean()

    # --- HAR-RV-X (its own separate warm-up / split, exactly as in the notebook) ---
    har_df = pd.DataFrame(index=df_model.index)
    har_df["Date"] = df_model["Date"]
    har_df["RV_day"] = df_model["volatility_7d"].shift(24)
    har_df["RV_week"] = df_model["volatility_7d"].shift(24).rolling(24 * 7).mean()
    har_df["RV_month"] = df_model["volatility_7d"].shift(24).rolling(24 * 30).mean()
    har_df["ATR_X"] = df_model["atr_pct"].shift(1)
    har_df["target"] = df_model["target"]
    har_df = har_df.dropna().reset_index(drop=True)
    har_split = int(len(har_df) * 0.8)
    har_test = har_df.iloc[har_split:].reset_index(drop=True)
    har_model = joblib.load("models/har_rv_x_tuned.pkl")
    har_pred = har_model.predict(har_test[["RV_day", "RV_week", "RV_month", "ATR_X"]])
    har_frame = pd.DataFrame({"Date": har_test["Date"], "HAR": har_pred, "y_true_har": har_test["target"]})

    # --- GARCH (refit on train returns only, genuine rolling multi-step forecast) ---
    df_clean_dated = df_clean.set_index("Date")
    returns_pct = df_clean_dated["hourly_return"].dropna() * 100
    returns_train = returns_pct.loc[:train_end_date]
    spec = joblib.load("models/garch_best_spec.pkl")
    fit = arch_model(returns_train, vol=spec["vol"], p=spec["p"], o=spec["o"], q=spec["q"], dist=spec["dist"]).fit(disp="off")
    m_full = arch_model(returns_pct, vol=spec["vol"], p=spec["p"], o=spec["o"], q=spec["q"], dist=spec["dist"])
    m_fixed = m_full.fix(fit.params)
    fc = m_fixed.forecast(horizon=HORIZON, start=test_start_date, reindex=False)
    garch_pred = (np.sqrt(fc.variance.values[:, -1]) / 100)[: len(y_test)]

    # --- EWMA (tuned lambda) ---
    returns_frac = df_clean_dated["hourly_return"]
    lam = joblib.load("models/ewma_best_lambda.pkl")
    sigma_full = vl.ewma_sigma(returns_frac, lam)
    ewma_pred = sigma_full.loc[test_start_date:test_end_date].values[: len(y_test)]

    main_frame = pd.DataFrame({
        "Date": dates_test, "y_true": y_test.values, "Naive": naive_pred.values,
        "GARCH": garch_pred, "EWMA": ewma_pred,
    })

    # Each model's OWN native metrics (its own full test window, matching the notebook exactly)
    # — printed as a cross-check that this independent reproduction agrees with the notebook,
    # before anything gets aligned/truncated for the pairwise statistical tests below.
    native_metrics = {
        "GARCH": vl.rmse_mae_r2(y_test.values, garch_pred),
        "EWMA": vl.rmse_mae_r2(y_test.values, ewma_pred),
        "HAR": vl.rmse_mae_r2(har_test["target"].values, har_pred),
    }

    # Align everything (HAR has its own, slightly different warm-up boundary) to the
    # intersection of dates so pairwise statistical tests compare identical observations.
    merged = main_frame.merge(har_frame, on="Date", how="inner")
    assert np.allclose(merged["y_true"], merged["y_true_har"], atol=1e-9), "target mismatch between splits"
    merged = merged.drop(columns=["y_true_har"])

    # --- Purge/embargo audit: how many train rows' targets bleed past train_end_date? ---
    train_dates_all = df_model["Date"].iloc[:split_idx]
    contaminated = (train_dates_all > (train_end_date - pd.Timedelta(hours=HORIZON))).sum()

    return {
        "df": merged,
        "n_train": split_idx,
        "n_test_main": len(y_test),
        "n_test_aligned": len(merged),
        "har_own_test_start": har_test["Date"].iloc[0],
        "main_test_start": test_start_date,
        "boundary_contaminated_train_rows": int(contaminated),
        "train_mean_vol": train_mean_vol,
        "native_metrics": native_metrics,
    }


def build_live_split():
    df_raw = pd.read_csv("data/btc_hourly_ohlcv.csv", parse_dates=["Date"])
    df_live = pd.read_csv("data/btc_hourly_live.csv", parse_dates=["Date"])
    combined_raw = (
        pd.concat([df_raw.tail(1500), df_live], ignore_index=True)
        .drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    )
    combined = vl.engineer(combined_raw)
    vol_series = combined.set_index("Date")["volatility_7d"]
    naive_series = vol_series.shift(1)

    preds = pd.read_csv("data/live_test_predictions.csv", parse_dates=["Date"])
    preds["y_true"] = (preds["Date"] + pd.Timedelta(hours=HORIZON)).map(vol_series)
    preds["Naive"] = preds["Date"].map(naive_series)
    preds = preds.rename(columns={"GARCH_Pred": "GARCH", "HAR_RV_X_Pred": "HAR", "EWMA_Pred": "EWMA"})
    scored = preds.dropna(subset=["y_true", "Naive", "GARCH", "HAR", "EWMA"]).reset_index(drop=True)

    # Train-period mean, used as the mean-reversion baseline anchor (no leakage: computed only
    # from data strictly before the live-test window begins).
    train_mean_vol = vl.engineer(df_raw)["volatility_7d"].mean()

    return {
        "df": scored[["Date", "y_true", "Naive", "GARCH", "HAR", "EWMA"]],
        "n_total_logged": len(preds),
        "n_scoreable": len(scored),
        "train_mean_vol": train_mean_vol,
    }


def main():
    vl.print_header("ALPHA AUDIT — BTC 7-Day Realized Volatility Forecasts")
    print("Independent reproduction from raw data + saved model artifacts (not the notebook's own printed numbers).")

    hist = build_historical_split()
    print(f"\nHistorical split reproduced: {hist['n_train']} train rows, "
          f"{hist['n_test_main']} test rows (main models), {hist['n_test_aligned']} aligned across all 3 models "
          f"(HAR's own test window starts {hist['har_own_test_start'].date()} vs. {hist['main_test_start'].date()} "
          f"for GARCH/EWMA/naive — a real, if minor, split-boundary inconsistency in the original notebook).")

    live = build_live_split()
    print(f"Live-test split loaded: {live['n_scoreable']} of {live['n_total_logged']} logged predictions "
          f"already have a real, elapsed outcome to score.")

    for name, bundle in [("BTC — HISTORICAL TEST SET", hist), ("BTC — LIVE TEST SET", live)]:
        df = bundle["df"]
        vl.audit_split(name, df, native_metrics=bundle.get("native_metrics"))
        vl.audit_directional(name, df, bundle["train_mean_vol"])
        vl.audit_mz(name, df)

    vl.print_header("DATA LEAKAGE / SANITY CHECK")
    print(f"1. Feature-target causality: verified by code review (notebooks/capstone-btc-volatility.ipynb,")
    print(f"   cells 25-26). Every feature is `_lag1/_lag2/_lag3` of its source column (`.shift(lag)`,")
    print(f"   lag >= 1) — strictly prior information relative to each row. Target is `volatility_7d.shift(-{HORIZON})`,")
    print(f"   i.e. the row's own current value is never a feature of its own target. No same-time leakage found.")
    print(f"\n2. Train/test split boundary (purge/embargo): NO embargo gap currently exists — split_idx is a hard")
    print(f"   80/20 index cut with zero gap. {hist['boundary_contaminated_train_rows']} of {hist['n_train']} train rows "
          f"({100*hist['boundary_contaminated_train_rows']/hist['n_train']:.2f}%) have a target value computed from")
    print(f"   return observations that chronologically fall inside the nominal test period (because the target is")
    print(f"   itself a {HORIZON}-hour-forward rolling statistic). This does not leak test ROWS into training features,")
    print(f"   but it is exactly the boundary-overlap issue purged/embargoed CV is designed to eliminate.")
    print(f"   RECOMMENDATION: drop the last {HORIZON} rows of train (or first {HORIZON} rows of test) as an explicit")
    print(f"   embargo. Given it affects <1% of rows, it is very unlikely to be moving the headline RMSE/R2 materially")
    print(f"   — but it should be fixed for rigor before any claim of a clean walk-forward evaluation.")
    print(f"\n3. HAR-RV-X evaluates on a slightly different, later-starting test window than GARCH/EWMA/naive")
    print(f"   (see split reproduction note above) — a minor inconsistency, not a leakage risk, but it means the")
    print(f"   headline comparison table is not evaluating all four series on literally identical rows unless")
    print(f"   explicitly aligned (as this script does for its own DM tests).")
    print(f"\n4. Directional hit-rate check: see the >{int(vl.SUSPICIOUS_HIT_RATE*100)}% flag above for each model/split —")
    print(f"   flagged automatically if triggered.")


if __name__ == "__main__":
    main()
