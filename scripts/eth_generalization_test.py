"""
ETH Generalization Test: does the BTC-trained volatility forecasting methodology hold up on a
second, independent asset?

Design (the strongest version of this test, not the easiest one):
  - HAR-RV-X: BTC's exact trained Ridge coefficients (models/har_rv_x_tuned.pkl), applied
    directly to ETH's engineered features. ZERO retuning.
  - EWMA: BTC's tuned decay factor lambda=0.995 (models/ewma_best_lambda.pkl), applied directly
    to ETH's own return series. ZERO retuning.
  - GARCH: the SAME already-chosen specification (models/garch_best_spec.pkl, e.g. GJR-GARCH
    with Student-t errors) is refit on ETH's OWN training returns. This is not "retuning" in the
    sense that matters here — no new AIC search over specs is performed — but GARCH parameters
    (omega/alpha/beta) are inherently properties of one specific return series' conditional
    variance process and cannot be transplanted between two assets with very different
    unconditional volatility levels the way a Ridge coefficient or a lambda can.

If HAR-RV-X and EWMA — using BTC's frozen parameters, unchanged — still beat naive persistence
on ETH by a similar margin, that is real evidence the original BTC result reflects something
structural about crypto volatility rather than an artifact of BTC's specific history.

Run from the project root (after scripts/fetch_eth_data.py has produced data/eth_hourly_ohlcv.csv
and data/eth_hourly_live.csv):
    venv/bin/python3 scripts/eth_generalization_test.py
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

HAR_MODEL = joblib.load("models/har_rv_x_tuned.pkl")          # frozen, BTC-trained
EWMA_LAMBDA = joblib.load("models/ewma_best_lambda.pkl")       # frozen, BTC-tuned
GARCH_SPEC = joblib.load("models/garch_best_spec.pkl")         # spec choice only, reused


def build_historical_split_eth():
    df_raw = pd.read_csv("data/eth_hourly_ohlcv.csv", parse_dates=["Date"])
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

    # --- HAR-RV-X: frozen BTC-trained Ridge model, no refitting ---
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
    har_pred = HAR_MODEL.predict(har_test[["RV_day", "RV_week", "RV_month", "ATR_X"]])
    har_frame = pd.DataFrame({"Date": har_test["Date"], "HAR": har_pred, "y_true_har": har_test["target"]})

    # --- GARCH: same spec choice, refit fresh on ETH's own training returns ---
    df_clean_dated = df_clean.set_index("Date")
    returns_pct = df_clean_dated["hourly_return"].dropna() * 100
    returns_train = returns_pct.loc[:train_end_date]
    fit = arch_model(returns_train, vol=GARCH_SPEC["vol"], p=GARCH_SPEC["p"], o=GARCH_SPEC["o"],
                      q=GARCH_SPEC["q"], dist=GARCH_SPEC["dist"]).fit(disp="off")
    m_full = arch_model(returns_pct, vol=GARCH_SPEC["vol"], p=GARCH_SPEC["p"], o=GARCH_SPEC["o"],
                         q=GARCH_SPEC["q"], dist=GARCH_SPEC["dist"])
    m_fixed = m_full.fix(fit.params)
    fc = m_fixed.forecast(horizon=HORIZON, start=test_start_date, reindex=False)
    garch_pred = (np.sqrt(fc.variance.values[:, -1]) / 100)[: len(y_test)]

    # --- EWMA: frozen BTC-tuned lambda, no retuning ---
    returns_frac = df_clean_dated["hourly_return"]
    sigma_full = vl.ewma_sigma(returns_frac, EWMA_LAMBDA)
    ewma_pred = sigma_full.loc[test_start_date:test_end_date].values[: len(y_test)]

    main_frame = pd.DataFrame({
        "Date": dates_test, "y_true": y_test.values, "Naive": naive_pred.values,
        "GARCH": garch_pred, "EWMA": ewma_pred,
    })
    native_metrics = {
        "GARCH": vl.rmse_mae_r2(y_test.values, garch_pred),
        "EWMA": vl.rmse_mae_r2(y_test.values, ewma_pred),
        "HAR": vl.rmse_mae_r2(har_test["target"].values, har_pred),
    }
    merged = main_frame.merge(har_frame, on="Date", how="inner")
    assert np.allclose(merged["y_true"], merged["y_true_har"], atol=1e-9)
    merged = merged.drop(columns=["y_true_har"])

    return {
        "df": merged, "n_train": split_idx, "n_test_main": len(y_test),
        "n_test_aligned": len(merged), "har_own_test_start": har_test["Date"].iloc[0],
        "main_test_start": test_start_date, "train_mean_vol": train_mean_vol,
        "native_metrics": native_metrics, "garch_fit": fit,
    }


def build_live_split_eth():
    df_raw = pd.read_csv("data/eth_hourly_ohlcv.csv", parse_dates=["Date"])
    df_live = pd.read_csv("data/eth_hourly_live.csv", parse_dates=["Date"])
    combined_raw = (
        pd.concat([df_raw.tail(1500), df_live], ignore_index=True)
        .drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    )
    combined = vl.engineer(combined_raw)
    vol_series = combined.set_index("Date")["volatility_7d"]
    naive_series = vol_series.shift(1)
    train_mean_vol = vl.engineer(df_raw)["volatility_7d"].mean()

    last_training_date = df_raw["Date"].max()
    live_dates = df_live["Date"]
    live_scored_dates = combined[
        (combined["Date"] >= pd.Timestamp("2026-01-02")) & combined["Date"].isin(live_dates)
    ]["Date"].reset_index(drop=True)

    # --- HAR-RV-X: frozen BTC-trained Ridge model ---
    har_feats = pd.DataFrame(index=combined.index)
    har_feats["Date"] = combined["Date"]
    har_feats["RV_day"] = combined["volatility_7d"].shift(24)
    har_feats["RV_week"] = combined["volatility_7d"].shift(24).rolling(24 * 7).mean()
    har_feats["RV_month"] = combined["volatility_7d"].shift(24).rolling(24 * 30).mean()
    har_feats["ATR_X"] = combined["atr_pct"].shift(1)
    har_feats = har_feats.set_index("Date")

    # --- GARCH: same spec, refit on ETH's FULL training-period history (mirrors the BTC
    #     live-testing section's own methodology exactly) ---
    returns_pct_train = vl.engineer(df_raw)["hourly_return"].dropna() * 100
    fit = arch_model(returns_pct_train, vol=GARCH_SPEC["vol"], p=GARCH_SPEC["p"], o=GARCH_SPEC["o"],
                      q=GARCH_SPEC["q"], dist=GARCH_SPEC["dist"]).fit(disp="off")
    combined_returns_pct = combined.set_index("Date")["hourly_return"].dropna() * 100
    m_full = arch_model(combined_returns_pct, vol=GARCH_SPEC["vol"], p=GARCH_SPEC["p"], o=GARCH_SPEC["o"],
                         q=GARCH_SPEC["q"], dist=GARCH_SPEC["dist"])
    m_fixed = m_full.fix(fit.params)
    fc = m_fixed.forecast(horizon=HORIZON, start=live_scored_dates.iloc[0], reindex=False)
    garch_series = pd.Series((np.sqrt(fc.variance.values[:, -1]) / 100),
                              index=combined_returns_pct.loc[live_scored_dates.iloc[0]:].index[: len(fc.variance)])

    # --- EWMA: frozen BTC-tuned lambda ---
    returns_frac = combined.set_index("Date")["hourly_return"]
    ewma_series = vl.ewma_sigma(returns_frac, EWMA_LAMBDA)

    rows = []
    for d in live_scored_dates:
        if d not in har_feats.index or har_feats.loc[d].isna().any():
            continue
        rows.append({
            "Date": d,
            "HAR": float(HAR_MODEL.predict(har_feats.loc[[d], ["RV_day", "RV_week", "RV_month", "ATR_X"]])[0]),
            "EWMA": float(ewma_series.get(d, np.nan)),
            "GARCH": float(garch_series.get(d, np.nan)),
            "y_true": float(vol_series.get(d + pd.Timedelta(hours=HORIZON), np.nan)),
            "Naive": float(naive_series.get(d, np.nan)),
        })
    preds = pd.DataFrame(rows)
    n_total = len(preds)
    scored = preds.dropna(subset=["y_true", "Naive", "GARCH", "HAR", "EWMA"]).reset_index(drop=True)

    return {
        "df": scored[["Date", "y_true", "Naive", "GARCH", "HAR", "EWMA"]],
        "n_total_logged": n_total, "n_scoreable": len(scored),
        "train_mean_vol": train_mean_vol,
    }


def main():
    vl.print_header("ETH GENERALIZATION TEST — Does the BTC Methodology Hold Up on a Second Asset?")
    print("HAR-RV-X and EWMA use BTC's exact trained parameters (zero retuning). GARCH refits its")
    print("already-chosen specification fresh on ETH's own returns (GARCH parameters are inherently")
    print("per-series and cannot be transplanted between assets with different volatility levels).")

    hist = build_historical_split_eth()
    print(f"\nETH historical split reproduced: {hist['n_train']} train rows, "
          f"{hist['n_test_main']} test rows (main models), {hist['n_test_aligned']} aligned across all 3 models.")
    print(f"GARCH refit on ETH training returns -> AIC={hist['garch_fit'].aic:.2f} "
          f"(BTC's own AIC for this spec was 96,564.53 — not directly comparable across assets with "
          f"different data volumes/scales, shown only for reference).")

    live = build_live_split_eth()
    print(f"ETH live-test split built: {live['n_scoreable']} of {live['n_total_logged']} logged predictions "
          f"already have a real, elapsed outcome to score.")

    for name, bundle in [("ETH — HISTORICAL TEST SET", hist), ("ETH — LIVE TEST SET", live)]:
        df = bundle["df"]
        if len(df) == 0:
            print(f"\n{name}: no scoreable rows, skipping.")
            continue
        vl.audit_split(name, df, native_metrics=bundle.get("native_metrics"))
        vl.audit_directional(name, df, bundle["train_mean_vol"])
        vl.audit_mz(name, df)

    vl.print_header("GENERALIZATION VERDICT")
    print("Compare this section's DM test p-values and QLIKE reductions directly against")
    print("scripts/alpha_audit.py's BTC output for the same models. If HAR-RV-X/EWMA still show a")
    print("directionally similar, statistically detectable edge over naive persistence on ETH —")
    print("using BTC's frozen parameters, unchanged — that's real evidence the BTC result reflects")
    print("something structural about crypto volatility, not an artifact specific to BTC's own history.")


if __name__ == "__main__":
    main()
