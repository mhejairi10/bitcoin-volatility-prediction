"""
Shared feature engineering + evaluation library for the volatility alpha audit, used by both
scripts/alpha_audit.py (BTC) and scripts/eth_generalization_test.py (ETH). Keeping this in one
place means both coins are scored with an identical methodology — any difference in the results
is a difference in the asset, not a difference in how it was measured.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROLLING_WINDOW = 168
HORIZON = 168
LAGS = [1, 2, 3]
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume", "hourly_return", "log_return",
    "volatility_7d", "ma_7d", "volume_chg", "ema_12", "ema_26", "ema_spread",
    "rsi_14", "atr_14", "atr_pct", "bb_width", "macd_hist",
]
SUSPICIOUS_HIT_RATE = 0.62


# ---------------------------------------------------------------------------
# Feature engineering — reproduced verbatim from the notebook (cells 12-14)
# ---------------------------------------------------------------------------

def compute_indicators(df):
    close, high, low = df["Close"], df["High"], df["Low"]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12"], df["ema_26"] = ema_12, ema_26
    df["ema_spread"] = (ema_12 - ema_26) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line
    return df


def engineer(df_raw):
    df = df_raw.copy().sort_values("Date").reset_index(drop=True)
    df["hourly_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volume_chg"] = df["Volume"].pct_change()
    inf_cols = ["hourly_return", "log_return", "volume_chg"]
    df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
    df["volatility_7d"] = df["hourly_return"].rolling(ROLLING_WINDOW).std()
    df["ma_7d"] = df["Close"].rolling(ROLLING_WINDOW).mean()
    df = compute_indicators(df)
    return df


def ewma_sigma(returns, lam):
    r2 = returns.values ** 2
    n = len(r2)
    sigma2 = np.full(n, np.nan)
    valid = np.flatnonzero(~np.isnan(r2))
    if len(valid) == 0:
        return pd.Series(sigma2, index=returns.index)
    start = valid[0]
    sigma2[start] = np.nanmean(r2[start:start + 24])
    for t in range(start + 1, n):
        prev_r2 = r2[t - 1] if not np.isnan(r2[t - 1]) else 0.0
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * prev_r2
    return pd.Series(np.sqrt(sigma2), index=returns.index)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def rmse_mae_r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def pct_reduction(baseline_metric, model_metric):
    return (1 - model_metric / baseline_metric) * 100


def qlike_losses(y_true, y_pred, eps=1e-12):
    """Per-observation QLIKE loss (Patton, 2011), evaluated on variance (y**2), lower is better.
    QLIKE = ratio - ln(ratio) - 1, where ratio = actual_variance / predicted_variance.
    Unlike MSE, QLIKE-based model rankings stay consistent even though `y_true` (realized
    volatility) is itself a noisy proxy for the true latent volatility, not the true value —
    the property Patton calls "robustness," and the reason QLIKE is the standard loss function
    in the academic/practitioner volatility-forecasting literature. It's also asymmetric: it
    penalizes under-predicting volatility more heavily than over-predicting it, which matches
    how a risk desk actually experiences the cost of being wrong.
    """
    var_true = np.asarray(y_true, float) ** 2
    var_pred = np.clip(np.asarray(y_pred, float) ** 2, eps, None)
    ratio = var_true / var_pred
    return ratio - np.log(ratio) - 1


def qlike_mean(y_true, y_pred):
    return float(np.mean(qlike_losses(y_true, y_pred)))


def _loss_diff(y_true, pred1, pred2, loss):
    y_true, pred1, pred2 = np.asarray(y_true, float), np.asarray(pred1, float), np.asarray(pred2, float)
    if loss == "squared":
        e1, e2 = y_true - pred1, y_true - pred2
        return e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        e1, e2 = y_true - pred1, y_true - pred2
        return np.abs(e1) - np.abs(e2)
    elif loss == "qlike":
        return qlike_losses(y_true, pred1) - qlike_losses(y_true, pred2)
    elif callable(loss):
        return loss(y_true, pred1) - loss(y_true, pred2)
    else:
        raise ValueError("loss must be 'squared', 'absolute', 'qlike', or a callable(y_true, y_pred)->array")


def diebold_mariano(y_true, pred1, pred2, h=HORIZON, loss="squared", name1="Model 1", name2="Model 2"):
    """Diebold-Mariano test, H0: models 1 and 2 have equal predictive accuracy.
    Uses a Newey-West long-run variance estimate with truncation lag h-1 (the standard choice
    for h-step-ahead forecasts, whose errors are serially correlated up to lag h-1 by
    construction) and the Harvey-Leybourne-Newbold (1997) small-sample correction, with a
    Student-t reference distribution (n-1 dof) rather than the asymptotic normal.
    `loss` may be 'squared', 'absolute', 'qlike', or any callable(y_true, y_pred) -> per-obs array.
    """
    d = _loss_diff(y_true, pred1, pred2, loss)
    n = len(d)
    d_bar = d.mean()
    max_lag = h - 1
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / (max_lag + 1)) * cov
    var_d = max(var_d, 1e-300)

    dm_stat = d_bar / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_adj = dm_stat * hln
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_adj), df=n - 1))

    better = name1 if d_bar < 0 else name2
    return {
        "dm_stat": dm_stat_adj, "p_value": p_value, "mean_loss_diff": d_bar,
        "n_obs": n, "favors": better,
    }


def directional_diagnostics(df, train_mean_vol):
    """Hit rate for the *change* from the known origin value, plus a mean-reversion baseline:
    'the value will move toward its long-run training-period mean.' A model with no real
    directional skill should still be within sampling noise of this baseline, not just of a
    coin flip — volatility mean-reverts, so a naive mean-reversion caller already has an edge
    over 50/50 by construction, and that's the bar a genuinely useful model needs to clear.
    """
    origin = df["Naive"].values  # most recently known actual value = the forecast origin
    actual_dir = np.sign(df["y_true"].values - origin)
    mean_rev_dir = np.sign(train_mean_vol - origin)

    out = {}
    valid_actual = actual_dir != 0
    out["mean_reversion_baseline_hit_rate"] = float((mean_rev_dir[valid_actual] == actual_dir[valid_actual]).mean())
    for col in ["GARCH", "HAR", "EWMA"]:
        model_dir = np.sign(df[col].values - origin)
        mask = valid_actual & (model_dir != 0)
        hit_rate = float((model_dir[mask] == actual_dir[mask]).mean())
        out[col] = {
            "hit_rate": hit_rate,
            "n": int(mask.sum()),
            "suspicious": hit_rate > SUSPICIOUS_HIT_RATE,
        }
    return out


def mincer_zarnowitz(y_true, y_pred, h=HORIZON, label=""):
    """Regress actual on predicted: y_true = alpha + beta * y_pred + e.
    A well-calibrated forecast has alpha=0, beta=1. Reparametrized as
    (y_true - y_pred) = a0 + a1 * y_pred + e, so a0=alpha and a1=beta-1 — testing the original
    null (alpha=0, beta=1) becomes a standard zero-restriction Wald test on (a0, a1), which
    statsmodels supports directly. HAC (Newey-West) standard errors with max_lag = h-1 account
    for the serial correlation h-step-ahead forecast errors carry by construction, so the joint
    calibration test isn't spuriously over-confident.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    X = sm.add_constant(y_pred)
    model = sm.OLS(y_true - y_pred, X).fit()
    robust = model.get_robustcov_results(cov_type="HAC", maxlags=h - 1)
    a0, a1 = robust.params
    alpha, beta = a0, a1 + 1
    r2 = sm.OLS(y_true, sm.add_constant(y_pred)).fit().rsquared
    wald_result = robust.wald_test(np.eye(2), use_f=True)
    return {
        "alpha": alpha, "beta": beta, "r2": r2,
        "alpha_se": robust.bse[0], "beta_se": robust.bse[1],
        "joint_calibration_pvalue": float(wald_result.pvalue),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Reporting (shared formatting so BTC and ETH reports are directly comparable)
# ---------------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def audit_split(name, df, native_metrics=None):
    print_header(f"{name} — Baseline Comparison & Statistical Significance")
    if native_metrics is not None:
        print("Cross-check: each model's OWN native test window (matches the notebook's reported numbers exactly):")
        for col in ["GARCH", "HAR", "EWMA"]:
            m = native_metrics[col]
            print(f"  {col:6s} (native)  RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  R2={m['R2']:.4f}")
        print("(Below: metrics recomputed on the date-aligned intersection of all three models' test windows —")
        print(" slightly different from native because HAR's own window starts later — required so the")
        print(" Diebold-Mariano/Mincer-Zarnowitz tests below compare identical observations.)\n")

    baseline_metrics = rmse_mae_r2(df["y_true"], df["Naive"])
    baseline_qlike = qlike_mean(df["y_true"], df["Naive"])
    print(f"{'Naive Persistence (baseline)':32s} RMSE={baseline_metrics['RMSE']:.6f}  MAE={baseline_metrics['MAE']:.6f}  "
          f"R2={baseline_metrics['R2']:.4f}  QLIKE={baseline_qlike:.4f}")

    model_metrics = {}
    for col in ["GARCH", "HAR", "EWMA"]:
        m = rmse_mae_r2(df["y_true"], df[col])
        q = qlike_mean(df["y_true"], df[col])
        model_metrics[col] = m
        rmse_red = pct_reduction(baseline_metrics["RMSE"], m["RMSE"])
        mae_red = pct_reduction(baseline_metrics["MAE"], m["MAE"])
        qlike_red = pct_reduction(baseline_qlike, q)
        print(f"{col:32s} RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  R2={m['R2']:.4f}  QLIKE={q:.4f}   "
              f"(RMSE {rmse_red:+.1f}%, MAE {mae_red:+.1f}%, QLIKE {qlike_red:+.1f}% vs naive)")

    print("\nDiebold-Mariano tests vs. naive persistence, squared-error loss (H0: equal predictive accuracy):")
    for col in ["GARCH", "HAR", "EWMA"]:
        dm = diebold_mariano(df["y_true"], df["Naive"], df[col], loss="squared", name1="Naive", name2=col)
        sig = "***" if dm["p_value"] < 0.01 else "**" if dm["p_value"] < 0.05 else "*" if dm["p_value"] < 0.10 else "n.s."
        print(f"  Naive vs {col:6s}  DM={dm['dm_stat']:+.3f}  p={dm['p_value']:.4f} {sig:5s} n={dm['n_obs']:6d}  favors: {dm['favors']}")

    print("\nDiebold-Mariano tests vs. naive persistence, QLIKE loss (the volatility-forecasting standard):")
    for col in ["GARCH", "HAR", "EWMA"]:
        dm = diebold_mariano(df["y_true"], df["Naive"], df[col], loss="qlike", name1="Naive", name2=col)
        sig = "***" if dm["p_value"] < 0.01 else "**" if dm["p_value"] < 0.05 else "*" if dm["p_value"] < 0.10 else "n.s."
        print(f"  Naive vs {col:6s}  DM={dm['dm_stat']:+.3f}  p={dm['p_value']:.4f} {sig:5s} n={dm['n_obs']:6d}  favors: {dm['favors']}")

    print("\nPairwise Diebold-Mariano, squared-error loss (model vs. model):")
    cols = ["GARCH", "HAR", "EWMA"]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            dm = diebold_mariano(df["y_true"], df[cols[i]], df[cols[j]], loss="squared", name1=cols[i], name2=cols[j])
            sig = "***" if dm["p_value"] < 0.01 else "**" if dm["p_value"] < 0.05 else "*" if dm["p_value"] < 0.10 else "n.s."
            print(f"  {cols[i]:6s} vs {cols[j]:6s}  DM={dm['dm_stat']:+.3f}  p={dm['p_value']:.4f} {sig:5s}  favors: {dm['favors']}")
    print("  (* p<0.10, ** p<0.05, *** p<0.01, n.s. = not significant)")

    return baseline_metrics, model_metrics


def audit_directional(name, df, train_mean_vol):
    print_header(f"{name} — Directional Accuracy (mean-reversion-aware)")
    d = directional_diagnostics(df, train_mean_vol)
    print(f"Mean-reversion-only baseline hit rate: {d['mean_reversion_baseline_hit_rate']*100:.2f}%  "
          f"(always guesses the move is toward the training-period mean, {train_mean_vol:.6f})")
    for col in ["GARCH", "HAR", "EWMA"]:
        v = d[col]
        flag = "  <-- ABOVE 62% SUSPICIOUS-LEAKAGE THRESHOLD" if v["suspicious"] else ""
        beats = "beats" if v["hit_rate"] > d["mean_reversion_baseline_hit_rate"] else "does NOT beat"
        print(f"  {col:6s} hit rate: {v['hit_rate']*100:6.2f}%  (n={v['n']:6d})  — {beats} the mean-reversion baseline{flag}")
    return d


def audit_mz(name, df):
    print_header(f"{name} — Explained Variance (Mincer-Zarnowitz regression)")
    for col in ["GARCH", "HAR", "EWMA"]:
        mz = mincer_zarnowitz(df["y_true"], df[col], label=col)
        calib = "well-calibrated (fails to reject alpha=0, beta=1)" if mz["joint_calibration_pvalue"] > 0.05 \
            else "MISCALIBRATED (rejects alpha=0, beta=1 at 5%)"
        print(f"  {col:6s}  alpha={mz['alpha']:+.6f} (se={mz['alpha_se']:.6f})   "
              f"beta={mz['beta']:+.4f} (se={mz['beta_se']:.4f})   R2={mz['r2']:.4f}")
        print(f"          joint calibration test p-value={mz['joint_calibration_pvalue']:.4f}  -> {calib}")
