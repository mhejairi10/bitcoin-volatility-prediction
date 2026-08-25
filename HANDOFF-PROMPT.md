I'm continuing a Data Science capstone project that was built in a separate Claude conversation. All the deliverables already exist — I've extracted them into this folder. I need help setting up the git repo, running the notebook, and fixing whatever breaks.

## The project

**Bitcoin Volatility Prediction** — forecasting BTC's 7-day realised volatility (rolling standard deviation of hourly returns over a 168-hour window, updated hourly) from 10 years of hourly OHLCV data pulled from Binance via CCXT.

Crucially: this predicts **how much** the price will move, not **where** it goes. It's a regression problem (continuous σ), deliberately not classification — downstream users of a volatility number (Black-Scholes, VaR, position sizing) need the actual value, not a "high/low" bucket.

## What's in this folder

```
notebooks/capstone-btc-volatility.ipynb    # The full technical report — main deliverable
docs/time-series-explainer.pdf             # Concepts + maths + model rationale
presentation/capstone-btc-volatility.pptx  # 13-slide deck
data/                                       # Empty; CSVs generated on first run (gitignored)
models/                                     # Empty; saved models (gitignored)
README.md, requirements.txt, .gitignore, SETUP-GITHUB.md
```

## Key design decisions — please preserve these, don't relitigate unless I ask

**Models (3, all time-series-native):**
1. **GARCH family** — GARCH(1,1) base → improved to EGARCH/GJR-GARCH, selected by AIC (captures the leverage effect: bad news raises volatility more than equally-sized good news)
2. **HAR-RV** → improved to **HAR-RV-X** (adds lagged ATR as an exogenous regressor) with Ridge regularisation tuned via GridSearchCV
3. **LSTM** — base (50 units, 30h lookback) → tuned over units/dropout/lookback

**Baselines:** naive persistence, and Holt-Winters exponential smoothing (which also tests whether the hour-of-day seasonality found in EDA is exploitable).

**Deliberately excluded: Linear Regression, Random Forest, XGBoost.** My instructor pushed for a quant-flavoured time-series project, and these have no built-in notion of time — they only see sequence structure through hand-engineered lag columns. Please don't suggest adding them back.

**Methodology that must not be broken:**
- Every feature lagged 1–3 hours — no lookahead leakage, ever
- Strict chronological 80/20 train/test split, never a random shuffle
- `TimeSeriesSplit` for cross-validation, never random K-fold
- `StandardScaler` fit on training data only
- Zero-volume candles produce `inf` from `pct_change()` (not NaN), so infs are converted to NaN *before* any rolling calculation runs — this was a real bug that surfaced as a cryptic StandardScaler error

**Live testing section:** the notebook saves trained models to `models/`, then a dedicated section fetches genuinely new hourly candles, logs predictions to `predictions_log.csv`, and scores them against reality once 168 hours have elapsed. This is real prospective validation, not a simulated hold-out.

## What I need help with

1. **Set up the git repo and push to GitHub.** `SETUP-GITHUB.md` has the steps — help me run them. I need to create an empty repo on GitHub first (no README/gitignore/license, or the first push conflicts).

2. **Fill in my placeholders.** `[Your Name]` and `[Cohort Ref.]` appear in: README.md (last line), the notebook's title cell, and the deck's title slide.

3. **Run the notebook end to end.** It has never been executed against live data — it was written in a sandbox with no exchange API access, so every cell is syntax-checked but untested at runtime. Expect things to break. The first run fetches 10 years of hourly data and will take a while.

4. **Fill in the results.** Once it runs, the notebook auto-generates a model comparison table. The presentation has `—` placeholders in its metric tables that need those real numbers.

## Realistic expectations

Volatility is driven substantially by information absent from price history (news, regulation, exchange failures), so there's a hard ceiling on accuracy. Modest improvements over naive persistence are the honest expectation. If a model comes back with near-perfect accuracy, treat it as evidence of leakage and help me hunt down the bug — don't celebrate it.

Please start by reading `notebooks/capstone-btc-volatility.ipynb` to get the full picture, then tell me you're up to speed before we begin.
