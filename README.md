# Bitcoin Volatility Prediction

**Data Science Capstone** — forecasting Bitcoin's realised volatility from hourly market data using time-series-native models.

---

## What this project does

Most Bitcoin modelling projects try to predict **where the price is going**. This one deliberately does not. It predicts **volatility** — *how much* the price is likely to move, in either direction.

An analogy: forecasting price is like predicting exactly where a boat will be tomorrow. Forecasting volatility is like predicting how rough the sea will be. The second question is more tractable, and for anyone managing risk it is usually the more useful one.

**Target variable:** 7-day realised volatility, forecast 168 hours ahead of the features used to predict it — the rolling standard deviation of hourly returns over a 168-hour window, updated every hour. The horizon is deliberately set equal to the window length: a shorter horizon would share almost all its return observations with the most recently known volatility value, making the forecast trivially easy rather than genuine.

**Problem type:** Regression. Downstream consumers of a volatility number (Black-Scholes σ, Value-at-Risk, position sizing) need the actual continuous value, not a "high/low" category.

---

## Why volatility matters

| Use case | How volatility enters |
| --- | --- |
| Option pricing | Black-Scholes takes σ as a required input |
| Value-at-Risk | Loss estimates scale directly with σ |
| Position sizing | Trade smaller when choppy, larger when calm |
| Exchange margin | Collateral requirements rise with volatility |

---

## Data

- **Source:** [CCXT](https://docs.ccxt.com/) → Binance, `BTC/USDT`
- **Granularity:** hourly candles
- **History:** fixed range, requested from 2010-01-01 (Bitcoin's own origin) through 2026-01-01 — Binance itself didn't exist before mid-2017, so the fetch naturally starts wherever Binance's real history begins (~8.4 years of hourly candles). Explicit dates rather than "N years back from whenever this runs", so the dataset is identical on every run
- **Fetch-once design:** the notebook pulls the history a single time and writes `data/btc_hourly_ohlcv.csv`. Every subsequent cell reads that CSV rather than the live API, so re-running is fast and fully reproducible.

**Engineered features:** hourly returns, log returns, rolling volatility, moving averages, volume change, EMA (12/26) and their normalised spread, RSI-14, ATR-14, Bollinger Band width, MACD histogram — all lagged 1–3 hours so the model only ever sees information available before the prediction time.

---

## Models

Every model here is **time-series-native by construction** — time is built into the model's own mathematics, not supplied to it as hand-engineered columns.

> **Why no Random Forest, XGBoost, or LSTM?** Generic ML models (including an LSTM originally explored in this slot) have no internal notion of time as an econometric or risk-management concept; they either see time only through hand-engineered lag columns, or — for a sequence model like LSTM — learn an opaque, unstructured mapping with no financial theory behind it. Every model kept here has a genuine econometric or risk-management pedigree instead.

| Model | Base version | Improved version | What it models |
| --- | --- | --- | --- |
| **1. GARCH family** | GARCH(1,1) | EGARCH / GJR-GARCH (selected by AIC) | Conditional variance, with a leverage term for down-move asymmetry |
| **2. HAR-RV** | HAR-RV (day/week/month) | HAR-RV-X + Ridge | Volatility across three time horizons, plus an exogenous ATR regressor |
| **3. EWMA** | λ = 0.94 (RiskMetrics industry default) | λ backtested via `TimeSeriesSplit` | The same exponentially-weighted volatility estimator risk desks use, per J.P. Morgan's *RiskMetrics* (1996) |

**Baselines:** naive persistence (predict tomorrow = today) and Holt-Winters exponential smoothing (which additionally tests whether the hour-of-day seasonality found in EDA is exploitable).

---

## Validation

**Retrospective** — strict chronological 80/20 split, with `TimeSeriesSplit` for hyperparameter tuning. Never a random shuffle: on autocorrelated data, a shuffled split lets the model train on rows sitting between its own test rows, inflating scores while learning nothing generalisable.

**Prospective (live)** — the stronger test. Trained models are saved to `models/`, and a dedicated notebook section fetches a fixed, later period — 2026-01-02 through yesterday — that was never part of the training data at all (training stops 2026-01-01), and generates a prediction for every hour in it. Deliberately wide rather than a single week: a 168-hour-ahead forecast can only be scored once 168 real hours have elapsed, so a multi-month window means many more already-elapsed, already-scoreable predictions instead of a handful.

> A historical hold-out can still flatter a model — through subtle leakage, or through the analyst iterating against the test set while developing. A later period the model never saw during training or tuning is a stronger check.

---

## Repository structure

```
.
├── notebooks/
│   └── capstone-btc-volatility.ipynb    # Full technical report
├── docs/
│   └── time-series-explainer.pdf        # Concepts, maths, and model rationale
├── presentation/
│   └── capstone-btc-volatility.pptx     # Slide deck
├── data/                                # CSVs generated on first run (gitignored)
├── models/                              # Saved models (gitignored)
├── requirements.txt
└── README.md
```

---

## Running it

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/capstone-btc-volatility.ipynb
```

Run the notebook top to bottom. The first execution fetches Binance's full BTC/USDT hourly history (~8.4 years), which takes a while — subsequent runs read the cached CSV and are much faster.

**Live testing section:** fetches the fixed 2026-01-02 → yesterday window once and evaluates every saved model against whatever predictions have already had 168 hours elapse — no re-running required, and more predictions become scoreable as later days pass.

---

## A note on expected results

Volatility is driven substantially by information absent from price history — regulatory announcements, exchange failures, macroeconomic news. No amount of feature engineering recovers what is not in the data, so there is a hard ceiling on achievable accuracy.

**Modest improvements over naive persistence are the realistic and honest expectation.** A model reporting near-perfect accuracy on this task should be treated as evidence of data leakage, not of success.

---

## References

- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics, 31(3).
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics, 7(2).
- Engle, R. (1982). *Autoregressive Conditional Heteroscedasticity...* Econometrica, 50(4).
- Glosten, L., Jagannathan, R., & Runkle, D. (1993). *On the Relation between the Expected Value and the Volatility...* Journal of Finance, 48(5).
- J.P. Morgan (1996). *RiskMetrics — Technical Document* (4th ed.).
- Nelson, D. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach.* Econometrica, 59(2).

---

**Author:** Mohammed Hejairi · Cohort DSB2
