# Bitcoin Volatility Prediction

**Data Science Capstone** — forecasting Bitcoin's realised volatility from hourly market data using time-series-native models.

---

## What this project does

Most Bitcoin modelling projects try to predict **where the price is going**. This one deliberately does not. It predicts **volatility** — *how much* the price is likely to move, in either direction.

An analogy: forecasting price is like predicting exactly where a boat will be tomorrow. Forecasting volatility is like predicting how rough the sea will be. The second question is more tractable, and for anyone managing risk it is usually the more useful one.

**Target variable:** 7-day realised volatility — the rolling standard deviation of hourly returns over a 168-hour window, updated every hour.

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
- **History:** 10 years
- **Fetch-once design:** the notebook pulls the history a single time and writes `data/btc_hourly_ohlcv.csv`. Every subsequent cell reads that CSV rather than the live API, so re-running is fast and fully reproducible.

**Engineered features:** hourly returns, log returns, rolling volatility, moving averages, volume change, EMA (12/26) and their normalised spread, RSI-14, ATR-14, Bollinger Band width, MACD histogram — all lagged 1–3 hours so the model only ever sees information available before the prediction time.

---

## Models

Every model here is **time-series-native by construction** — time is built into the model's own mathematics, not supplied to it as hand-engineered columns.

> **Why no Random Forest or XGBoost?** Generic ML models have no internal notion of time; to them each row is an unordered bag of numbers. They "see" time only through lag columns the analyst creates manually. Since this project is framed as a quantitative time-series study, models whose structure is inherently temporal were chosen instead.

| Model | Base version | Improved version | What it models |
| --- | --- | --- | --- |
| **1. GARCH family** | GARCH(1,1) | EGARCH / GJR-GARCH (selected by AIC) | Conditional variance, with a leverage term for down-move asymmetry |
| **2. HAR-RV** | HAR-RV (day/week/month) | HAR-RV-X + Ridge | Volatility across three time horizons, plus an exogenous ATR regressor |
| **3. LSTM** | 1 layer, 50 units, 30h lookback | Tuned units / dropout / lookback | A learned nonlinear mapping over the raw sequence |

**Baselines:** naive persistence (predict tomorrow = today) and Holt-Winters exponential smoothing (which additionally tests whether the hour-of-day seasonality found in EDA is exploitable).

---

## Validation

**Retrospective** — strict chronological 80/20 split, with `TimeSeriesSplit` for hyperparameter tuning. Never a random shuffle: on autocorrelated data, a shuffled split lets the model train on rows sitting between its own test rows, inflating scores while learning nothing generalisable.

**Prospective (live)** — the stronger test. Trained models are saved to `models/`, and a dedicated notebook section fetches genuinely new hourly candles, logs a prediction for each, and scores them against reality once 168 hours have elapsed and the true volatility becomes computable.

> A historical hold-out can still flatter a model — through subtle leakage, or through the analyst iterating against the test set while developing. Predictions logged *before* the outcome exists cannot be fooled.

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

Run the notebook top to bottom. The first execution fetches 10 years of hourly data from Binance, which takes a while — subsequent runs read the cached CSV and are much faster.

**To use the live testing section:** run the notebook once to train and save the models, then re-run *only* the Live Testing section on later days. Each run fetches new candles and logs more predictions. After a week has passed, those predictions become scoreable against real outcomes.

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
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Nelson, D. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach.* Econometrica, 59(2).

---

**Author:** Mohammed Hejairi · Cohort DSB2
