# Pushing this to GitHub from your Mac

Everything in this folder is ready to go. Follow these steps.

---

## Step 1 — Create an empty repo on GitHub

1. Go to **https://github.com/new**
2. **Repository name:** `bitcoin-volatility-prediction` (or whatever you prefer)
3. **Description:** `Data Science Capstone — forecasting Bitcoin realised volatility with GARCH, HAR-RV, and LSTM`
4. Choose **Public** or **Private**
5. **Important:** leave *"Add a README file"*, *"Add .gitignore"*, and *"Choose a license"* all **unchecked** — this folder already has those, and pre-adding them creates a merge conflict on your first push.
6. Click **Create repository**

GitHub will then show you a page with a URL like:
```
https://github.com/YOUR-USERNAME/bitcoin-volatility-prediction.git
```
Keep that tab open.

---

## Step 2 — Fill in your details

Before pushing, replace the placeholders in two files:

- **`README.md`** — last line: `[Your Name]` and `[Ref.]`
- **`notebooks/capstone-btc-volatility.ipynb`** — the title cell at the very top: `[YOUR FULL NAME HERE]` and `[COHORT REF. HERE]`

Also worth doing at some point (not required to push): the **presentation** has `[Your Name]` on the title slide, and `—` placeholders in the metric tables that you fill in after running the notebook.

---

## Step 3 — Push from Terminal

Open Terminal, `cd` into this folder, then run:

```bash
# Initialise the repo
git init
git branch -M main

# Stage and commit everything
git add .
git commit -m "Initial commit: BTC volatility prediction capstone"

# Connect to your GitHub repo (use YOUR url from Step 1)
git remote add origin https://github.com/YOUR-USERNAME/bitcoin-volatility-prediction.git

# Push
git push -u origin main
```

If Git asks for a password, note that **GitHub no longer accepts account passwords over HTTPS.** You need a Personal Access Token:

1. Go to **https://github.com/settings/tokens**
2. **Generate new token (classic)** → tick the **`repo`** scope → generate
3. Copy the token and paste it when Git prompts for a password

*(Alternatively, install the GitHub CLI — `brew install gh`, then `gh auth login` — which handles authentication for you.)*

---

## Step 4 — Verify

Refresh your GitHub repo page. You should see:

```
notebooks/     capstone-btc-volatility.ipynb
docs/          time-series-explainer.pdf
presentation/  capstone-btc-volatility.pptx
data/          (empty — CSVs are gitignored)
models/        (empty — saved models are gitignored)
README.md
requirements.txt
```

GitHub renders `.ipynb` files natively, so your notebook will be readable directly in the browser — useful when sharing it with an instructor or reviewer.

---

## About the empty `data/` and `models/` folders

These are intentionally empty, with `.gitignore` rules excluding their contents. Two reasons:

1. **Size.** Ten years of hourly OHLCV data plus a saved LSTM would bloat the repo considerably.
2. **Reproducibility.** The notebook regenerates both on first run. Anyone cloning the repo gets fresh data straight from the exchange rather than a stale snapshot.

The `.gitkeep` files exist purely so the empty folders are tracked by Git — otherwise Git would drop them entirely.

---

## Later updates

Once the repo exists, pushing changes is just:

```bash
git add .
git commit -m "Describe what changed"
git push
```
