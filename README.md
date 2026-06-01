# Stock-Predictor

Stock market regime detection on Microsoft (MSFT) daily history using a Gaussian Hidden Markov Model.

CS179 (Probabilistic Graphical Models) course project.

## Approach

Fit a Gaussian HMM on a 2-D observation sequence — daily log returns and 20-day annualized realized volatility — to infer latent market regimes (bull / bear / high-vol). Select the number of hidden states by BIC, evaluate via held-out log-likelihood on a 2018+ test window, and characterize each regime by its emission mean and self-transition probability.

## Files

- `data_loader.py` — load Yahoo Finance OHLCV CSV; compute log returns, realized volatility, volume z-score; train/test split.
- `hmm_model.py` — fit Gaussian HMM with multiple random restarts; model selection over $K \in \{2,3,4,5\}$ by BIC; regime summary.
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Place the MSFT OHLCV CSV (from [Yahoo Finance](https://finance.yahoo.com/quote/MSFT/history) or the equivalent Kaggle dataset) at the project root as `MSFT.csv`.

## Run

```bash
python hmm_model.py MSFT.csv
```

This runs the full pipeline end-to-end: load → feature engineering → train/test split at 2018-01-01 → sweep $K$ → print BIC table → held-out log-likelihood → per-regime summary.

## Data

Daily OHLCV for MSFT from IPO (1986) to present. Source: Yahoo Finance. Not committed — see `.gitignore`.
