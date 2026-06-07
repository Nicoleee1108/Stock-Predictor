"""Download daily OHLCV history for a basket of tickers into ./dataset/.

Each ticker is saved as dataset/<TICKER>.csv with columns matching what
data_loader.py expects: Date, Open, High, Low, Close, Adj Close, Volume.

Usage:
    python make_dataset.py                  # default basket, 2000-01-01 -> today
    python make_dataset.py SPY MSFT AAPL    # custom tickers
    python make_dataset.py --start 1990-01-01 SPY
"""

from __future__ import annotations

import argparse
import os

import yfinance as yf

# SPY = S&P 500 ETF -> cleanest "market regime" series (primary model).
# The single stocks are for comparison experiments.
DEFAULT_TICKERS = ["SPY", "MSFT", "AAPL"]

DATASET_DIR = "dataset"


def download(ticker: str, start: str, end: str | None) -> str:
    """Download one ticker and save to dataset/<ticker>.csv. Returns the path."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,  # keep both 'Close' and 'Adj Close'
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} (check ticker / network)")

    # yfinance may return MultiIndex columns when given a single ticker; flatten.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # Date becomes a column
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols]

    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, f"{ticker}.csv")
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV history into ./dataset/")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS, help="ticker symbols")
    parser.add_argument("--start", default="2000-01-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="end date (YYYY-MM-DD); default = today")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    for t in tickers:
        path = download(t, args.start, args.end)
        # report row count for a quick sanity check
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        print(f"  {t:6s} -> {path}  ({n} rows)")


if __name__ == "__main__":
    main()
