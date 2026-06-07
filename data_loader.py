"""Load a ticker's OHLCV history and compute regime-detection features."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError(f"CSV missing 'Date' column. Got: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df.rename(columns={price_col: "price"})

    keep = [c for c in ["Open", "High", "Low", "Close", "price", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["price"])
    return df


def compute_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["price"] / out["price"].shift(1))
    out["realized_vol"] = out["log_return"].rolling(vol_window).std() * np.sqrt(TRADING_DAYS)

    if "Volume" in out.columns:
        log_vol = np.log(out["Volume"].replace(0, np.nan))
        out["volume_z"] = (log_vol - log_vol.rolling(60).mean()) / log_vol.rolling(60).std()

    return out.dropna()


def train_test_split(
    df: pd.DataFrame,
    split_date: str = "2022-01-01",
    train_start: str | None = "2000-01-01",
):
    """Split chronologically. Train = [train_start, split_date), test = [split_date, end].

    train_start bounds how far back the training window reaches (None = use all history).
    """
    split = pd.Timestamp(split_date)
    train = df.loc[df.index < split]
    if train_start is not None:
        train = train.loc[train.index >= pd.Timestamp(train_start)]
    test = df.loc[df.index >= split]
    return train.copy(), test.copy()


def to_observations(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[feature_cols].to_numpy(dtype=float)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "MSFT.csv"
    raw = load_ohlcv(path)
    feats = compute_features(raw)
    train, test = train_test_split(feats)
    print(f"Loaded {len(raw)} rows | features {feats.shape} | train {len(train)} | test {len(test)}")
    print(feats[["log_return", "realized_vol"]].describe())
