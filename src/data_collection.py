"""
data_collection.py
------------------
Coleta dados históricos do IBOVESPA e S&P500 via Yahoo Finance.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_series(ticker: str, name: str, start: str, end: str) -> pd.Series:
    data = yf.Ticker(ticker)
    df = data.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}.")
    series = df["Close"].rename(name)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    logger.info(f"{name}: {len(series)} observações ({series.index[0].date()} → {series.index[-1].date()})")
    return series


def collect_data(
    start: str = "2005-01-01",
    end: str | None = None,
    save_path: str = "data/series_diarias.csv",
    **kwargs,
) -> pd.DataFrame:
    end = end or datetime.today().strftime("%Y-%m-%d")

    ibov  = fetch_series("^BVSP", "IBOVESPA", start, end)
    sp500 = fetch_series("^GSPC", "SP500",    start, end)

    df = pd.concat([ibov, sp500], axis=1, join="inner").dropna()
    df.index.name = "Date"
    logger.info(f"Séries alinhadas: {len(df)} dias em comum.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    logger.info(f"Dados salvos em: {save_path}")
    return df


if __name__ == "__main__":
    df = collect_data(start="2005-01-01")
    print(df.tail())
