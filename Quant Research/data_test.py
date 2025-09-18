import yfinance as yf
from datetime import date, timedelta
import pandas as pd
import numpy as np


def get_processed_stock_data(ticker, years=2, months_to_remove=None):

    if months_to_remove is None:
        months_to_remove = []  # (2025, 5), (2025, 4),
        # (2025, 3), (2024, 12), (2024, 11)]

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)
    original_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    combined_mask = pd.Series(False, index=original_data.index)

    for year, month in months_to_remove:
        month_mask = (original_data.index.month == month) & (
            original_data.index.year == year)
        combined_mask = combined_mask | month_mask
    processed_df = original_data[~combined_mask]

    return processed_df
