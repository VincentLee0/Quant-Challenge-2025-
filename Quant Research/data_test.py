import yfinance as yf
from datetime import date, timedelta
import pandas as pd


def get_processed_stock_data(ticker='AAPL', years=2, remove_year=2025, remove_month=5):

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    original_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    processed_df = original_data[condition]
    return processed_df


get_processed_stock_data()
