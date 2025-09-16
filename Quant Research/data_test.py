import yfinance as yf
from datetime import date, timedelta
import pandas as pd


def fetch_stock_data(ticker, start, end):
    print(f"Fetching data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data


def remove_data_for_month(df, year, month):
    condition = ~((df.index.month == month) & (df.index.year == year))
    return df[condition]


if __name__ == "__main__":
    ticker_symbol = 'AAPL'
    end_date = date.today()
    start_date = end_date - timedelta(days=730)  # Approx. 2 years
    year_to_remove = 2025
    month_to_remove = 5  # 5 = May
    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)

    if not stock_data.empty:
        print("--- Original Data (Last 5 Days) ---")
        print(stock_data.tail())
        print(f"\nOriginal data contains {len(stock_data)} trading days.\n")
        data_with_month_removed = remove_data_for_month(
            stock_data, year_to_remove, month_to_remove)

        print(
            f"--- Removing data for month {month_to_remove}/{year_to_remove} ---")
