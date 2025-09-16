import yfinance as yf
from datetime import date, timedelta
import pandas as pd


ticker_symbol = 'AAPL'
end_date = date.today()
start_date = end_date - timedelta(days=730)  # Approx. 2 years
print(
    f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...\n")
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

print("--- Original Data ---")
print(stock_data.tail())
print(f"\nOriginal data contains {len(stock_data)} trading days.\n")


year_to_remove = 2025
month_to_remove = 5  # 5 = May

print(f"--- Removing data for month {month_to_remove}/{year_to_remove} ---")

condition = ~((stock_data.index.month == month_to_remove)
              & (stock_data.index.year == year_to_remove))

data_with_month_removed = stock_data[condition]
print(data_with_month_removed.loc['2025-04'].tail())
print(data_with_month_removed.loc['2025-06'].head())
