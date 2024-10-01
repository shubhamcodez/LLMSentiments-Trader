import databento as db
import pandas as pd

# Create a client for Databento API
client = db.Historical("your key")

# Define the NASDAQ stock symbols
nasdaq_stocks = ['AAPL', 'TSLA', 'INTC', 'NVAX', 'FIVN', 'EQIX', 'MIRM']

# Function to fetch stock data
def get_data(symbols, dataset):
    try:
        return client.timeseries.get_range(
            dataset=dataset,
            schema="ohlcv-1m",
            symbols=symbols,
            start="2024-05-12T21:00:00",
            end="2024-05-13T22:00:00"
        ).to_df()
    except Exception as e:
        print(f"Error fetching data for {dataset}: {e}")
        return pd.DataFrame()

# Fetch data for NASDAQ stocks
df = get_data(nasdaq_stocks, "XNAS.ITCH")

# Print column names
print("Columns in the DataFrame:")
print(df.columns.tolist())

# Check if the index is 'ts_event'
if isinstance(df.index, pd.DatetimeIndex):
    # If 'ts_event' is the index, convert it from nanoseconds to datetime and format it
    df.index = pd.to_datetime(df.index, unit='ns', utc=True)
    
    # Extract only the time (hours, minutes, and seconds)
    df['timestamp'] = df.index.strftime('%H:%M:%S')

    # Save the DataFrame with the formatted 'timestamp' column to a CSV file
    df.to_csv("price_data.csv", index=False)
    print("Formatted data saved to formatted_stocks_data.csv")

    # Print the first few rows to verify the result
    print(df[['timestamp']].head())

    # Print additional info about fetched data
    print(f"Total rows of data fetched: {len(df)}")
    print(f"Unique symbols in the data: {df['symbol'].nunique()}")
    print(f"Symbols with data: {', '.join(df['symbol'].unique())}")
    
    # Print timestamp range
    print(f"Timestamp range: from {df.index.min()} to {df.index.max()}")
else:
    print("The index is not a DatetimeIndex or 'ts_event' was not found in the data.")
