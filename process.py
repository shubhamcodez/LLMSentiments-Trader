import pandas as pd

# Load the CSV data
df = pd.read_csv("price_data.csv")

# Pivot the table so that 'timestamp' is the index and 'symbol' becomes the columns
pivot_df = df.pivot(index='timestamp', columns='symbol', values='close')

# Ensure all required symbols are present, filling missing symbols with NaN
required_symbols = ['AAPL', 'TSLA', 'INTC', 'NVAX', 'FIVN', 'EQIX', 'MIRM']

# Create missing columns if not present
for symbol in required_symbols:
    if symbol not in pivot_df.columns:
        pivot_df[symbol] = None  # Fill with None

# Reorder columns based on the required symbols list
pivot_df = pivot_df[required_symbols]

# Forward fill to replace NaN values with the last valid observation
pivot_df.ffill(inplace=True)
pivot_df.fillna(0)
# Check for any remaining NaN values after forward fill
if pivot_df.isnull().values.any():
    print("Some NaN values remain after forward filling. Consider additional filling methods.")

# Reset index to bring 'timestamp' back as a column
pivot_df.reset_index(inplace=True)

# Save the resulting DataFrame to a new CSV
pivot_df.to_csv("price_data.csv", index=False)

# Display the first few rows of the resulting DataFrame
print(pivot_df.head())
