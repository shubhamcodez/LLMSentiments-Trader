import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Load the stocks data
stocks_df = pd.read_csv("data/price_data.csv")

# Load the portfolio positions data
benchmark_df = pd.read_csv("data/benchmark_news_portfolio.csv")
our_portfolio_df = pd.read_csv("data/portfolio_positions.csv")

# Convert 'timestamp' to datetime format for stocks_df and our_portfolio_df
stocks_df['timestamp'] = pd.to_datetime(stocks_df['timestamp'], format='%H:%M:%S', errors='coerce')
our_portfolio_df['timestamp'] = pd.to_datetime(our_portfolio_df['timestamp'], format='%H:%M:%S', errors='coerce')

# Convert benchmark_df timestamp to match the format of stocks_df
benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
benchmark_df['timestamp'] = benchmark_df['timestamp'].dt.strftime('%H:%M:%S')
benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'], format='%H:%M:%S', errors='coerce')

# Check for conversion errors and drop rows with NaT values
for df_name, df in [("stocks", stocks_df), ("benchmark", benchmark_df), ("our portfolio", our_portfolio_df)]:
    if df['timestamp'].isnull().any():
        print(f"Warning: Some timestamps could not be converted in the {df_name} DataFrame. Dropping rows with NaT values.")
        df.dropna(subset=['timestamp'], inplace=True)

# Handle duplicates (average stock prices)
stocks_df = stocks_df.groupby('timestamp').mean().reset_index()
stocks_df.fillna(0, inplace=True)

# Merge the DataFrames on 'timestamp'
merged_benchmark = pd.merge(benchmark_df, stocks_df, on='timestamp', how='inner')
merged_our_portfolio = pd.merge(our_portfolio_df, stocks_df, on='timestamp', how='inner')

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(merged_df, starting_balance=200000):
    stocks = ['AAPL', 'TSLA', 'INTC', 'NVAX', 'FIVN', 'EQIX', 'MIRM']
    
    merged_df['Cash Balance'] = starting_balance
    merged_df['Total Position Value'] = starting_balance
    
    for stock in stocks:
        merged_df[f'{stock} Invested'] = 0
    
    for i, row in merged_df.iterrows():
        if i == 0:
            continue
        
        prev_row = merged_df.iloc[i-1]
        cash_balance = prev_row['Cash Balance']
        
        for stock in stocks:
            position_change = row[f'{stock}_x'] - prev_row[f'{stock}_x']
            stock_price = row[f'{stock}_y']
            
            if position_change > 0:  # Buying
                money_spent = position_change * stock_price
                cash_balance -= money_spent
                merged_df.at[i, f'{stock} Invested'] = prev_row[f'{stock} Invested'] + money_spent
            elif position_change < 0:  # Selling
                money_received = -position_change * stock_price
                cash_balance += money_received
                merged_df.at[i, f'{stock} Invested'] = prev_row[f'{stock} Invested'] * (row[f'{stock}_x'] / prev_row[f'{stock}_x'])
            else:
                merged_df.at[i, f'{stock} Invested'] = prev_row[f'{stock} Invested']
        
        merged_df.at[i, 'Cash Balance'] = cash_balance
    
    for i, row in merged_df.iterrows():
        total_value = row['Cash Balance']
        for stock in stocks:
            total_value += row[f'{stock}_x'] * row[f'{stock}_y']
        merged_df.at[i, 'Total Position Value'] = total_value
    
    merged_df['PnL'] = merged_df['Total Position Value'] - starting_balance
    merged_df['Returns'] = merged_df['Total Position Value'].pct_change()
    
    window = 20
    merged_df['Rolling Volatility'] = merged_df['Returns'].rolling(window=window).std() * np.sqrt(252) * 100
    merged_df['Rolling Sharpe'] = (merged_df['Returns'].rolling(window=window).mean() * 252) / (merged_df['Rolling Volatility'] / 100)
    merged_df['Drawdown'] = ((merged_df['Total Position Value'] / merged_df['Total Position Value'].cummax()) - 1) * 100
    
    return merged_df

# Calculate metrics for both portfolios
benchmark_metrics = calculate_portfolio_metrics(merged_benchmark)
our_portfolio_metrics = calculate_portfolio_metrics(merged_our_portfolio)

# Create comparative plots
plot_titles = [
    'Total Position Value Over Time',
    'Profit and Loss (PnL) Over Time',
    'Rolling Sharpe Ratio (Window: 20)',
    'Rolling Volatility (Window: 20)',
    'Drawdown Over Time'
]

plot_data = [
    ('Total Position Value', '$'),
    ('PnL', '$'),
    ('Rolling Sharpe', ''),
    ('Rolling Volatility', '%'),
    ('Drawdown', '%')
]

for i, (title, (column, unit)) in enumerate(zip(plot_titles, plot_data)):
    plt.figure(figsize=(12, 6))
    plt.plot(benchmark_metrics['timestamp'], benchmark_metrics[column], label='Benchmark Portfolio', color='#1f77b4')
    plt.plot(our_portfolio_metrics['timestamp'], our_portfolio_metrics[column], label='Our Portfolio', color='#ff7f0e')
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel(f'{column} ({unit})', fontsize=14)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'comparative_performance_{i + 1}.png', dpi=300)
    plt.close()

# Print summary statistics
def print_portfolio_summary(metrics, name):
    print(f"\n{name} Portfolio Summary:")
    print(f"Starting Balance: ${200000:,.2f}")
    print(f"Ending Balance: ${metrics['Total Position Value'].iloc[-1]:,.2f}")
    print(f"Total Profit/Loss: ${metrics['PnL'].iloc[-1]:,.2f}")
    
    cumulative_returns = (metrics['Total Position Value'].iloc[-1] - 200000) / 200000
    final_sharpe_ratio = metrics['Rolling Sharpe'].replace(0, np.nan).dropna().mean()
    average_volatility = metrics['Rolling Volatility'].replace(0, np.nan).dropna().mean()
    max_drawdown = metrics['Drawdown'].min()
    
    print(f"Total Return: {cumulative_returns:.2%}")
    print(f"Final Sharpe Ratio: {final_sharpe_ratio:.2f}")
    print(f"Average Volatility: {average_volatility:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

print_portfolio_summary(benchmark_metrics, "Benchmark")
print_portfolio_summary(our_portfolio_metrics, "Our")