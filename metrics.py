from jsonlines import InvalidLineError
import sys
import pandas as pd
import numpy as np
import csv
import transformers
import csp
import torch
from datetime import *
from csp.adapters.csv import *
from transformers import pipeline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PortfolioData(csp.Struct):
    ticker: str
    timestamp: int
    position: float

@csp.node
def calculate_portfolio_metrics(data: PortfolioData, starting_balance: float, risk_free_rate: float = 0.02):
    portfolio = {}
    portfolio_values = []
    returns = []
    max_drawdown = 0
    peak = starting_balance

    def update(ticker, timestamp, position):
        nonlocal peak, max_drawdown
        
        # Update portfolio
        portfolio[ticker] = position
        
        # Calculate total portfolio value (assuming price is 1 for simplicity)
        total_value = sum(portfolio.values()) + starting_balance
        portfolio_values.append(total_value)
        
        # Calculate return
        if len(portfolio_values) > 1:
            returns.append((total_value / portfolio_values[-2]) - 1)
        
        # Update max drawdown
        if total_value > peak:
            peak = total_value
        drawdown = (peak - total_value) / peak
        max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate metrics
        total_pnl = total_value - starting_balance
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        information_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns) if len(returns) > 1 else 0
        
        return {
            'timestamp': timestamp,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown
        }

    return csp.map(update)(data.ticker, data.timestamp, data.position)

@csp.graph
def portfolio_analysis(csv_path: str, starting_balance: float):
    reader = CSVReader(csv_path, delimiter=',')
    data = reader.subscribe(PortfolioData)
    return calculate_portfolio_metrics(data, starting_balance)

# Set up the graph
csv_path = "path/to/your/csv/file.csv"
starting_balance = 100000  # Example starting balance
graph = portfolio_analysis(csv_path, starting_balance)

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
lines = [ax1.plot([], [], label=metric)[0] for metric in ['Total Value', 'Total PnL']]
lines.extend([ax2.plot([], [], label=metric)[0] for metric in ['Sharpe Ratio', 'Information Ratio', 'Max Drawdown']])

ax1.set_ylabel('USD')
ax2.set_ylabel('Ratio')
ax2.set_xlabel('Time')

for ax in (ax1, ax2):
    ax.legend()
    ax.grid(True)

# Animation update function
def update(frame):
    data = frame
    for i, key in enumerate(['total_value', 'total_pnl', 'sharpe_ratio', 'information_ratio', 'max_drawdown']):
        lines[i].set_data(data['timestamp'], data[key])
    
    for ax in (ax1, ax2):
        ax.relim()
        ax.autoscale_view()
    
    return lines

# Run the animation
ani = FuncAnimation(fig, update, frames=graph, interval=100, blit=True)
plt.show()

# Run the graph
csp.run(graph)