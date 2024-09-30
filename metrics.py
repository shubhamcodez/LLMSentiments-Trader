import csp
import pandas as pd
import numpy as np
from csp.adapters.csv import CSVReader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

class PortfolioData(csp.Struct):
    timestamp: int
    AAPL: float
    TSLA: float
    SNY: float
    WFC: float
    INTC: float
    UBS: float
    BA: float
    CPR_MI: float
    VMW: float
    BCS: float
    T7731: float
    TTM: float
    HD: float
    FTT_TO: float
    NVAX: float
    AKER_OL: float
    AKE_PA: float
    SECUB_ST: float
    LXS_F: float
    CRR_UN_TO: float
    TRN_MI: float
    T8473: float
    PYCR: float
    CON_DE: float
    ARG_TO: float
    KXS_TO: float
    LOOMIS_ST: float
    CGX_TO: float
    DPM_TO: float
    LDO_MI: float
    AVDX: float
    BRCC: float
    SIS_TO: float
    INNV: float
    AMPS: float
    ZIP: float
    STC_V: float
    FIVN: float
    EQIX: float
    LSAK: float
    KELYB: float
    CXT: float
    PRA: float
    LAB: float
    PHI: float
    MIRM: float
    WPM: float
    FA: float

def time_converter(column, tz=None):
    def convert(row):
        v = row[column]
        dt = datetime.utcfromtimestamp(int(v)).replace(tzinfo=None)
        if tz is not None:
            dt = tz.localize(dt)
        return dt
    return convert

@csp.node
def calculate_portfolio_metrics(data: PortfolioData, starting_balance: float, risk_free_rate: float = 0.02):
    portfolio_values = []
    returns = []
    max_drawdown = 0
    peak = starting_balance

    def update(timestamp, *positions):
        nonlocal peak, max_drawdown
        
        total_value = sum(positions) + starting_balance
        portfolio_values.append(total_value)
        
        if len(portfolio_values) > 1:
            returns.append((total_value / portfolio_values[-2]) - 1)
        
        if total_value > peak:
            peak = total_value
        drawdown = (peak - total_value) / peak
        max_drawdown = max(max_drawdown, drawdown)
        
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

    return csp.map(update)(
        data.timestamp, data.AAPL, data.TSLA, data.SNY, data.WFC, data.INTC, data.UBS, data.BA, data.CPR_MI, data.VMW, 
        data.BCS, data.T7731, data.TTM, data.HD, data.FTT_TO, data.NVAX, data.AKER_OL, data.AKE_PA, data.SECUB_ST, 
        data.LXS_F, data.CRR_UN_TO, data.TRN_MI, data.T8473, data.PYCR, data.CON_DE, data.ARG_TO, data.KXS_TO, 
        data.LOOMIS_ST, data.CGX_TO, data.DPM_TO, data.LDO_MI, data.AVDX, data.BRCC, data.SIS_TO, data.INNV, data.AMPS, 
        data.ZIP, data.STC_V, data.FIVN, data.EQIX, data.LSAK, data.KELYB, data.CXT, data.PRA, data.LAB, data.PHI, 
        data.MIRM, data.WPM, data.FA
    )

@csp.graph
def portfolio_analysis(csv_path: str, starting_balance: float):
    reader = CSVReader(csv_path, time_converter('timestamp'), delimiter=' ')
    data = reader.subscribe(PortfolioData)
    return calculate_portfolio_metrics(data, starting_balance)

# Set up the graph
csv_path = "portfolio_positions.csv"  # Update this path
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