import csp
import pandas as pd
import numpy as np
from csp.adapters.csv import CSVReader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    T7731: float  # 7731.T
    TTM: float
    HD: float
    FTT_TO: float  # FTT.TO
    NVAX: float
    AKER_OL: float  # AKER.OL
    AKE_PA: float  # AKE.PA
    SECUB_ST: float  # SECUB.ST
    LXS_F: float  # LXS.F
    CRR_UN_TO: float  # CRR.UN.TO
    TRN_MI: float
    T8473: float  # 8473.T
    PYCR: float
    CON_DE: float  # CON.DE
    ARG_TO: float  # ARG.TO
    KXS_TO: float  # KXS.TO
    LOOMIS_ST: float  # LOOMIS.ST
    CGX_TO: float  # CGX.TO
    DPM_TO: float  # DPM.TO
    LDO_MI: float
    AVDX: float
    BRCC: float
    SIS_TO: float  # SIS.TO
    INNV: float
    AMPS: float
    ZIP: float
    STC_V: float  # STC.V
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

@csp.node
def calculate_portfolio_metrics(data: PortfolioData, starting_balance: float, risk_free_rate: float = 0.02):
    portfolio_values = []
    returns = []
    max_drawdown = 0
    peak = starting_balance

    def update(timestamp, *positions):
        nonlocal peak, max_drawdown
        
        # Calculate total portfolio value (assuming price is 1 for simplicity)
        total_value = sum(positions) + starting_balance
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

    return csp.map(update)(
        data.timestamp, data.AAPL, data.TSLA, data.SNY, data.WFC, 
        data.INTC, data.UBS, data.BA, data.CPR_MI, data.VMW, data.BCS,
        data.T7731, data.TTM, data.HD, data.FTT_TO, data.NVAX,
        data.AKER_OL, data.AKE_PA, data.SECUB_ST, data.LXS_F,
        data.CRR_UN_TO, data.TRN_MI, data.T8473, data.PYCR,
        data.CON_DE, data.ARG_TO, data.KXS_TO, data.LOOMIS_ST,
        data.CGX_TO, data.DPM_TO, data.LDO_MI, data.AVDX,
        data.BRCC, data.SIS_TO, data.INNV, data.AMPS, data.ZIP,
        data.STC_V, data.FIVN, data.EQIX, data.LSAK, data.KELYB,
        data.CXT, data.PRA, data.LAB, data.PHI, data.MIRM,
        data.WPM, data.FA
    )

@csp.graph
def portfolio_analysis(csv_path: str, starting_balance: float):
    reader = CSVReader(csv_path, delimiter=' ')
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