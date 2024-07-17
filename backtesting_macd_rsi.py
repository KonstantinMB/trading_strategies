import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Downloading the data
symbol = "BTC-USD"
start_date = "2023-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')
interval = "1h"
data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
data.dropna(inplace=True)

# Calculate MACD and RSI
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['MACDs'] = data['MACD'].ewm(span=9, adjust=False).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Generate signals
def determine_signal(row):
    if row['RSI'] > 70 and row['MACD'] < row['MACDs']:
        return -1  # Sell
    elif row['RSI'] < 30 and row['MACD'] > row['MACDs']:
        return 1   # Buy
    return 0

data['Signal'] = data.apply(determine_signal, axis=1)

# Add a debugging line to check how many signals are generated
print("Buy signals count:", data[data['Signal'] == 1].shape[0])
print("Sell signals count:", data[data['Signal'] == -1].shape[0])

# Backtesting logic
initial_capital = 10000.0
positions = []
cash = initial_capital
holding = False
entry_price = 0

# Simulate trading
for index, row in data.iterrows():
    if row['Signal'] == 1 and not holding:  # Buy signal and not already holding
        entry_price = row['Close']
        holding = True
    elif row['Signal'] == -1 and holding:  # Sell signal and currently holding
        exit_price = row['Close']
        profit = exit_price - entry_price
        cash += profit
        positions.append({'Entry': entry_price, 'Exit': exit_price, 'Profit': profit})
        holding = False

# Output results
final_capital = cash if not holding else cash + (data.iloc[-1]['Close'] - entry_price)
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
if positions:
    print("Trades executed:")
    for pos in positions:
        print(f"Entry: ${pos['Entry']:.2f}, Exit: ${pos['Exit']:.2f}, Profit: ${pos['Profit']:.2f}")
else:
    print("No trades executed.")
