import os
import csv
def WriteOutTrades(df):
    """Writes out the trades to CSV file. Writes header only if file does not exist."""

    file_exists = os.path.isfile("data/dataLogs/TradeOnly.csv")

    with open("data_log.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        
        if not file_exists:  # File doesn't exist , write header
            writer.writerow(["Time", "Closing Price", "RSI Value", "Stochastic RSI"])
        
        # Write latest data row
        writer.writerow([df["timeStamp"].iloc[-1],
                         df["close"].iloc[-1],
                         df["rsi"].iloc[-1],
                         df["stoch_rsi"].iloc[-1]])
    
    print(f'{df["timeStamp"].iloc[-1]}, {df["close"].iloc[-1]}, {df["rsi"].iloc[-1]:.2f}, {df["stoch_rsi"].iloc[-1]:.2f}')