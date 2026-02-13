import os
from dotenv import load_dotenv
from numpy import double
import pandas as pd
import torch
from AI.TradingEnv import TradingEnv
from AI.brain import trainDQN
from indicators.bb import Bollinger
from indicators.RSIIndicators import RSI, StochRSI
from indicators.volume import zVolume

load_dotenv() 

url_public = os.getenv("KRAKEN_PUBLIC")
pair = os.getenv("PAIR")
interval = int(os.getenv("INTERVAL")) # in seconds
candle = os.getenv("CANDLE") # in minutes
RSIPeriod = int(os.getenv("RSIPERIOD"))
sellThreshold = float(os.getenv("SELLTHRESHOLD"))
buyThreshold = float(os.getenv("BUYTHRESHOLD"))
startMoney = int(os.getenv("INITIALPAPERMONEY"))
lotSize = float(os.getenv("HOWMANYYOUWANT"))


# def load_data(filename, RSI_period=14):
#     df = pd.read_csv(filename)
#     df = RSI(df, RSI_period)              # add RSI column
#     df = StochRSI(df, RSI_period)         # add stochastic RSI
#     df = zVolume(df)              # normalized volume for DQN
#     df = df.dropna().reset_index(drop=True)
#     return df

def normalizeOHLC(df):
    """Normalizes OHLC data to between -1, 1 for the best learning for model"""

    # Create a shifted column for the previous close
    prev_close = df['close'].shift(1)

    # Calculate percentage change for each OHLC parameter relative to previous close
    df['nOpen']  = (df['open'] - prev_close) / prev_close
    df['nHigh']  = (df['high'] - prev_close) / prev_close
    df['nLow']   = (df['low'] - prev_close) / prev_close
    df['nClose'] = (df['close'] - prev_close) / prev_close
    df["nRSI"] = df["rsi"] / 100
    df["nStochRSI"] = df["stoch_rsi"] / 100
    df["nZVolume"] = df["zVolume"] / 3
    df = df.dropna().reset_index(drop=True)
    
    return df

def load_data(filename, RSI_period=14):
    """
    Loads Binance OHLCV CSV data and adds indicators:
    - RSI
    - Stochastic RSI
    - normalized volume (zVolume)
    - normalized OHLC for DQN
    """

    # Read CSV; assume Binance CSV with 12 columns
    df = pd.read_csv(
        filename,
        header=None,  # no header in CSV
        names=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
    )

    # Convert relevant columns to float
    df[["open","high","low","close","volume","quote_volume",
        "taker_buy_base_volume","taker_buy_quote_volume"]] = df[[
        "open","high","low","close","volume","quote_volume",
        "taker_buy_base_volume","taker_buy_quote_volume"
    ]].astype(float)

    # Add indicators
    df["rsi"] = RSI(df["close"], period=RSIPeriod)
    df["stoch_rsi"] = StochRSI(df["rsi"], period=RSIPeriod)

    # Add normalized volume (zVolume) — use base volume column
    df = zVolume(df)  # make sure zVolume can accept this argument

    df = Bollinger(df)

    # Normalize OHLC + indicators for DQN
    df = normalizeOHLC(df)

    # Drop NaNs created by indicators
    df = df.dropna().reset_index(drop=True)

    return df


def train(fileName = "trading_model", resume=False):
    df = load_data("data/historical_data/BTCUSD-5m-2025-12.csv", RSIPeriod)

    env = TradingEnv(df, startBalance=startMoney)

    policy = trainDQN(env, episodes = 2715, gamma=0.95, lr=1e-3, epsilon=0.1, stateSize = 12, actionSize = 3, resume=resume)

    model_path = os.path.join("AImodels", f"{fileName}.pth")
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved as {model_path}") 