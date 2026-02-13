
import math
import requests
import os
from dotenv import load_dotenv
import time
import csv
import pandas as pd
import torch

from AI.brain import policyNetwork
from AI.train import train
from indicators.bb import Bollinger
from indicators.RSIIndicators import RSI, StochRSI
from data.writeOut import WriteOut
from data.paperTrade import load_portfolio, paperTrade, save_portfolio
from indicators.volume import zVolume
from data.time import WhatTime, sleep_until_next_candle
from data.getData import GetCandle, PublicInfo


load_dotenv() 
pair = os.getenv("PAIR")
interval = int(os.getenv("INTERVAL")) # in seconds
candle = os.getenv("CANDLE") # in minutes
RSIPeriod = int(os.getenv("RSIPERIOD"))
startMoney = int(os.getenv("INITIALPAPERMONEY"))
lotSize = float(os.getenv("HOWMANYYOUWANT"))
session =requests.Session() # start the session
CANDLE_SECONDS = int(candle) * 60


DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1467265319876690010/jLkgDBcsRVklEb5zgGdnNWkXJSTvA8Fgr9bsU91NYkvPGpu8ks39mx77NNQmmCERKN_M"

def notify_discord(message):
    try:
        payload = {
            "content": message
        }
        requests.post(DISCORD_WEBHOOK, json=payload)
    except Exception:
        pass



def normalizeOHLC(df):
    #  convert to numeric first
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    prev_close = df["close"].shift(1)

    df['nOpen']  = ((df['open'] - prev_close) / prev_close).clip(-1, 1)
    df['nHigh']  = ((df['high'] - prev_close) / prev_close).clip(-1, 1)
    df['nLow']   = ((df['low'] - prev_close) / prev_close).clip(-1, 1)
    df['nClose'] = ((df['close'] - prev_close) / prev_close).clip(-1, 1)

    df["nRSI"] = df["rsi"] / 100
    df["nStochRSI"] = df["stoch_rsi"] / 100
    df["nZVolume"] = df["zVolume"] / 3

    # remove NaNs from shift / coercion
    df = df.dropna().reset_index(drop=True)

    return df



# Order: [RSI, stochRSI, z_volume, holdingNum, balance]
def extract_state(df, holdingNum=0, balance=1000, lotSize=1, startBalance=1000):
    last = df.iloc[-1]

    return torch.tensor([
        last["nOpen"],
        last["nHigh"],
        last["nLow"],
        last["nClose"],
        last["nRSI"],
        last["nStochRSI"],
        last["nZVolume"],
        last["bb_pos"],
        last["bb_width"],
        last["bb_dist_mid"],
        holdingNum / lotSize,        # normalized position (0–1)
        balance / startBalance       # normalized balance (0–1)
    ], dtype=torch.float32)



def select_model(model_dir="AImodels"):
    models = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

    if not models:
        print("No models found, please train one prior to running one.")
        return None

    print("\nAvailable models:")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    while True:
        choice = input("Select a model by number: ").strip()
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(models):
                return os.path.join(model_dir, models[choice - 1])
        print("Invalid selection. Try again.")


def AIStartUp():
    """starts the program, asks if training is needed or not"""
    loop = True
    loop2 = True
    while loop:
        trainOption = input("Do you want to train the model? (y/n): ").lower()
        if trainOption == "":
            print("No input detected, skipping training.")
            loop = False

        # If user wants to train
        elif trainOption == "y":
            fileName = input("Please Input Your Model Name: ").lower()
            
            # Ask if we want to resume or not
            while loop2:
                resume = input("Would you like to resume a previous session? (y/n): ").lower()
                if resume == "":
                    train(fileName, False)
                    print("New training.")
                    loop2 = False
                elif resume == "y":
                    train(fileName, True)
                    print("Continuing training.")
                    loop2 = False
                elif resume == "n":
                    train(fileName, False)
                    print("New training.")
                    loop2 = False
                else:
                    print(f"Invalid input: '{trainOption}'.")

            train(fileName)
            loop = False
        elif trainOption == "n":
            print("Skipping training.")
            loop = False
        else:
            print(f"Invalid input: '{trainOption}'.")
    
    if (trainOption == "" or trainOption == "n"):
        AIrun(select_model())
    else:
        model_path = os.path.join("AImodels", f"{fileName}.pth")
        AIrun(model_path)
        

def AIrun(model):
    policy = policyNetwork(stateSize=12, actionSize=3)
    policy.load_state_dict(torch.load(model))
    policy.eval()

    portfolio = load_portfolio(startMoney)
    balance = portfolio["balance"]
    holdingNum = portfolio["position"]

    while True:
        sleep_until_next_candle()

        df = GetCandle(pair, candle, session)
        df["timeStamp"] = WhatTime()

        df["rsi"] = RSI(df["close"], RSIPeriod)
        df["stoch_rsi"] = StochRSI(df["rsi"])
        df = zVolume(df)
        df = Bollinger(df)
        df = normalizeOHLC(df)
        df.loc[df.index[-1], "Score"] = 0

        state = extract_state(df, holdingNum, balance)

        with torch.no_grad():
            qvals = policy(state)
            action = torch.argmax(qvals).item()

        price = df.iloc[-1]["close"]

        # === EXECUTE TRADE USING NEW PAPERTRADE ===
        trade_pnl = 0
        if action == 1:
            trade_pnl = paperTrade("BUY", price, lotSize)
            notify_discord(
                f"📈📈📈 **BUY SIGNAL**\n"
                f"Price: {df['close'].iloc[-1]:.2f}\n"
                f"RSI: {df['rsi'].iloc[-1]:.1f}\n"
                f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
                f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
                f"Score: {df['Score'].iloc[-1]:.1f}\n"
            )
        elif action == 2:
            trade_pnl = paperTrade("SELL", price, lotSize)
            notify_discord(
                f"📉📉📉 **Sell SIGNAL**\n"
                f"Price: {df['close'].iloc[-1]:.2f}\n"
                f"RSI: {df['rsi'].iloc[-1]:.1f}"
                f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
                f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
                f"Score: {df['Score'].iloc[-1]:.1f}\n"
                f"PnL: {trade_pnl:.1f}\n"
            )

        # === RELOAD UPDATED PORTFOLIO ===
        portfolio = load_portfolio(startMoney)
        balance = portfolio["balance"]
        holdingNum = portfolio["position"]

        # Save results
        df["Balance"] = balance
        WriteOut(df)

        save_portfolio(portfolio)








