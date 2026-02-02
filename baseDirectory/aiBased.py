
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


# Order: [RSI, stochRSI, z_volume, holdingNum, balance]
def extract_state(df, holdingNum=0, balance=1000):
    last = df.iloc[-1]
    return torch.tensor([
        last["rsi"],         # or RSI column from your indicators
        last["stoch_rsi"],   # StochRSI column
        last["zVolume"],      # normalized volume
        holdingNum,          # current holding number
        balance              # current balance
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
    # Load policy
    policy = policyNetwork(stateSize=5, actionSize=3)
    policy.load_state_dict(torch.load(model))
    policy.eval()

    # Load the json portfolio values
    portfolio = load_portfolio(startMoney)
    balance = portfolio["balance"]
    holdingNum = portfolio["position"]

    #Main Loop
    while True:
        sleep_until_next_candle()

        #Get Data from kraken
        df = GetCandle(pair, candle, session)
        df["timeStamp"] = WhatTime()

        # compute indicators (assuming your RSIIndicators module does this)
        df["rsi"] = RSI(df["close"], RSIPeriod)
        df["stoch_rsi"] = StochRSI(df["rsi"])
        df = zVolume(df)

        # Extract state
        state = extract_state(df, holdingNum, balance)

        # Decide action
        with torch.no_grad():
            qvals = policy(state)
            action = torch.argmax(qvals).item()

        # Execute trade
        price = df.iloc[-1]["close"]
        if action == 1:
            paperTrade.buy(price)
            holdingNum += lotSize
            balance -= price * lotSize
            portfolio["num_trades"] += 1
            
        elif action == 2:
            paperTrade.sell(price)
            holdingNum -= lotSize
            balance += price * lotSize
            portfolio["num_trades"] += 1
            

        # Save results
        df["Balance"] = balance
        WriteOut(df)

        portfolio["balance"] = balance
        portfolio["position"] = holdingNum
        save_portfolio(portfolio)








