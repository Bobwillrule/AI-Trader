import math
import requests
import os
from dotenv import load_dotenv
import time
import csv
import pandas as pd

from baseDirectory.ruleBased import addWeight
from indicators.SupportResistance import detect_break_retest, get_nearest_levels, get_rolling_levels
from data.writeOutTrades import WriteOutTrades
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
buyThreshold = float(os.getenv("BUYTHRESHOLD"))
sellThreshold = float(os.getenv("SELLTHRESHOLD"))

CANDLE_SECONDS = int(candle) * 60


def load_rule_data(filename):
    df = pd.read_csv(
        filename,
        header=None,
        names=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
    )

    df[["open","high","low","close","volume"]] = df[[
        "open","high","low","close","volume"
    ]].astype(float)

    df["rsi"] = RSI(df["close"], period=RSIPeriod)
    df["stoch_rsi"] = StochRSI(df["rsi"], period=RSIPeriod)
    df = zVolume(df)

    return df.dropna().reset_index(drop=True)

def paperTrade_backtest(action, price, lotSize, portfolio):
    if action == "BUY" and portfolio["position"] == 0:
        portfolio["position"] = lotSize
        portfolio["entry_price"] = price
        portfolio["balance"] -= price * lotSize

    elif action == "SELL" and portfolio["position"] > 0:
        pnl = (price - portfolio["entry_price"]) * portfolio["position"]
        portfolio["balance"] += price * portfolio["position"]
        portfolio["position"] = 0
        portfolio["entry_price"] = 0
        return pnl

    return 0

def evaluation_backtest(df, buyThreshold, sellThreshold, portfolio):
    df = addWeight(df)
    price = df.iloc[-1]["close"]
    score = df["Score"].iloc[-1]

    if score >= buyThreshold and portfolio["position"] == 0:
        paperTrade_backtest("BUY", price, lotSize, portfolio)
        df.loc[df.index[-1], "decision"] = "BUY"

    elif score <= sellThreshold and portfolio["position"] > 0:
        pnl = paperTrade_backtest("SELL", price, lotSize, portfolio)
        df.loc[df.index[-1], "decision"] = "SELL"
        df.loc[df.index[-1], "PnL"] = pnl

    else:
        df.loc[df.index[-1], "decision"] = "HOLD"

    df.loc[df.index[-1], "Balance"] = portfolio["balance"]
    df.loc[df.index[-1], "Position"] = portfolio["position"]

    return df

def backtest_rule_bot(csv_file):
    df_all = load_rule_data(csv_file)

    portfolio = {
        "balance": startMoney,
        "position": 0,
        "entry_price": 0
    }

    results = []
    warmup = RSIPeriod * 3

    for i in range(warmup, len(df_all)):
        df_slice = df_all.iloc[:i+1].copy()
        df_slice = evaluation_backtest(
            df_slice,
            buyThreshold,
            sellThreshold,
            portfolio
        )
        results.append(df_slice.iloc[-1])

    result_df = pd.DataFrame(results)
    result_df.to_csv("rule_backtest_results.csv", index=False)

    last_price = result_df.iloc[-1]["close"]

    balanceTotal = (
        portfolio["balance"] +
        portfolio["position"] * last_price
    )

    print(
        "Final balance (cash):", portfolio["balance"],
        "\nFinal position:", portfolio["position"],
        "\nFinal equity:", balanceTotal
    )

    return result_df


