import math
import requests
import os
from dotenv import load_dotenv
import time
import csv
import pandas as pd

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
notifSellThreshold = float(os.getenv("NOTIFSELLTHRESHOLD"))
notifBuyThreshold = float(os.getenv("NOTIFBUYTHRESHOLD"))

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


def rsi_sigmoid(x, offset):
    """
    Computes f(x) = 4/(1 + e^(-17*(x-3.3))) + 4/(1 + e^(-17*(x-6.7))) + 0.05*x - 4.25
    RSI: centered at 3, 7 is 3.3 and 6.7
    StochRSI: centered at 2, 8 is 2.3, 7.7
    """
    term1 = 4 / (1 + math.exp(-17 * (x - offset)))
    term2 = 4 / (1 + math.exp(-17 * (x - (10-offset))))
    linear = 0.05 * x
    return -(term1 + term2 + linear - 4.25)

def zVolume_sigmoid(x):
    """
    Computes f(x) = 2/(1 + e^(-15*(x + 0.6))) + 2/(1 + e^(-15*(x - 0.6))) + 0.05*x
    """
    term2 = 2 / (1 + math.exp(-15 * (x - 0.6)))
    linear = 0.2 * x
    return term2 + linear - 2


def addWeight(df):
    """
    Calculates each indicator and adds it to total score
    """
    df["rsi"] = RSI(df["close"], period=RSIPeriod)
    df["stoch_rsi"] = StochRSI(df["rsi"], period=RSIPeriod)
    # Add normalized volume (zVolume) — use base volume column
    df = zVolume(df) 

    score = 0

    score += rsi_sigmoid(df["rsi"].iloc[-1]/10, 3.3) #add RSI
    score += rsi_sigmoid(df["stoch_rsi"].iloc[-1]/10, 2.3) #add stochRSI
    if (score > 0):
        score += zVolume_sigmoid(df["zVolume"].iloc[-1]) 
    elif (score < 0):
        score -= zVolume_sigmoid(df["zVolume"].iloc[-1]) 

    # set score only for the last row
    df.loc[df.index[-1], "Score"] = score

    return df

def evaluation(df, buyThreshold, sellThreshold):
    """
    evaluates the score and take action against it
    """
    df = addWeight(df) # calculate the score
    price = df.iloc[-1]["close"]
    portfolio = load_portfolio() # loads balance

    if (df["Score"].iloc[-1] >= buyThreshold) and portfolio["position"] == 0: # If score is above threshold
        pnl = paperTrade("BUY", price, lotSize) #buy
        # Send to discord
        notify_discord(
            f"📈📈📈 **BUY SIGNAL**\n"
            f"Price: {df['close'].iloc[-1]:.2f}\n"
            f"RSI: {df['rsi'].iloc[-1]:.1f}\n"
            f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
            f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
            f"Score: {df['Score'].iloc[-1]:.1f}\n"
        )
        WriteOutTrades(df)
    elif (df["Score"].iloc[-1] <= sellThreshold) and portfolio["position"] >= 1: # If score is below seel threshold
        pnl = paperTrade("SELL", price, lotSize) #buy
        # send to discord
        notify_discord(
            f"📉📉📉 **Sell SIGNAL**\n"
            f"Price: {df['close'].iloc[-1]:.2f}\n"
            f"RSI: {df['rsi'].iloc[-1]:.1f}"
            f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
            f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
            f"Score: {df['Score'].iloc[-1]:.1f}\n"
            f"PnL: {pnl:.1f}\n"
        )
        WriteOutTrades(df)
    else: # Hold what you have 
        df["decision"] = "Hold"
    return df

def closeToNotifications(df):
    """
    notifications to discord if it is almost buy or almost sell
    """
    if (df["Score"].iloc[-1] >= notifBuyThreshold): # If score is above threshold
        notify_discord(
            f"📈 Notification BUY SIGNAL\n"
            f"Price: {df['close'].iloc[-1]:.2f}\n"
            f"RSI: {df['rsi'].iloc[-1]:.1f}\n"
            f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
            f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
            f"Score: {df['Score'].iloc[-1]:.1f}\n"
        )
    elif (df["Score"].iloc[-1] <= notifSellThreshold): # If score is below seel threshold
        notify_discord(
            f"📉 Notification Sell SIGNAL\n"
            f"Price: {df['close'].iloc[-1]:.2f}\n"
            f"RSI: {df['rsi'].iloc[-1]:.1f}"
            f"Stochastic RSI: {df['stoch_rsi'].iloc[-1]:.1f}\n"
            f"zVolume: {df['zVolume'].iloc[-1]:.1f}\n"
            f"Score: {df['Score'].iloc[-1]:.1f}\n"
        )
    else: # Hold what you have 
        df["decision"] = "Hold"
    return df

def ruleBasedRun():
    """
    Runs the rule based bot
    """
    notify_discord("✅ Bot is online!")

    # Load the json portfolio values
    portfolio = load_portfolio(startMoney)
    #Main Loop
    try: #try catch to notification when it crashes
        while True:
            sleep_until_next_candle()

            #Get Data from kraken
            df = GetCandle(pair, candle, session)
            df["timeStamp"] = WhatTime()
            df = evaluation(df, buyThreshold, sellThreshold)

            # Save results
            portfolio = load_portfolio()
            df["Balance"] = portfolio["balance"]
            df["Position"] = portfolio["position"]
            WriteOut(df)
    except Exception as e:
        notify_discord(
            f"🔴 **BOT CRASHED**\n"
            f"Error: `{type(e).__name__}`\n"
            f"Message: {e}"
        )
        raise
    finally:
        notify_discord("⚠️ Bot terminated (manual stop or crash)")