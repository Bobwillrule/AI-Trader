import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv() 

url_public = os.getenv("KRAKEN_PUBLIC")


def PublicInfo(linkEnd, pair, candle, session):
    """Gets the public info of the kraken API. Link end appends to the end of the link
    to send the right request and the pair is the pair of coins we wish to get"""
    url = f"{url_public}/{linkEnd}"
    response = session.get(url, params={"pair": pair, "interval": candle}, timeout=10) # get the session and the API request
    response.raise_for_status() # Check for status, 200 is good
    data = response.json() # make the json into a map

    candles = data["result"].get(pair)
    if candles is None:
        raise ValueError(f"No OHLC data found for pair '{pair}'") # if get returns none
    return candles

def GetCandle(pair, candle, session):
    """Gets the latest price of the pair"""
    data = PublicInfo("OHLC", pair, candle, session) # Get the candle info 
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"]) # Put in dataframe
    df["close"] = df["close"].astype(float)
    return df