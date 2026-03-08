import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv() 

candle = os.getenv("CANDLE") # in minutes
CANDLE_SECONDS = int(candle) * 60

def WhatTime():
    """returns the current date and time"""
    return f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}/" \
           f"{datetime.now(timezone(timedelta(hours=-7))).strftime('%m-%d %H:%M:%S')}"

def sleep_until_next_candle():
    """
    Lines up the time the bot runs with the candle.
    Ie. if 5 minute candles we run at 5:05 then 10:00 then 10:05 and not in between
    """
    now = time.time()
    sleep_time = CANDLE_SECONDS - (now % CANDLE_SECONDS) #Calculate the time dfference
    time.sleep(sleep_time + 0.1)  # small buffer