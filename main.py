import pandas as pd

from backtesting.backTest import backtest_rule_bot

def choose():
    """starts the program, asks if training is needed or not"""
    loop = True
    while loop:
        baseOption = input("Please select a mode (1/2):\n 1. AI Base Trading bot\n 2. Rule Based Trading Bot\n>").lower()
        if baseOption == "1":
            print("AI Based Trading bot selected")
            loop = False
            from baseDirectory.aiBased import AIStartUp, AIrun
            AIStartUp()
            AIrun()
        elif baseOption == "2":
            print("Rule Base Trading bot selected.")
            from baseDirectory.ruleBased import ruleBasedRun
            loop = False
            ruleBasedRun()
        elif baseOption == "3":
            print("Back test Rule Trading bot selected.")
            backtest_rule_bot("data/historical_data/BTCUSD-5m-2025-12.csv")
            loop = False
        else:
            print(f"Invalid input: '{baseOption}'. Please select 1 or 2")

if __name__ == "__main__":
    choose()








