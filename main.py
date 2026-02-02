import pandas as pd

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
        else:
            print(f"Invalid input: '{baseOption}'. Please select 1 or 2")

if __name__ == "__main__":
    choose()








