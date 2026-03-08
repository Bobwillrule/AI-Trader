import pandas as pd

def choose():
    """starts the program, asks if training is needed or not"""
    loop = True
    while loop:
        #get user input
        baseOption = input("Please select a mode (1/2):\n 1. AI Base Trading bot\n 2. Rule Based Trading Bot\n>").lower()
        if baseOption == "1": #AI option
            print("AI Based Trading bot selected")
            loop = False
            from baseDirectory.aiBased import AIStartUp, AIrun
            AIStartUp()
            AIrun()
        elif baseOption == "2": # Rule based option
            print("Rule Base Trading bot selected.")
            from baseDirectory.ruleBased import ruleBasedRun
            loop = False
            ruleBasedRun()
        elif baseOption == "3":
            print("Back test Rule Trading bot selected.")
            loop = False
        else:
            print(f"Invalid input: '{baseOption}'. Please select 1 or 2")

if __name__ == "__main__":
    choose()








