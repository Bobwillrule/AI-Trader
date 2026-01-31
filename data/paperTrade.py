import json
import os

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio(start_balance=1000):
    """loads the portfolio if it exists, if not, create a new one"""
    if not os.path.exists(PORTFOLIO_FILE):
        portfolio = {
            "balance": start_balance,
            "position": 0,
            "num_trades": 0
        }
        save_portfolio(portfolio)
        return portfolio

    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def save_portfolio(portfolio):
    """saves the json portfolio for loading next time"""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def paperTrade(df, buy, lotSize):
    """Practice trading updating only the last row (no history added)"""

    if buy:
        # Calculate cost of purchase
        price = lotSize * df["close"].iloc[-1]

        # Check that we have enough money to buy
        if price <= df["Balance"].iloc[-1]:
            # Deduct cost from balance
            df.loc[df.index[-1], "Balance"] -= price
            # Add coins to our holdings
            df.loc[df.index[-1], "Amount"]  += lotSize

    else:
        # Check that we have enough coins to sell
        if df["Amount"].iloc[-1] >= lotSize:
            # Calculate proceeds of sale
            price = lotSize * df["close"].iloc[-1]
            # Add money to balance
            df.loc[df.index[-1], "Balance"] += price
            # Subtract coins from holdings
            df.loc[df.index[-1], "Amount"]  -= lotSize
    return df