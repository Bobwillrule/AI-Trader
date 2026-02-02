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


def paperTrade(action, price, lotSize):
    portfolio = load_portfolio()

    if action == "BUY":
        cost = lotSize * price
        if portfolio["balance"] >= cost:
            portfolio["balance"] -= cost
            portfolio["position"] += lotSize
            portfolio["num_trades"] += 1

    elif action == "SELL":
        if portfolio["position"] >= lotSize:
            portfolio["balance"] += lotSize * price
            portfolio["position"] -= lotSize
            portfolio["num_trades"] += 1

    save_portfolio(portfolio)
