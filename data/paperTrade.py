import json
import os

PORTFOLIO_FILE = "portfolio.json"
TAKER_FEE = 0.004   # 0.40%

def load_portfolio(start_balance=1000):
    """loads the portfolio if it exists, if not, create a new one"""
    if not os.path.exists(PORTFOLIO_FILE):
        portfolio = {
            "balance": start_balance,
            "position": 0,
            "num_trades": 0,
            "realized_pnl": 0.0
        }
        save_portfolio(portfolio)
        return portfolio

    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def save_portfolio(portfolio):
    """saves the json portfolio for loading next time"""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def paperTrade(action, price, quantity):
    portfolio = load_portfolio()

    if action == "BUY" and portfolio["position"] == 0:
        notional = price * quantity
        fee = notional * TAKER_FEE
        total_cost = notional + fee
        trade_pnl = 0

        if portfolio["balance"] >= total_cost:
            portfolio["balance"] -= total_cost
            portfolio["position"] = 1
            portfolio["entry_price"] = price
            portfolio["quantity"] = quantity
            portfolio["last_fee"] = fee

    elif action == "SELL" and portfolio["position"] > 0:
        notional = price * portfolio["quantity"]
        fee = notional * TAKER_FEE
        proceeds = notional - fee

        # Realized PnL (fees included)
        trade_pnl = (
            (price - portfolio["entry_price"])
            * portfolio["quantity"]
            - fee
            - portfolio.get("entry_fee", 0)
        )

        portfolio["balance"] += proceeds
        portfolio["position"] = 0
        portfolio["realized_pnl"] += trade_pnl
        portfolio["last_fee"] = fee

    save_portfolio(portfolio)
    return trade_pnl
