import math
import numpy as np

class TradingEnv:
    """Simulates trading of the bot using the most recent rows of data."""

    def __init__(self, df, lotSize=0.001, startBalance=1000, window=6000, fee = 0.004):
        # Keep only the last 'window' rows
        self.df = df.tail(window).reset_index(drop=True)
        self.lotSize = lotSize
        self.fee = fee
        self.startBalance = startBalance
        self.reset()

    def reset(self):
        """Resets portfolio for a new episode"""
        self.t = 0
        self.balance = self.startBalance
        self.holdingNum = 0
        self.done = False
        return self._getState()

    def _getState(self):
        """Returns current state for AI"""
        row = self.df.iloc[self.t]
        return np.array([
            row["nOpen"],                   # open price
            row["nHigh"],                   # high price
            row["nLow"],                    # low price
            row["nClose"],                  # close price
            row["nRSI"],                    # 0-100
            row["nStochRSI"],              # 0-100
            row["nZVolume"],                # roughly ~[-3,3]
            row["bb_pos"],        # 0-1
            row["bb_width"],      # volatility
            row["bb_dist_mid"],   # mean reversion

            self.holdingNum / self.lotSize, # fraction of max lot held (0-1)
            self.balance / self.startBalance # normalized balance (0-1)
        ], dtype=np.float32)


    def step(self, action):
            row = self.df.iloc[self.t]
            price = row["close"]
            old_value = self.balance + self.holdingNum * price # Calculate old position first for later lose/gain

            reward = 0.0
            
            # 1. ACTION LOGIC WITH TRAPPING
            if action == 1: # BUY
                if self.holdingNum == 0:
                    cost = price * self.lotSize
                    total_cost = cost * (1 + self.fee) # cost + fee
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.holdingNum = self.lotSize
                    else:
                        reward = -0.0002 # Penalty for being broke
                else:
                    reward = -0.0002 # Penalty for double buying

            elif action == 2: # SELL
                if self.holdingNum > 0:
                    revenue = price * self.lotSize
                    net_revenue = revenue * (1 - self.fee)
                    self.balance += net_revenue
                    self.holdingNum = 0
                else:
                    reward = -0.0002 # Penalty for selling air

            # 2. ADVANCE TIME
            self.t += 1
            self.done = self.t >= len(self.df) - 1
            if self.done:
                return self._getState(), reward, self.done

            # 3. CALCULATE PNL REWARD
            new_price = self.df.iloc[self.t]["close"]
            new_value = self.balance + self.holdingNum * new_price
            
            # Log return of total portfolio value
            pnl_reward = math.log(new_value / old_value)
            reward += pnl_reward



            return self._getState(), reward, self.done


    def _advance_with_penalty(self, penalty_value):
        """Helper to penalize illegal moves and move time forward"""
        self.t += 1
        self.done = self.t >= len(self.df) - 1
        return self._getState(), penalty_value, self.done
