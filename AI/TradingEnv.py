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

    # def step(self, action):
    #     row = self.df.iloc[self.t]
    #     price = row["close"]
    #     oldValue = self.balance + (price * self.holdingNum)

    #     # 1. HANDLE BUY (Action 1)
    #     if action == 1: 
    #         if self.holdingNum == 0 and self.balance >= price * self.lotSize:
    #             # Open the single allowed position
    #             self.balance -= price * self.lotSize
    #             self.holdingNum = self.lotSize
    #         elif self.holdingNum > 0:
    #             # Penalty for trying to buy while already holding
    #             return self._advance_with_penalty(-1.0)

    #     # 2. HANDLE SELL (Action 2)
    #     elif action == 2: 
    #         if self.holdingNum > 0:
    #             # Close the single allowed position
    #             self.balance += price * self.lotSize
    #             self.holdingNum = 0
    #         else:
    #             # Penalty for trying to sell while holding nothing
    #             return self._advance_with_penalty(-1.0)

    #     # 3. NORMAL FLOW (Advance time)
    #     self.t += 1
    #     self.done = self.t >= len(self.df) - 1
    #     if self.done:
    #         return self._getState(), 0.0, self.done
        
    #     newPrice = self.df.iloc[self.t]["close"]
    #     newValue = self.balance + (newPrice * self.holdingNum)
        
    #     # Reward Calculation (0.1% move = 1.0 Reward)
    #     reward = ((newValue - oldValue) / oldValue) * 1000
        
    #     # Smarter Lazy Logic (Pat on the back for staying out during a crash)
    #     if self.holdingNum == 0 and action == 0:
    #         price_change = ((newPrice - price) / price) * 1000
    #         # If price went up and we missed it: Penalty
    #         # If price went down and we avoided it: Small Reward
    #         reward = -price_change * 0.1 if price_change > 0 else abs(price_change) * 0.05
                
    #     return self._getState(), reward, self.done


    # def step(self, action):
    #     row = self.df.iloc[self.t]
    #     price = row["close"]
    #     oldValue = self.balance + (price * self.holdingNum)

    #     # 1. HANDLE ILLEGAL MOVES (Increased Penalty to -5.0)
    #     if action == 1 and self.holdingNum > 0: # Buy while already holding
    #         return self._advance_with_penalty(-5.0)
        
    #     if action == 2 and self.holdingNum == 0: # Sell while holding nothing
    #         return self._advance_with_penalty(-5.0)

    #     # 2. EXECUTE VALID ACTIONS
    #     if action == 1 and self.holdingNum == 0 and self.balance >= price * self.lotSize:
    #         self.balance -= price * self.lotSize
    #         self.holdingNum = self.lotSize
    #         # Small "Action" penalty to prevent jittering
    #         action_penalty = -0.1 
    #     elif action == 2 and self.holdingNum > 0:
    #         self.balance += price * self.lotSize
    #         self.holdingNum = 0
    #         action_penalty = -0.1
    #     else:
    #         action_penalty = 0

    #     # 3. ADVANCE TIME
    #     self.t += 1
    #     self.done = self.t >= len(self.df) - 1
    #     if self.done:
    #         return self._getState(), 0.0, self.done
        
    #     newPrice = self.df.iloc[self.t]["close"]
    #     newValue = self.balance + (newPrice * self.holdingNum)
        
    #     # 4. REWARD CALCULATION
    #     # Base reward: Percentage change in portfolio
    #     reward = ((newValue - oldValue) / oldValue) * 1000
        
    #     # Add the action penalty (stops the bot from clicking buttons for no reason)
    #     reward += action_penalty

    #     # 5. SMARTER LAZY LOGIC (Reduced multipliers)
    #     if self.holdingNum == 0 and action == 0:
    #         price_change = ((newPrice - price) / price) * 1000
    #         # Small pat on the back for avoiding drops, small sting for missing gains
    #         reward = -price_change * 0.05 if price_change > 0 else abs(price_change) * 0.02
                
    #     return self._getState(), reward, self.done


    # def step(self, action):
    #     row = self.df.iloc[self.t]
    #     price = row["close"]

    #     old_value = self.balance + self.holdingNum * price

    #     executed_buys = 0
    #     executed_sells = 0


    #     # ============================
    #     # 1. HANDLE ILLEGAL ACTIONS
    #     # ============================
    #     if action == 1 and self.holdingNum > 0:
    #         # Buy while already holding → penalty, NO time advance
    #         return self._getState(), -5.0, self.done

    #     if action == 2 and self.holdingNum == 0:
    #         # Sell while holding nothing → penalty, NO time advance
    #         return self._getState(), -5.0, self.done
        

    #     # ============================
    #     # 2. EXECUTE VALID ACTION
    #     # ============================
    #     action_penalty = 0.0

    #     if action == 1:  # BUY
    #         cost = price * self.lotSize
    #         fee_cost = cost * self.fee
    #         total_cost = cost + fee_cost

    #         if self.balance >= total_cost:
    #             self.balance -= total_cost
    #             self.holdingNum = self.lotSize
    #             action_penalty = -0.05  # optional small friction

    #     elif action == 2:  # SELL
    #         revenue = price * self.lotSize
    #         fee_cost = revenue * self.fee
    #         net_revenue = revenue - fee_cost

    #         self.balance += net_revenue
    #         self.holdingNum = 0
    #         action_penalty = -0.05

    #     # action == 0 → HOLD (no changes)

    #     # ============================
    #     # 3. ADVANCE TIME (ONLY HERE)
    #     # ============================
    #     self.t += 1
    #     self.done = self.t >= len(self.df) - 1

    #     if self.done:
    #         return self._getState(), 0.0, self.done

    #     # ============================
    #     # 4. REWARD CALCULATION
    #     # ============================
    #     new_price = self.df.iloc[self.t]["close"]
    #     new_value = self.balance + self.holdingNum * new_price

    #     # Portfolio percentage change
    #     reward = ((new_value - old_value) / old_value) * 1000
    #     reward += action_penalty

    #     # ============================
    #     # 5. SMALL HOLD BONUS (OPTIONAL)
    #     # ============================
    #     if action == 0 and self.holdingNum == 0:
    #         price_change = ((new_price - price) / price) * 1000
    #         reward += -price_change * 0.02 if price_change > 0 else abs(price_change) * 0.01

    #     return self._getState(), reward, self.done

    def step(self, action):
            row = self.df.iloc[self.t]
            price = row["close"]
            old_value = self.balance + self.holdingNum * price

            reward = 0.0
            
            # 1. ACTION LOGIC WITH TRAPPING
            if action == 1: # BUY
                if self.holdingNum == 0:
                    cost = price * self.lotSize
                    total_cost = cost * (1 + self.fee)
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
