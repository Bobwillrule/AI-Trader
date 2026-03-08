# Crypto Trading Bot with DQN Reinforcement Learning

A **reinforcement learning-based trading bot** that simulates cryptocurrency trading using historical market data. It uses a **Deep Q-Network (DQN)** to learn optimal buy, sell, or hold strategies based on technical indicators.

---

## Features
- **DQN-based AI**: Learns trading strategies using historical data.
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - Stochastic RSI
  - Z-scored volume (`zVolume`)
- **Paper Trading Simulation**: Simulate trades without risking real money.
- **Customizable Parameters**: Pair, interval, lot size, initial balance, RSI period, buy/sell thresholds.
- **GPU Support**: Optional CUDA acceleration for faster training.

---

## Quickstart

- **Required Installations**: python, .env, Pandas, PyTorch 2.9.0 with cuda

| Step | Command | Description |
|------|---------|-------------|
| 1 | `git clone https://github.com/yourusername/crypto-dqn-bot.git` | Clone the repository |
| 2 | `py main.py` | Launch the program from main |
| 3 | `y or n` | Select 'y' or 'n' to train a model or use an existing trained model|

## Directory Organization
```
bot/
│
├─ AI/
│   ├─ brain.py        # DQN network and training functions
│   ├─ train.py        # Training pipeline
│   └─ TradingEnv.py   # Trading environment simulation
│
├─ data/
│   ├─ writeOut.py     # Save trade logs
│   └─ time.py         # Timestamp helper
│
├─ indicators/
│   ├─ RSIIndicators.py # RSI & Stochastic RSI calculations
│   └─ volume.py        # zVolume calculation
│
├─ main.py             # Main entry point
└─ BTTUSD_5.csv        # Historical market data


