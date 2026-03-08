
# AI Trader

A Python-based cryptocurrency paper-trading project with **two strategy modes**:

- **Rule-based trading** driven by RSI, Stochastic RSI, and z-scored volume.
- **AI-based trading** powered by a Deep Q-Network (DQN) policy.

The bot fetches Kraken market candles, scores each new candle, simulates BUY/SELL actions against a local portfolio file, logs results to CSV, and can send Discord notifications.

---

## What this project does

- Runs on a fixed candle interval (configured in `.env`).
- Pulls recent OHLCV data from Kraken.
- Computes indicators (RSI, Stoch RSI, volume normalization, and Bollinger features for AI mode).
- Makes a trading decision (rule logic or trained AI model).
- Simulates trades in `portfolio.json` (paper trading only).
- Writes logs to `data/dataLogs/data_log.csv`.

> This repo is designed for experimentation and learning. It is **not financial advice** and should not be used as-is for live trading with real funds.

---

## Project structure

```
AI-Trader/
├── main.py                    # Entry point: choose AI mode or rule-based mode
├── baseDirectory/
│   ├── aiBased.py             # AI startup + live inference loop
│   └── ruleBased.py           # Rule-based scoring + trade loop
├── AI/
│   ├── brain.py               # DQN model + replay memory utilities
│   ├── train.py               # DQN training flow
│   └── TradingEnv.py          # RL environment for training
├── indicators/
│   ├── RSIIndicators.py       # RSI + Stochastic RSI
│   ├── volume.py              # z-score volume feature
│   └── bb.py                  # Bollinger-band features
├── data/
│   ├── getData.py             # Kraken candle fetcher
│   ├── paperTrade.py          # Portfolio simulation (buy/sell/balance)
│   ├── writeOut.py            # General logging
│   └── writeOutTrades.py      # Trade event logging
├── AImodels/                  # Saved `.pth` models
├── data/dataLogs/             # CSV logs
└── portfolio.json             # Paper portfolio state
```

---

## Requirements

- Python 3.10+
- `pip`
- Optional: CUDA-enabled PyTorch for faster training

Install dependencies:

```bash
pip install pandas numpy python-dotenv requests torch
```

---

## Environment configuration

Create a `.env` file in the project root.

```env
# Kraken config
KRAKEN_PUBLIC=https://api.kraken.com/0/public/OHLC
PAIR=BTCUSD
INTERVAL=300
CANDLE=5

# Indicator + sizing
RSIPERIOD=14
HOWMANYYOUWANT=0.01
INITIALPAPERMONEY=1000

# Rule-based thresholds
BUYTHRESHOLD=1.5
SELLTHRESHOLD=-1.5
NOTIFBUYTHRESHOLD=1.0
NOTIFSELLTHRESHOLD=-1.0
```

### Variable notes

- `CANDLE` is in minutes (e.g. `5`).
- `INTERVAL` is in seconds (e.g. `300`) and should match `CANDLE`.
- `HOWMANYYOUWANT` is position size per trade.
- `INITIALPAPERMONEY` seeds the simulated balance.

---

## How to run

From the repo root:

```bash
py main.py
```

You will be prompted to choose:

1. **AI Based Trading bot**
2. **Rule Based Trading Bot**

### AI mode

- Can train a new model or resume training.
- Loads a selected model from `AImodels/`.
- Uses the model to choose `HOLD`, `BUY`, or `SELL` each candle.

### Rule-based mode

- Computes a weighted score from RSI / Stoch RSI / zVolume.
- Buys above `BUYTHRESHOLD`, sells below `SELLTHRESHOLD`.

---

## Training notes (AI)

Training is launched from the AI startup prompt. Models are saved as `.pth` in `AImodels/`.

If you want to retrain from scratch:

1. Start `python main.py`
2. Choose option `1` (AI)
3. Enter `y` for training
4. Provide a model name
5. Choose resume `n`

---

## Logging and outputs

- Portfolio state persists in `portfolio.json`.
- Candle-by-candle logs are written to `data/dataLogs/data_log.csv`.
- Trade-specific events are written via `writeOutTrades.py`.
- Discord notifications are sent when webhook is configured in code.

---

## Known limitations

- This is a paper-trading simulator, not an exchange execution engine.
- No built-in risk management (max drawdown guard, stop-loss module, etc.) beyond current strategy logic.

---

## Roadmap ideas

- Add `.env.example`.
- Add backtesting CLI with metrics (Sharpe, drawdown, win rate).
- Add unit tests for indicators and paper-trade accounting.
- Move Discord webhook and model settings into environment config.
- Add Docker support for reproducible runs.

---
