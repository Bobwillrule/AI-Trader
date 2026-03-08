def add_moving_averages(df):
    """
    Creates a moving average of 50 and 200
    """
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    return df

def add_ema(df, span=20):
    """
    creates an exponential moving average
    """
    df[f"EMA{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df

