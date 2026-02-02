def add_moving_averages(df):
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    return df

def add_ema(df, span=20):
    df[f"EMA{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df

