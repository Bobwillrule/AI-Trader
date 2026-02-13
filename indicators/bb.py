def Bollinger(df, period=20, std_mult=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()

    df["bb_mid"] = sma
    df["bb_upper"] = sma + std_mult * std
    df["bb_lower"] = sma - std_mult * std

    # === RL-friendly normalized features ===
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Position inside band (0-1)
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Distance from middle band
    df["bb_dist_mid"] = (df["close"] - df["bb_mid"]) / df["bb_mid"]

    df = df.dropna().reset_index(drop=True)
    return df
