
def RSI(close, period=14):
    """
    Calculates the RSI indicator
    """
    delta = close.diff() # Calculate the difference
    gain = delta.clip(lower=0) #Get pos changes
    loss = -delta.clip(upper=0) # get neg change

    #Avg for pos and neg in period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    #Divide them
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def StochRSI(rsi, period=14):
    """
    Calculates the stochastic version of the RSI indicator
    """
    #Find the max a min rsi in the period
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()

    #normalize within min max range
    return ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
