import pandas as pd
import numpy as np

from data.getData import GetCandle

def get_rolling_levels(df, lookback=100, window=5, proximity_pct=0.0015, min_touches=2):
    """
    Identifies volume-weighted S/R levels using a rolling lookback window.
    
    df: Dataframe with ['High', 'Low', 'Close', 'Volume']
    lookback: Number of candles to look back from the current point
    proximity_pct: How tight the cluster must be (0.0015 = 0.15%)
    """
    
    # 1. Slice the dataframe for the rolling window
    # We take the last 'lookback' candles
    recent_df = df.tail(lookback).copy()
    
    # 2. Find Pivot Highs and Lows
    recent_df['is_high'] = recent_df['High'] == recent_df['High'].rolling(2*window+1, center=True).max()
    recent_df['is_low'] = recent_df['Low'] == recent_df['Low'].rolling(2*window+1, center=True).min()
    
    # Store pivots with their volume for weighting
    pivots = []
    for idx, row in recent_df.iterrows():
        if row['is_high']:
            pivots.append({'price': row['High'], 'vol': row['Volume']})
        if row['is_low']:
            pivots.append({'price': row['Low'], 'vol': row['Volume']})
            
    # Sort pivots by price for easier clustering
    pivots = sorted(pivots, key=lambda x: x['price'])
    
    confirmed_levels = []
    used_indices = set()

    # 3. Clustering Logic
    for i in range(len(pivots)):
        if i in used_indices:
            continue
            
        current_pivot = pivots[i]
        margin = current_pivot['price'] * proximity_pct
        
        # Look for neighbors
        cluster = [current_pivot]
        for j in range(i + 1, len(pivots)):
            if abs(pivots[j]['price'] - current_pivot['price']) <= margin:
                cluster.append(pivots[j])
                used_indices.add(j)
        
        # 4. Validating and Weighting
        if len(cluster) >= min_touches:
            # Volume-Weighted Average Price for the level
            total_vol = sum(c['vol'] for c in cluster)
            weighted_price = sum(c['price'] * c['vol'] for c in cluster) / total_vol
            
            confirmed_levels.append({
                'price': round(weighted_price, 2),
                'touches': len(cluster),
                'volume_score': total_vol,
                'min_edge': min(c['price'] for c in cluster), # Bottom of zone
                'max_edge': max(c['price'] for c in cluster)  # Top of zone
            })
            
    return pd.DataFrame(confirmed_levels).sort_values(by='volume_score', ascending=False)

def get_structure_levels_1h(pair, session):
    df_1h = GetCandle(pair, "60", session)  # 60 min candles

    levels = get_rolling_levels(
        df_1h.rename(columns={
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }),
        lookback=120,
        window=5,
        proximity_pct=0.002,
        min_touches=2
    )

    return levels

def detect_break_retest(df_5m, level_price, direction, lookback=6):
    """
    direction:
        "bull" → break above resistance, retest as support
        "bear" → break below support, retest as resistance
    """
    recent = df_5m.tail(lookback)

    if direction == "bull":
        broke = (recent["close"] > level_price).any()
        retest = (
            recent["low"].min() <= level_price * 1.001 and
            recent.iloc[-1]["close"] > level_price
        )
        return broke and retest

    if direction == "bear":
        broke = (recent["close"] < level_price).any()
        retest = (
            recent["high"].max() >= level_price * 0.999 and
            recent.iloc[-1]["close"] < level_price
        )
        return broke and retest

    return False


def get_nearest_levels(levels_df, price, tolerance_pct=0.002):
    """
    Returns nearest support and resistance relative to price
    """
    if levels_df.empty:
        return None, None

    tolerance = price * tolerance_pct

    supports = levels_df[levels_df["price"] <= price + tolerance]
    resistances = levels_df[levels_df["price"] >= price - tolerance]

    nearest_support = supports.iloc[-1] if not supports.empty else None
    nearest_resistance = resistances.iloc[0] if not resistances.empty else None

    return nearest_support, nearest_resistance


