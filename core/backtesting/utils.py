import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """
    Calculates a set of common technical indicators for a given DataFrame.
    Assumes the DataFrame has 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    """
    df_copy = df.copy()

    # Simple Moving Average (SMA)
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    df_copy['EMA_10'] = df_copy['Close'].ewm(span=10, adjust=False).mean()
    df_copy['EMA_20'] = df_copy['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']

    # Bollinger Bands
    df_copy['Bollinger_Bands_Middle'] = df_copy['Close'].rolling(window=20).mean()
    std_dev = df_copy['Close'].rolling(window=20).std()
    df_copy['Bollinger_Bands_Upper'] = df_copy['Bollinger_Bands_Middle'] + (std_dev * 2)
    df_copy['Bollinger_Bands_Lower'] = df_copy['Bollinger_Bands_Middle'] - (std_dev * 2)

    # Average True Range (ATR)
    high_low = df_copy['High'] - df_copy['Low']
    high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
    low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_copy['ATR'] = tr.ewm(span=14, adjust=False).mean()

    # Average Directional Index (ADX) - Simplified
    # This is a complex indicator, a full implementation would be more involved.
    # For simplicity, we'll just add a placeholder or a very basic version.
    # A proper ADX calculation involves +DM, -DM, TR, smoothed versions, and then DX.
    # Placeholder:
    df_copy['ADX'] = np.nan 

    # Commodity Channel Index (CCI)
    TP = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
    SMA_TP = TP.rolling(window=20).mean()
    MD = np.abs(TP - SMA_TP).rolling(window=20).mean()
    df_copy['CCI'] = (TP - SMA_TP) / (0.015 * MD)

    # Stochastic Oscillator
    lowest_low = df_copy['Low'].rolling(window=14).min()
    highest_high = df_copy['High'].rolling(window=14).max()
    df_copy['Stochastic_K'] = ((df_copy['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df_copy['Stochastic_D'] = df_copy['Stochastic_K'].rolling(window=3).mean()

    # Volume Change
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()

    # Volume Weighted Average Price (VWAP) - Requires intraday data for true VWAP
    # For daily data, a simplified version can be used, or it can be excluded.
    # Placeholder:
    df_copy['VWAP'] = np.nan 

    # On-Balance Volume (OBV)
    df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mfv = ((df_copy['Close'] - df_copy['Low']) - (df_copy['High'] - df_copy['Close'])) / (df_copy['High'] - df_copy['Low']) * df_copy['Volume']
    df_copy['CMF'] = mfv.rolling(window=20).sum() / df_copy['Volume'].rolling(window=20).sum()

    # Force Index (FI)
    df_copy['FI'] = df_copy['Close'].diff(1) * df_copy['Volume']

    # Ease of Movement (EOM)
    dm = ((df_copy['High'] + df_copy['Low']) / 2) - ((df_copy['High'].shift(1) + df_copy['Low'].shift(1)) / 2)
    vr = (df_copy['Volume'] / (df_copy['High'] - df_copy['Low']))
    df_copy['EOM'] = dm / vr

    # Drop rows with NaN values created by rolling windows
    df_copy.dropna(inplace=True)
    
    return df_copy