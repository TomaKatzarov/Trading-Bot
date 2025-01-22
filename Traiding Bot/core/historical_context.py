# core/historical_context.py
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class HistoricalContext:
    def __init__(self, data_dir='data/historical', rsi_period=14):
        self.data_dir = data_dir
        self.rsi_period = rsi_period
        self.df = self._load_data()
        self.scaler = None
        self.nn = None
        self._prepare_system()

    def _calculate_rsi(self, close_prices):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _load_data(self):
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        dfs = [pd.read_parquet(os.path.join(self.data_dir, f)) for f in all_files]
        combined = pd.concat(dfs).sort_index().reset_index()
        
        # Calculate features
        combined['RSI'] = self._calculate_rsi(combined['Close'])
        combined['returns_1h'] = combined['Close'].pct_change()
        combined['volume_pct'] = combined['Volume'].pct_change()
        
        return combined.dropna()

    def _prepare_system(self):
        features = self.df[['RSI', 'volume_pct']].values
        self.scaler = StandardScaler().fit(features)
        self.nn = NearestNeighbors(n_neighbors=3).fit(self.scaler.transform(features))

    # MUST INCLUDE THIS METHOD
    def find_similar(self, current_rsi: float, current_volume_pct: float):
        query = np.array([[current_rsi, current_volume_pct]])
        scaled_query = self.scaler.transform(query)
        distances, indices = self.nn.kneighbors(scaled_query)
        return self.df.iloc[indices[0]][['timestamp', 'RSI', 'Volume', 'returns_1h']]