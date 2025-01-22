# test_historical.py
from core.historical_context import HistoricalContext

def test_historical_system():
    print("\nStarting Historical Context Test...")
    try:
        hc = HistoricalContext(rsi_period=14)
        
        # Test RSI calculation
        print("\nFirst 5 RSI Values:")
        print(hc.df['RSI'].head())
        
        # Test similarity search
        similar = hc.find_similar(current_rsi=60, current_volume_pct=0.1)
        print("\nSimilar Cases:")
        print(similar)
        
    except Exception as e:
        print(f"Test Failed: {str(e)}")

if __name__ == "__main__":
    test_historical_system()