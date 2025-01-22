from core.realtime_handler import RealTimeDataHandler

if __name__ == "__main__":
    print("Testing IEX symbol availability...")
    try:
        handler = RealTimeDataHandler()
        print(f"First 10 symbols: {handler.valid_symbols[:10]}")
        print(f"Total symbols: {len(handler.valid_symbols)}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Possible solutions:")
        print("1. Verify API credentials in Credential.env")
        print("2. Ensure 'alpaca-py' package is installed")
        print("3. Check internet connection")