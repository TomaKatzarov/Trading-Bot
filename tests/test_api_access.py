import os
from pathlib import Path
from alpaca_trade_api import REST
from core.news_sentiment import NewsSentimentAnalyzer
from dotenv import load_dotenv

# Load .env with explicit path
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

# VERBOSE DEBUGGING
print("⛔ Env file exists?", env_path.exists())  # Should print True
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
print(f"🔑 Key: {api_key}")  # TEMPORARY - DELETE AFTER TESTING
print(f"🔒 Secret: {secret_key}")  # TEMPORARY - DELETE AFTER TESTING

analyzer = NewsSentimentAnalyzer()
sentiment_data = analyzer.get_all_sentiment_scores()
print(sentiment_data)  # Should print sentiment scores for all symbols

api = REST(
    key_id=api_key,
    secret_key=secret_key,
    base_url="https://data.alpaca.markets/v1beta1/news",  # No trailing slash
    api_version="v2"  # Explicit version specification
)
account = api.get_account()
print("✅ Success! Account:", account)