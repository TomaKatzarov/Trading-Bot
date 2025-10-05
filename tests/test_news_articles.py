from core.news_sentiment import NewsSentimentAnalyzer

# --------------------
# Testing block
# --------------------
if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    
    # Test 1: Fetch and print news articles for a stock symbol.
    symbol = "AAPL"
    news_articles = analyzer.fetch_news(symbol)
    if news_articles:
        print(f"\n✅ Successfully fetched {len(news_articles)} articles for {symbol}")
        print("Sample:", news_articles[:3])
    else:
        print(f"\n❌ No news articles retrieved for {symbol}")
    
    # Test 2: Compute sentiment score for a raw text sample.
    text_sample = "Apple stock surged today after strong earnings reports."
    score_text = analyzer.get_normalized_score(text_sample)
    print(f"\nSentiment Score for text sample: {score_text}")
    
    # Test 3: Compute sentiment score for a stock symbol.
    score_symbol = analyzer.get_normalized_score("AAPL")
    print(f"Sentiment Score for symbol AAPL: {score_symbol}")