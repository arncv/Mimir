import os
from dotenv import load_dotenv
import logging
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StockSentimentAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.available = False
        
        if self.news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=self.news_api_key)
                self.available = True
            except Exception as e:
                logger.error(f"Failed to initialize NewsAPI client: {str(e)}")
        else:
            logger.warning("NEWS_API_KEY not found in environment variables")
    
    def get_news_and_sentiment(self, days=7):
        """Get news articles and analyze sentiment for a stock"""
        if not self.available:
            return {
                "status": "error",
                "message": "News API not configured",
                "analysis": "Sentiment analysis not available - API key not configured",
                "news": []
            }
        
        try:
            # Get news articles
            news = self.news_client.get_everything(
                q=f"{self.symbol} stock",
                language='en',
                from_param=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                sort_by='relevancy'
            )
            
            if not news['articles']:
                return {
                    "status": "success",
                    "message": "No recent news found",
                    "analysis": "No recent news articles found for sentiment analysis",
                    "news": []
                }
            
            # Take top 10 most relevant articles
            articles = news['articles'][:10]
            
            return {
                "status": "success",
                "message": "Successfully retrieved news articles",
                "analysis": self._analyze_sentiment(articles),
                "news": articles
            }
            
        except Exception as e:
            logger.error(f"Error getting news for {self.symbol}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "analysis": "Error retrieving news articles",
                "news": []
            }
    
    def _analyze_sentiment(self, articles):
        """Basic sentiment analysis based on news headlines"""
        if not articles:
            return "No articles available for analysis"
        
        summary = (
            f"Found {len(articles)} recent news articles about {self.symbol}.\n\n"
            "Recent Headlines:\n" +
            "\n".join([f"- {article['title']}" for article in articles[:5]]) +
            "\n\nNote: Detailed sentiment analysis requires AI service which is currently unavailable."
        )
        
        return summary