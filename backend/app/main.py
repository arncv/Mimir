from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import google.generativeai as genai
import ta
import numpy as np
from typing import List, Optional, Dict, Literal
import time
from functools import wraps
import re
from app.trading import trading_service
from app.portfolio_optimizer import portfolio_optimizer
import logging
from app.portfolio_generator import portfolio_generator, PortfolioGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Financial Product Backend API")

# Configure CORS with specific origins
origins = [
    "http://localhost:3000",    # Next.js development server
    "http://127.0.0.1:3000",
    "http://localhost:3001",    # Next.js alternative port
    "http://127.0.0.1:3001",
    "http://localhost:8000",    # Alternative port
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

class AIService:
    def __init__(self):
        self.model = None
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.max_requests_per_minute = 60
        self.backoff_time = 1
        self.available = False
        try:
            self.initialize_model()
        except Exception as e:
            logger.warning(f"AI service initialization failed: {str(e)}")
            logger.warning("AI features will be disabled")
    
    def initialize_model(self):
        """Initialize the Gemini model with error handling"""
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_AI_API_KEY not found in environment variables")
            return
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Test connection
            test_response = self.model.generate_content("Test connection")
            if test_response and test_response.text:
                self.available = True
                logger.info("AI service initialized successfully")
            else:
                logger.warning("Failed to get response from Gemini API")
        except Exception as e:
            logger.error(f"Error initializing AI service: {str(e)}")
    
    def generate_content(self, prompt):
        """Generate content with rate limiting and error handling"""
        if not self.available:
            return type('Response', (), {'text': 'AI analysis not available - API key not configured or invalid'})()
        
        current_time = time.time()
        
        if current_time - self.last_request_time < 1:
            time.sleep(1)
        
        if self.requests_this_minute >= self.max_requests_per_minute:
            time.sleep(self.backoff_time)
            self.requests_this_minute = 0
            self.backoff_time *= 2
        else:
            self.backoff_time = 1
        
        try:
            response = self.model.generate_content(prompt)
            self.last_request_time = time.time()
            self.requests_this_minute += 1
            return response
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return type('Response', (), {'text': f'Error generating AI analysis: {str(e)}'})()

# Initialize services
ai_service = AIService()
news_api_key = os.getenv("NEWS_API_KEY")
if news_api_key:
    from newsapi import NewsApiClient
    news_client = NewsApiClient(api_key=news_api_key)
else:
    news_client = None

portfolio_generator = PortfolioGenerator()

# Models
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    period: str = Field(default="1y", description="Time period for analysis")
    interval: str = Field(default="1d", description="Data interval")

class SectorPreference(BaseModel):
    sector: str
    weight: float = Field(..., ge=0, le=100, description="Sector weight (0-100)")

class PortfolioRequest(BaseModel):
    investment_amount: float = Field(..., ge=1000, description="Amount to invest")
    risk_appetite: Literal["conservative", "moderate", "aggressive"]
    investment_period: int = Field(..., ge=1, le=30, description="Investment period in years")
    company_count: int = Field(..., ge=5, le=30, description="Number of companies")

class SentimentRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol for sentiment analysis")
    days: int = Field(default=7, ge=1, le=30, description="Analysis period in days")

class ComprehensivePortfolioRequest(BaseModel):
    investment_amount: float = Field(..., ge=1000, description="Amount to invest")
    risk_appetite: Literal["conservative", "moderate", "aggressive"]
    investment_period: int = Field(..., ge=1, le=30, description="Investment period in years")
    company_count: int = Field(..., ge=5, le=30, description="Number of companies")
    sectors: Optional[List[str]] = None
    esg_focus: Optional[bool] = False

# Endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Financial Product Backend API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/portfolio/generate")
async def generate_portfolio(request: PortfolioRequest):
    """Generate investment portfolio based on user preferences"""
    try:
        logger.info(f"Received portfolio request: {request}")
        
        result = portfolio_generator.generate_portfolio({
            'risk_appetite': request.risk_appetite,
            'investment_amount': request.investment_amount,
            'investment_period': request.investment_period,
            'company_count': request.company_count
        })
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a stock with technical indicators and AI insights"""
    try:
        symbol = request.symbol.strip().upper()
        stock = yf.Ticker(symbol)
        
        # Get historical data
        hist = stock.history(period=request.period, interval=request.interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate technical indicators
        hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
        hist['SMA50'] = ta.trend.sma_indicator(hist['Close'], window=50)
        hist['RSI'] = ta.momentum.rsi(hist['Close'])
        hist['MACD'] = ta.trend.macd_diff(hist['Close'])
        
        # Get info and generate analysis
        info = stock.info
        current_price = hist['Close'].iloc[-1]
        change_percent = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        
        analysis = ai_service.generate_content(
            f"Analyze {symbol} stock: Current price ${current_price:.2f}, "
            f"Change: {change_percent:.2f}%, "
            f"RSI: {hist['RSI'].iloc[-1]:.2f}, "
            f"Industry: {info.get('industry', 'Unknown')}"
        )
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "change_percent": change_percent,
            "technical_indicators": {
                "sma20": float(hist['SMA20'].iloc[-1]),
                "sma50": float(hist['SMA50'].iloc[-1]),
                "rsi": float(hist['RSI'].iloc[-1]),
                "macd": float(hist['MACD'].iloc[-1])
            },
            "company_info": {
                "name": info.get('longName', symbol),
                "industry": info.get('industry', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "market_cap": info.get('marketCap', None)
            },
            "analysis": analysis.text if analysis else "Analysis not available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a given stock"""
    from app.sentiment_analyzer import StockSentimentAnalyzer
    analyzer = StockSentimentAnalyzer(request.symbol)
    result = analyzer.get_news_and_sentiment(days=request.days)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result['message'])
    
    return result

@app.post("/portfolio/execute")
async def execute_portfolio(portfolio_data: Dict):
    """Execute the generated portfolio"""
    try:
        stock_recommendations = portfolio_data.get("portfolio", {}).get("recommendations", {}).get("stock_recommendations", {})
        
        if not stock_recommendations:
            raise HTTPException(status_code=400, detail="No valid portfolio recommendations found")
        
        orders = []
        for sector_stocks in stock_recommendations.values():
            for stock in sector_stocks:
                orders.append({
                    "symbol": stock["symbol"],
                    "quantity": stock["suggested_shares"]
                })
        
        result = trading_service.create_portfolio_orders(orders)
        
        return {
            "success": True,
            "message": f"Successfully executed {result['successful_orders']} out of {result['total_orders']} orders",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error executing portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))