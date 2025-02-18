from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from typing import Dict, List
import os
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AlpacaTradingService:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        self.base_url = "https://paper-api.alpaca.markets"  # Paper trading URL
        self.trading_client = None
        
        logger.info("Initializing Alpaca Trading Service")
        logger.info(f"API Key present: {bool(self.api_key)}")
        logger.info(f"API Secret present: {bool(self.api_secret)}")
        
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials not found in environment variables")
            return
        
        # Test network connectivity
        try:
            import socket
            socket.create_connection(("paper-api.alpaca.markets", 443), timeout=5)
            logger.info("Network connection to Alpaca API is available")
        except Exception as e:
            logger.error(f"Network connectivity test failed: {str(e)}")
            logger.warning("Cannot connect to Alpaca API - check your internet connection")
            return
        
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=True,
                url_override="https://paper-api.alpaca.markets"  # Explicitly set API URL
            )
            # Test connection
            try:
                account = self.trading_client.get_account()
                logger.info("Successfully connected to Alpaca Trading API")
            except Exception as e:
                logger.error(f"Failed to connect to Alpaca API: {str(e)}")
                self.trading_client = None
                return
            
            logger.info("Successfully created Alpaca Trading Client")
        except Exception as e:
            logger.error(f"Error creating Alpaca Trading Client: {str(e)}")
            self.trading_client = None
            raise

    def create_portfolio_orders(self, portfolio_allocation: List[Dict]):
        """
        Create orders based on portfolio allocation
        """
        if not self.trading_client:
            return {
                "orders": [],
                "total_orders": len(portfolio_allocation),
                "successful_orders": 0,
                "error": "Trading client not initialized - check your Alpaca API credentials"
            }

        logger.info(f"Creating portfolio orders for {len(portfolio_allocation)} positions")
        
        # First, close all existing positions
        try:
            logger.info("Closing existing positions")
            self.close_all_positions()
            logger.info("Successfully closed existing positions")
            # Wait for positions to be closed
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error closing existing positions: {str(e)}")
            return {
                "orders": [],
                "total_orders": len(portfolio_allocation),
                "successful_orders": 0,
                "error": f"Error closing existing positions: {str(e)}"
            }
        
        orders = []
        successful_orders = 0
        
        for allocation in portfolio_allocation:
            try:
                symbol = allocation["symbol"].upper()
                quantity = int(allocation["quantity"])
                
                if quantity <= 0:
                    logger.warning(f"Skipping {symbol} - quantity is 0")
                    continue
                
                logger.info(f"Attempting to buy {quantity} shares of {symbol}")
                
                # Submit order
                try:
                    market_order = MarketOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    order = self.trading_client.submit_order(market_order)
                    
                    # Wait briefly for order to be processed
                    time.sleep(1)
                    
                    order_response = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "order_id": order.id,
                        "status": order.status,
                        "created_at": str(order.created_at),
                        "filled_qty": float(order.filled_qty) if order.filled_qty else None,
                        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
                    }
                    
                    successful_orders += 1
                    orders.append(order_response)
                    logger.info(f"Successfully bought {quantity} shares of {symbol}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error buying {symbol}: {error_msg}")
                    orders.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "status": "failed",
                        "error": error_msg
                    })
                    
            except Exception as e:
                logger.error(f"Error processing order for {allocation['symbol']}: {str(e)}")
                orders.append({
                    "symbol": allocation["symbol"],
                    "quantity": allocation["quantity"],
                    "status": "failed",
                    "error": str(e)
                })
        
        logger.info(f"Completed portfolio orders. Success: {successful_orders}/{len(portfolio_allocation)}")
        
        return {
            "orders": orders,
            "total_orders": len(portfolio_allocation),
            "successful_orders": successful_orders
        }

    def get_account(self):
        """Get account information"""
        if not self.trading_client:
            raise ValueError("Trading client not initialized - check your Alpaca API credentials")
        try:
            logger.info("Attempting to get account information")
            account = self.trading_client.get_account()
            logger.info(f"Successfully retrieved account information")
            return account
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise ValueError(f"Error accessing Alpaca account: {str(e)}")

    def get_positions(self):
        """Get current positions"""
        try:
            logger.info("Getting current positions")
            positions = self.trading_client.get_all_positions()
            logger.info(f"Successfully retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.info("Attempting to close all positions")
            self.trading_client.close_all_positions()
            logger.info("Successfully closed all positions")
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            raise

    def test_buy_single_stock(self, symbol: str, quantity: int):
        """
        Test method to buy a single stock
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares to buy
        """
        try:
            # Create market order
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.trading_client.submit_order(market_order)
            return {
                "symbol": symbol,
                "quantity": quantity,
                "order_id": order.id,
                "status": order.status,
                "created_at": order.created_at,
                "filled_at": order.filled_at,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price
            }
        except Exception as e:
            raise Exception(f"Error placing order: {str(e)}")

# Initialize trading service
trading_service = AlpacaTradingService()