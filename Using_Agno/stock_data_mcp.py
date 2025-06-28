import os
import pandas as pd
import numpy as np
import requests
import talib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv(override=True)

# Initialize FastMCP
mcp = FastMCP(
    name="stock_data",
    version="1.0.0",
    description="Stock data tools for retrieving real-time price information and technical indicators",
    host="127.0.0.1",  # Host address (localhost)
    port=8002,  # Port number for the server (changed to avoid conflicts)
)

# Set API key
api_key = os.getenv("FMP_API_KEY", "6823c7f82ca6a09333c5a296c3269b5a")

@mcp.tool()
async def get_stock_price(ticker: str) -> dict:
    """Get current stock price data from Financial Modeling Prep API"""
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {
                "error": f"Failed to get stock price: {response.status_code}",
                "ticker": ticker,
                "current_price": 0,
                "previous_close": 0,
                "change_percent": 0,
                "open_price": 0,
                "high_price": 0,
                "low_price": 0,
                "volume": 0
            }
        
        data = response.json()
        if not data:
            return {
                "error": f"No data found for ticker: {ticker}",
                "ticker": ticker,
                "current_price": 0,
                "previous_close": 0,
                "change_percent": 0,
                "open_price": 0,
                "high_price": 0,
                "low_price": 0,
                "volume": 0
            }
        
        stock_data = data[0]
        
        return {
            "ticker": ticker,
            "current_price": stock_data.get("price", 0),
            "previous_close": stock_data.get("previousClose", 0),
            "change_percent": stock_data.get("changesPercentage", 0),
            "open_price": stock_data.get("open", 0),
            "high_price": stock_data.get("dayHigh", 0),
            "low_price": stock_data.get("dayLow", 0),
            "volume": stock_data.get("volume", 0)
        }
    except Exception as e:
        return {
            "error": f"Exception: {str(e)}",
            "ticker": ticker,
            "current_price": 0,
            "previous_close": 0,
            "change_percent": 0,
            "open_price": 0,
            "high_price": 0,
            "low_price": 0,
            "volume": 0
        }

@mcp.tool()
async def get_technical_indicators(ticker: str) -> dict:
    """Get technical indicators using historical price data and talib calculations"""
    try:
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        
        from_date_obj = yesterday - timedelta(days=90)
        from_date = from_date_obj.strftime('%Y-%m-%d')
        to_date = yesterday_str
        
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={from_date}&to={to_date}&apikey={api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return {
                "error": f"Failed to get historical data: {response.status_code}",
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_histogram": 0.0,
                "stochastic_k": 50.0,
                "stochastic_d": 50.0,
                "sma_9": 0.0,
                "sma_20": 0.0,
                "bb_upper": 0.0,
                "bb_middle": 0.0,
                "bb_lower": 0.0
            }
        
        data = response.json()
        if 'historical' not in data or not data['historical']:
            return {
                "error": f"No historical data found for ticker: {ticker}",
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_histogram": 0.0,
                "stochastic_k": 50.0,
                "stochastic_d": 50.0,
                "sma_9": 0.0,
                "sma_20": 0.0,
                "bb_upper": 0.0,
                "bb_middle": 0.0,
                "bb_lower": 0.0
            }
        
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        close = np.array(df['close'].astype(float))
        high = np.array(df['high'].astype(float))
        low = np.array(df['low'].astype(float))
        
        # Calculate indicators with talib
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['Stochastic_K'], df['Stochastic_D'] = talib.STOCH(
            high, low, close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['SMA_9'] = talib.SMA(close, timeperiod=9)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        
        latest_data = df.iloc[-1]
        print(latest_data)
        
        return {
            "rsi": float(latest_data['RSI']),
            "macd": float(latest_data['MACD']),
            "macd_signal": float(latest_data['MACD_signal']),
            "macd_histogram": float(latest_data['MACD_hist']),
            "stochastic_k": float(latest_data['Stochastic_K']),
            "stochastic_d": float(latest_data['Stochastic_D']),
            "sma_9": float(latest_data['SMA_9']),
            "sma_20": float(latest_data['SMA_20']),
            "bb_upper": float(latest_data['BB_Upper']),
            "bb_middle": float(latest_data['BB_Middle']),
            "bb_lower": float(latest_data['BB_Lower'])
        }
    except Exception as e:
        return {
            "error": f"Exception: {str(e)}",
            "rsi": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_histogram": 0.0,
            "stochastic_k": 50.0,
            "stochastic_d": 50.0,
            "sma_9": 0.0,
            "sma_20": 0.0,
            "bb_upper": 0.0,
            "bb_middle": 0.0,
            "bb_lower": 0.0
        }

if __name__ == "__main__":
    mcp.run(transport="sse")