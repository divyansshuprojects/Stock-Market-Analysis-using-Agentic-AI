import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
 
# Load environment variables
load_dotenv(override=True)
 
# Initialize FastMCP
mcp = FastMCP(
    name="price_prediction", 
    version="1.0.0",
    description="Price prediction tool for stock market analysis using technical indicators"
)
 
@mcp.tool()
async def get_price_prediction(
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    vwap: float,
    rsi: float,
    stochastic_k: float,
    stochastic_d: float,
    bb_upper: float,
    bb_middle: float,
    bb_lower: float,
    bb_width: float,
    macd: float,
    macd_signal: float,
    macd_hist: float,
    sma_9: float,
    sma_20: float,
    prediction_endpoint: str = "http://localhost:8000/predict/json"
) -> dict:
    """
    Get price prediction using technical indicators from the prediction API.
    Args:
        open: The opening price of the stock
        high: The highest price of the stock in the period
        low: The lowest price of the stock in the period
        close: The closing price of the stock
        volume: Trading volume
        vwap: Volume-weighted average price
        rsi: Relative Strength Index value
        stochastic_k: Stochastic Oscillator %K value
        stochastic_d: Stochastic Oscillator %D value
        bb_upper: Bollinger Bands upper band value
        bb_middle: Bollinger Bands middle band value
        bb_lower: Bollinger Bands lower band value
        bb_width: Bollinger Bands width
        macd: Moving Average Convergence Divergence value
        macd_signal: MACD signal line value
        macd_hist: MACD histogram value
        sma_9: 9-period Simple Moving Average value
        sma_20: 20-period Simple Moving Average value
        prediction_endpoint: URL of the prediction API endpoint
    Returns:
        Dictionary containing the prediction results
    """
    try:
        # Prepare the data payload
        data = {
            "open": [open],
            "high": [high],
            "low": [low],
            "close": [close],
            "volume": [volume],
            "vwap": [vwap],
            "RSI": [rsi],
            "Stochastic_K": [stochastic_k],
            "Stochastic_D": [stochastic_d],
            "BB_Upper": [bb_upper],
            "BB_Middle": [bb_middle],
            "BB_Lower": [bb_lower],
            "BB_Width": [bb_width],
            "MACD": [macd],
            "MACD_signal": [macd_signal],
            "MACD_hist": [macd_hist],
            "SMA_9": [sma_9],
            "SMA_20": [sma_20]
        }
        # Make the POST request
        response = requests.post(prediction_endpoint, json=data)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            return {
                "prediction": result,
                "status": "success"
            }
        else:
            return {
                "error": f"API request failed with status code: {response.status_code}",
                "response": response.text,
                "status": "error"
            }
    except Exception as e:
        return {
            "error": f"Error getting price prediction: {str(e)}",
            "status": "error"
        }
 
if __name__ == "__main__":
    mcp.run()