from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import asyncio
import json
import pandas as pd
import numpy as np
import requests
import mlflow
import talib
from datetime import datetime, timedelta
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Set API keys
os.environ["FIREWORKS_API_KEY"] = "fw_3ZZmgmxh6hPjXDM3QEdivKQg"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["FMP_API_KEY"] = "6823c7f82ca6a09333c5a296c3269b5a"  # Replace with your Financial Modeling Prep API key

# Configure Fireworks model
fireworks_model = OpenAIModel(
    'accounts/fireworks/models/llama-v3p1-8b-instruct',
    provider=OpenAIProvider(
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=os.environ["FIREWORKS_API_KEY"],
    ),
)

# AGENT 1: PRICE ANALYSIS AGENT
# Define Pydantic models for structured output
class PriceRecommendation(BaseModel):
    direction: str = Field(description="Buy, sell, or hold recommendation")
    confidence: int = Field(description="Confidence level from 1-10")
    target_price: float = Field(description="Target price forecast")
    recommended_allocation: int = Field(description="Recommended portfolio allocation percentage (0-100)")
    reasoning: str = Field(description="Reasoning behind the recommendation")
    ml_prediction_return: float = Field(description="ML model's predicted 5-day return percentage")

class StockPrice(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price")
    previous_close: float = Field(description="Previous day's closing price")
    change_percent: float = Field(description="Percentage change from previous close")
    open_price: float = Field(description="Opening price")
    high_price: float = Field(description="Day's high price")
    low_price: float = Field(description="Day's low price")
    volume: float = Field(description="Trading volume")
    
class TechnicalIndicators(BaseModel):
    rsi: float = Field(description="Relative Strength Index")
    macd: float = Field(description="MACD line value")
    macd_signal: float = Field(description="MACD signal line")
    macd_histogram: float = Field(description="MACD histogram")
    stochastic_k: float = Field(description="Stochastic %K")
    stochastic_d: float = Field(description="Stochastic %D")
    sma_9: float = Field(description="9-day Simple Moving Average")
    sma_20: float = Field(description="20-day Simple Moving Average")
    bb_upper: float = Field(description="Bollinger Band Upper")
    bb_middle: float = Field(description="Bollinger Band Middle")
    bb_lower: float = Field(description="Bollinger Band Lower")

class MLPrediction(BaseModel):
    predicted_return_5d: float = Field(description="Predicted 5-day return percentage")
    prediction: str = Field(description="Direction prediction (BULLISH, BEARISH, NEUTRAL)")
    expected_return: float = Field(description="Expected return value")

class FMPSearchArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")

# Price Analysis Agent
price_analysis_agent = Agent(
    fireworks_model,
    system_prompt=(
        "You are a quantitative financial analyst specializing in technical analysis and price forecasting. "
        "Analyze the provided stock price data, technical indicators, and ML model predictions to generate trading recommendations. "
        "The current price for {ticker} is {price} and current technical indicators show RSI: {rsi}, MACD: {macd}, "
        "Stochastic: K={stoch_k}/D={stoch_d}, Bollinger Bands: Upper={bb_upper}/Middle={bb_middle}/Lower={bb_lower}. "
        "My internal technical model says the next 5 day returns is {ml_return}%. "
        "Provide a clear buy, sell, or hold recommendation with confidence level, target price, "
        "and recommended portfolio allocation. Consider the technical signals and ML prediction carefully."
    ),
    result_type=PriceRecommendation
)

async def get_stock_price(search_args: FMPSearchArgs) -> StockPrice:
    """
    Get current stock price data from Financial Modeling Prep API
    """
    ticker = search_args.ticker
    api_key = os.environ.get("FMP_API_KEY")
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to get stock price: {response.status_code}")
    
    data = response.json()
    if not data:
        raise Exception(f"No data found for ticker: {ticker}")
    
    stock_data = data[0]
    
    return StockPrice(
        ticker=ticker,
        current_price=stock_data.get("price", 0),
        previous_close=stock_data.get("previousClose", 0),
        change_percent=stock_data.get("changesPercentage", 0),
        open_price=stock_data.get("open", 0),
        high_price=stock_data.get("dayHigh", 0),
        low_price=stock_data.get("dayLow", 0),
        volume=stock_data.get("volume", 0)
    )

async def get_technical_indicators(search_args: FMPSearchArgs) -> TechnicalIndicators:
    """
    Get technical indicators using the new FMP API approach with talib calculations
    """
    ticker = search_args.ticker
    api_key = os.environ.get("FMP_API_KEY")
    
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    from_date_obj = yesterday - timedelta(days=90)
    from_date = from_date_obj.strftime('%Y-%m-%d')
    to_date = yesterday_str
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={from_date}&to={to_date}&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to get historical data: {response.status_code}")
    
    data = response.json()
    if 'historical' not in data or not data['historical']:
        raise Exception(f"No historical data found for ticker: {ticker}")
    
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    close = np.array(df['close'].astype(float))
    high = np.array(df['high'].astype(float))
    low = np.array(df['low'].astype(float))
    
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
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        close,
        fastperiod=MACD_FAST,
        slowperiod=MACD_SLOW,
        signalperiod=MACD_SIGNAL
    )
    df['SMA_9'] = talib.SMA(close, timeperiod=9)
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    
    latest_data = df.iloc[-1]
    
    return TechnicalIndicators(
        rsi=float(latest_data['RSI']),
        macd=float(latest_data['MACD']),
        macd_signal=float(latest_data['MACD_signal']),
        macd_histogram=float(latest_data['MACD_hist']),
        stochastic_k=float(latest_data['Stochastic_K']),
        stochastic_d=float(latest_data['Stochastic_D']),
        sma_9=float(latest_data['SMA_9']),
        sma_20=float(latest_data['SMA_20']),
        bb_upper=float(latest_data['BB_Upper']),
        bb_middle=float(latest_data['BB_Middle']),
        bb_lower=float(latest_data['BB_Lower'])
    )

async def get_ml_prediction(search_args: FMPSearchArgs) -> MLPrediction:
    """
    Call ML model from MLflow registry and get prediction for the stock using the latest technical indicators
    """
    ticker = search_args.ticker
    
    try:
        mlflow.set_tracking_uri('http://127.0.0.1:5001')
        model_name = "sample_model_ml"
        model_version = "1"
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        
        price_data = await get_stock_price(search_args)
        indicators_data = await get_technical_indicators(search_args)
        
        vwap = price_data.current_price
        
        input_data = pd.DataFrame({
            'open': [price_data.open_price],
            'high': [price_data.high_price],
            'low': [price_data.low_price],
            'close': [price_data.current_price],
            'volume': [price_data.volume],
            'vwap': [vwap],
            'RSI': [indicators_data.rsi],
            'Stochastic_K': [indicators_data.stochastic_k],
            'Stochastic_D': [indicators_data.stochastic_d],
            'BB_Upper': [indicators_data.bb_upper],
            'BB_Middle': [indicators_data.bb_middle],
            'BB_Lower': [indicators_data.bb_lower],
            'BB_Width': [(indicators_data.bb_upper - indicators_data.bb_lower) / indicators_data.bb_middle],
            'MACD': [indicators_data.macd],
            'MACD_signal': [indicators_data.macd_signal],
            'MACD_hist': [indicators_data.macd_histogram],
            'SMA_9': [indicators_data.sma_9],
            'SMA_20': [indicators_data.sma_20]
        })
        
        predictions = model.predict(input_data)
        
        prediction_direction = "NEUTRAL"
        predicted_return = 0.0
        
        print(f"Raw prediction type: {type(predictions)}")
        print(f"Raw prediction value: {predictions}")
        
        if isinstance(predictions, pd.DataFrame):
            if not predictions.empty:
                # Assume DataFrame has 'prediction' and 'expected_return' columns
                prediction_direction = predictions.get('prediction', ['NEUTRAL']).iloc[0]
                try:
                    predicted_return = float(predictions.get('expected_return', [0.0]).iloc[0])
                except (TypeError, ValueError):
                    predicted_return = 0.0
        elif isinstance(predictions, (list, np.ndarray)):
            if len(predictions) > 0:
                # Assume first element is a dict with 'prediction' and 'expected_return'
                if isinstance(predictions[0], dict):
                    prediction_direction = predictions[0].get('prediction', 'NEUTRAL')
                    try:
                        predicted_return = float(predictions[0].get('expected_return', 0.0))
                    except (TypeError, ValueError):
                        predicted_return = 0.0
                else:
                    # Fallback: assume first element is return, direction based on threshold
                    try:
                        predicted_return = float(predictions[0])
                        prediction_direction = "BULLISH" if predicted_return > 0 else "BEARISH" if predicted_return < 0 else "NEUTRAL"
                    except (TypeError, ValueError):
                        predicted_return = 0.0
        elif isinstance(predictions, dict):
            # Handle dictionary output directly
            prediction_direction = predictions.get('prediction', 'NEUTRAL')
            try:
                predicted_return = float(predictions.get('expected_return', 0.0))
            except (TypeError, ValueError):
                predicted_return = 0.0
        else:
            # Fallback for unexpected types
            try:
                predicted_return = float(predictions)
                prediction_direction = "BULLISH" if predicted_return > 0 else "BEARISH" if predicted_return < 0 else "NEUTRAL"
            except (TypeError, ValueError):
                predicted_return = 0.0
                prediction_direction = "NEUTRAL"
        
        return MLPrediction(
            predicted_return_5d=predicted_return,
            prediction=prediction_direction,
            expected_return=predicted_return
        )
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return MLPrediction(
            predicted_return_5d=0.0,
            prediction="NEUTRAL",
            expected_return=0.0
        )

async def analyze_stock_price(ticker: str) -> dict:
    """
    Run the price analysis agent for a specific stock ticker.
    
    Args:
        ticker: Stock ticker symbol to analyze
        
    Returns:
        JSON with structured price analysis and recommendation
    """
    try:
        price_args = FMPSearchArgs(ticker=ticker)
        price_data = await get_stock_price(price_args)
        indicators_data = await get_technical_indicators(price_args)
        ml_prediction = await get_ml_prediction(price_args)
        
        prompt = f"""
        Analyze this stock and provide a recommendation:
        
        The current price for {ticker} is ${price_data.current_price} and current technical indicators are:
        - RSI: {indicators_data.rsi}
        - MACD: {indicators_data.macd} (Signal: {indicators_data.macd_signal}, Histogram: {indicators_data.macd_histogram})
        - Stochastic: K={indicators_data.stochastic_k}/D={indicators_data.stochastic_d}
        - SMA: 9-day={indicators_data.sma_9}, 20-day={indicators_data.sma_20}
        - Bollinger Bands: Upper={indicators_data.bb_upper}, Middle={indicators_data.bb_middle}, Lower={indicators_data.bb_lower}
        
        My internal technical model predicts a {ml_prediction.prediction} trend with a next 5 day return of {ml_prediction.predicted_return_5d*100}%.
        
        What do you recommend for direction and new cash deployment?
        """
        
        result = await price_analysis_agent.run(prompt)
        
        analysis_result = result.data.model_dump()
        analysis_result["ml_prediction_return"] = ml_prediction.predicted_return_5d
        
        return analysis_result
    except Exception as e:
        print(f"Error in price analysis: {e}")
        return {
            "direction": "HOLD", 
            "confidence": 5,
            "target_price": price_data.current_price if 'price_data' in locals() else 0,
            "recommended_allocation": 0,
            "reasoning": f"Error in analysis: {str(e)}",
            "ml_prediction_return": 0.0
        }

# AGENT 2: MARKET NEWS SENTIMENT ANALYSIS
class StockMention(BaseModel):
    stock_name: str = Field(description="Name of the stock mentioned")
    ticker_symbol: Optional[str] = Field(description="Stock ticker symbol")
    mention_context: str = Field(description="Context in which the stock was mentioned")
    sentiment: str = Field(description="Sentiment of the mention (positive, negative, neutral)")
    sentiment_score: float = Field(description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    sentiment_number: int = Field(description="Sentiment as integer from 1 (very negative) to 10 (very positive)")

class MarketTrend(BaseModel):
    trend_type: str = Field(description="Type of market trend (bullish, bearish, sideways)")
    description: str = Field(description="Description of the market trend")
    affected_sectors: List[str] = Field(description="Sectors affected by this trend")

class NewsAnalysisResult(BaseModel):
    overall_market_sentiment: str = Field(description="Overall market sentiment from the news")
    fear_index: int = Field(description="Fear index from 1 (extreme fear) to 100 (extreme greed)")
    optimism_score: int = Field(description="Optimism score from 1 (extremely pessimistic) to 100 (extremely optimistic)")
    key_stocks_mentioned: List[StockMention] = Field(description="Details about key stocks mentioned in the news")
    market_trends: List[MarketTrend] = Field(description="Market trends identified in the news")
    summary: str = Field(description="Summary of the market news and sentiment")
    insights: str = Field(description="Investment insights and recommendations based on the analysis")
    source_urls: List[str] = Field(description="URLs of the news sources analyzed")

class TavilySearchArgs(BaseModel):
    query: str = Field(description="Search query for finding market news")

# Store source URLs globally for simplicity
news_source_urls = []

# Set up the market news sentiment analysis agent
market_sentiment_agent = Agent(
    fireworks_model,
    system_prompt=(
        "You are a financial analyst specializing in market sentiment analysis. "
        "Analyze the provided market news from CNBC and extract sentiment information about stocks and markets. "
        "Focus on identifying positive, negative, or neutral sentiment, key stock mentions, and market trends. "
        "Provide a structured analysis with sentiment scores and summaries. "
        "For each stock, include both a sentiment score (float from -1.0 to 1.0) and a sentiment number (integer from 1 to 10). "
        "Calculate a fear index (integer from 1-100) where 1 is extreme fear and 100 is extreme greed. "
        "Calculate an optimism score (integer from 1-100) where 1 is extremely pessimistic and 100 is extremely optimistic. "
        "Also provide investment insights and recommendations based on your analysis of the market sentiment. "
        "Be objective and base your analysis solely on the information provided."
    ),
    result_type=NewsAnalysisResult
)

@market_sentiment_agent.tool
async def search_cnbc_news(ctx: RunContext[None], search_args: TavilySearchArgs) -> str:
    """
    Search for market news on CNBC related to specific stocks or financial information.
    """
    global news_source_urls
    from tavily import TavilyClient
   
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
   
    search_query = f"{search_args.query} site:www.cnbc.com"
    search_result = tavily_client.search(
        query=search_query,
        search_depth="advanced",
        include_domains=["www.cnbc.com"],
        max_results=5
    )
   
    content = ""
    urls = []
   
    for result in search_result.get("results", []):
        content += f"\nArticle Title: {result.get('title', '')}\n"
        content += f"Content: {result.get('content', '')}\n"
        content += f"URL: {result.get('url', '')}\n\n"
        urls.append(result.get('url', ''))
   
    news_source_urls = urls
    return content

# AGENT 3: CONSOLIDATED ANALYSIS AGENT
class ConsolidatedRecommendation(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    direction: str = Field(description="Buy, sell, or hold recommendation")
    confidence: int = Field(description="Confidence level from 1-10")
    exposure_percentage: int = Field(description="Recommended portfolio exposure percentage (0-100)")
    price_analysis_summary: str = Field(description="Summary of price and technical analysis")
    sentiment_analysis_summary: str = Field(description="Summary of news sentiment analysis")
    expected_return_5d: float = Field(description="Expected 5-day return percentage")
    risk_level: str = Field(description="Low, medium, or high risk assessment")
    reasoning: str = Field(description="Consolidated reasoning combining technical and sentiment analysis")
    action_steps: List[str] = Field(description="Recommended specific actions to take")

# Create Agent 3 for consolidated analysis
consolidated_analysis_agent = Agent(
    fireworks_model,
    system_prompt=(
        "You are a comprehensive financial analyst who combines quantitative technical analysis with qualitative "
        "market sentiment analysis to provide holistic investment recommendations. "
        "Your role is to synthesize the inputs from both technical price analysis and news sentiment analysis "
        "to generate a more complete picture of market conditions and investment opportunities. "
        "Compare and contrast the recommendations from each analysis type, explain any discrepancies or agreements, "
        "and provide a final recommendation that considers both perspectives. "
        "Be specific about directional recommendations (buy, sell, hold), position sizing, risk levels, and "
        "expected returns. When technical analysis and sentiment analysis disagree, explain why and which "
        "you believe is more reliable in the current context. "
        "Provide clear, actionable steps for investors to follow based on your consolidated analysis."
    ),
    result_type=ConsolidatedRecommendation
)

async def analyze_market_sentiment(stock_query: str) -> dict:
    """
    Analyze market sentiment for a specific stock or financial topic.
   
    Args:
        stock_query: Stock name, ticker symbol, or financial topic to analyze
   
    Returns:
        JSON with structured analysis of market news sentiment
    """
    global news_source_urls
    news_source_urls = []
   
    result = await market_sentiment_agent.run(
        f"Analyze the market news and sentiment for {stock_query}. Focus on recent news from CNBC. Include investment insights and recommendations based on your analysis."
    )
   
    analysis_result = result.data
    analysis_result.source_urls = news_source_urls
   
    return analysis_result.model_dump()

async def generate_consolidated_analysis(ticker: str, price_analysis: dict, sentiment_analysis: dict) -> dict:
    """
    Generate a consolidated analysis based on both price and sentiment analysis.
    
    Args:
        ticker: Stock ticker symbol
        price_analysis: Dict result from price analysis agent
        sentiment_analysis: Dict result from sentiment analysis agent
        
    Returns:
        Dict with consolidated analysis and recommendation
    """
    price_summary = f"""
    Technical Analysis for {ticker}:
    - Direction: {price_analysis.get('direction', 'Unknown')}
    - Confidence: {price_analysis.get('confidence', 0)}/10
    - Target Price: ${price_analysis.get('target_price', 0)}
    - Recommended Allocation: {price_analysis.get('recommended_allocation', 0)}%
    - ML Predicted 5-day Return: {price_analysis.get('ml_prediction_return', 0)*100}%
    - Technical Reasoning: {price_analysis.get('reasoning', 'No reasoning provided')}
    """
    
    sentiment_summary = f"""
    Sentiment Analysis for {ticker}:
    - Overall Market Sentiment: {sentiment_analysis.get('overall_market_sentiment', 'Unknown')}
    - Fear Index: {sentiment_analysis.get('fear_index', 0)}/100
    - Optimism Score: {sentiment_analysis.get('optimism_score', 0)}/100
    - Key Insight: {sentiment_analysis.get('insights', 'No insights provided')}
    """
    
    result = await consolidated_analysis_agent.run(
        f"Based on the technical analysis and the news sentiment analysis, what should we do today for direction and exposure for {ticker}?\n\n"
        f"{price_summary}\n\n{sentiment_summary}"
    )
    
    analysis_result = result.data
    analysis_result.ticker = ticker
    
    return analysis_result.model_dump()

async def run_financial_analysis_system(ticker: str) -> dict:
    """
    Run the complete financial analysis system with all three agents.
    
    Args:
        ticker: Stock ticker symbol to analyze
        
    Returns:
        JSON with consolidated analysis from all three agents
    """
    price_analysis = await analyze_stock_price(ticker)
    sentiment_analysis = await analyze_market_sentiment(f"{ticker} stock")
    consolidated_analysis = await generate_consolidated_analysis(ticker, price_analysis, sentiment_analysis)
    
    return {
        "ticker": ticker,
        "price_analysis": price_analysis,
        "sentiment_analysis": sentiment_analysis,
        "consolidated_analysis": consolidated_analysis
    }

def run_financial_analysis_system_sync(ticker: str) -> dict:
    """
    Synchronous version of the financial analysis system.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(run_financial_analysis_system(ticker))
        return result
    finally:
        loop.close()

if __name__ == "__main__":
    result = run_financial_analysis_system_sync("AAPL")
    print(json.dumps(result, indent=2))