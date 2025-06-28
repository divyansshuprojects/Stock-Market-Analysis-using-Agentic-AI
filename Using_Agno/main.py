import asyncio
import sys
import os
import argparse
import logging
import json
import time
from textwrap import dedent
from typing import Iterator
from filelock import FileLock
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.workflow import Workflow, RunResponse
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from web_search_mcp import search_cnbc_news
from stock_data_mcp import get_stock_price, get_technical_indicators
from ml_prediction_mcp import get_price_prediction

# Set up logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcode all API keys
AGNO_API_KEY = "ag-2q7ymCqaJ96JXvGmizvgtBapbR7ZhpjPHp5fwhJpupk"
FIREWORKS_API_KEY = "fw_3ZZmgmxh6hPjXDM3QEdivKQg"
TAVILY_API_KEY = "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"  # Replace with your actual Tavily API key
FMP_API_KEY = "6823c7f82ca6a09333c5a296c3269b5a"  # Replace with your actual FMP API key

# Set the keys in the environment (for MCP tools)
os.environ["AGNO_API_KEY"] = AGNO_API_KEY
os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["FMP_API_KEY"] = FMP_API_KEY

# Cache file for persistence across processes
CACHE_FILE = "cache.json"
CACHE_LOCK = "cache.json.lock"

# Load cache from file, or initialize if it doesn't exist
def load_cache():
    with FileLock(CACHE_LOCK):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    return {
        'stock_price': {},
        'technical_indicators': {},
        'news': {},
        'predictions': {}
    }

# Save cache to file
def save_cache(cache):
    with FileLock(CACHE_LOCK):
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)

# Initialize cache
_cache = load_cache()

async def preload_data(ticker: str = "AAPL", retries: int = 2):
    """Preload data for the given ticker with retries."""
    for attempt in range(retries + 1):
        try:
            # Preload stock price
            stock_data = await get_stock_price(ticker)
            _cache['stock_price'][ticker] = stock_data
            logger.info(f"Successfully preloaded stock price for {ticker}")

            # Preload technical indicators
            indicators = await get_technical_indicators(ticker)
            _cache['technical_indicators'][ticker] = indicators
            logger.info(f"Successfully preloaded technical indicators for {ticker}")

            # Preload news
            news = await search_cnbc_news(f"{ticker} stock")
            logger.debug(f"Raw news response for {ticker}: {news}")
            if news and 'content' in news and 'urls' in news:
                _cache['news'][ticker] = news
                logger.info(f"Successfully preloaded news for {ticker}")
            else:
                logger.warning(f"No valid news data returned for {ticker}, attempt {attempt + 1}")
                _cache['news'][ticker] = {'content': f"No recent news found for {ticker}", 'urls': []}
                continue

            # Preload prediction
            vwap = (stock_data.get('high_price', 0) + stock_data.get('low_price', 0) + stock_data.get('current_price', 0)) / 3
            bb_width = (indicators.get('bb_upper', 0) - indicators.get('bb_lower', 0)) / indicators.get('bb_middle', 0) if indicators.get('bb_middle', 0) else 0
            prediction = await get_price_prediction(
                open=stock_data.get('open_price', 0),
                high=stock_data.get('high_price', 0),
                low=stock_data.get('low_price', 0),
                close=stock_data.get('current_price', 0),
                volume=stock_data.get('volume', 0),
                vwap=vwap,
                rsi=indicators.get('rsi', 50.0),
                stochastic_k=indicators.get('stochastic_k', 50.0),
                stochastic_d=indicators.get('stochastic_d', 50.0),
                bb_upper=indicators.get('bb_upper', 0),
                bb_middle=indicators.get('bb_middle', 0),
                bb_lower=indicators.get('bb_lower', 0),
                bb_width=bb_width,
                macd=indicators.get('macd', 0),
                macd_signal=indicators.get('macd_signal', 0),
                macd_hist=indicators.get('macd_histogram', 0),
                sma_9=indicators.get('sma_9', 0),
                sma_20=indicators.get('sma_20', 0),
            )
            if prediction and 'prediction' in prediction:
                _cache['predictions'][ticker] = prediction
                logger.info(f"Successfully preloaded prediction for {ticker}")
            else:
                logger.warning(f"No valid prediction data returned for {ticker}, attempt {attempt + 1}")
                continue

            break  # Success, exit retry loop
        except Exception as e:
            logger.error(f"Error preloading data for {ticker}, attempt {attempt + 1}: {str(e)}")
            if attempt == retries:
                logger.error(f"Failed to preload data for {ticker} after {retries + 1} attempts")
                _cache['news'][ticker] = {'content': f"Failed to fetch news for {ticker}", 'urls': []}
                _cache['predictions'][ticker] = {'prediction': f"Failed to generate prediction for {ticker}"}
            else:
                logger.info(f"Retrying after 5 seconds due to error: {str(e)}")
                await asyncio.sleep(5)
    save_cache(_cache)

# Synchronous tool wrappers
def web_search_tool(ticker: str) -> str:
    """Synchronous wrapper for news search using cached data with fallback."""
    logger.debug(f"Starting web_search_tool for {ticker}, current cache: {_cache['news'].get(ticker)}")
    if ticker not in _cache['news']:
        logger.info(f"Ticker {ticker} not in cache, preloading now")
        asyncio.run(preload_data(ticker, retries=2))
    logger.debug(f"Cache in web_search_tool for {ticker} after preload: {_cache['news'].get(ticker)}")
    news = _cache['news'].get(ticker, {'content': f"No news data for {ticker}", 'urls': []})
    if 'error' in news or not news.get('content'):
        logger.warning(f"News retrieval failed for {ticker}, using fallback")
        return f"**News Search Results for {ticker}**\nNo recent news available. Market sentiment may be influenced by ongoing U.S.-China trade tensions and tariff concerns."
    return f"**News Search Results for {ticker}**\n{news['content']}\n**URLs**: {', '.join(news['urls'])}"

def stock_price_tool(ticker: str) -> str:
    """Synchronous wrapper for stock price using cached data."""
    logger.debug(f"Starting stock_price_tool for {ticker}, current cache: {_cache['stock_price'].get(ticker)}")
    if ticker not in _cache['stock_price']:
        logger.info(f"Ticker {ticker} not in cache, preloading now")
        asyncio.run(preload_data(ticker, retries=2))
    logger.debug(f"Cache in stock_price_tool for {ticker} after preload: {_cache['stock_price'].get(ticker)}")
    stock_data = _cache['stock_price'].get(ticker, {'error': f"No stock data for {ticker}"})
    if 'error' in stock_data:
        logger.error(f"Stock price retrieval failed for {ticker}: {stock_data['error']}")
        return f"Error fetching stock price: {stock_data['error']}"
    return (
        f"**Stock Price Data for {ticker}**\n"
        f"- Current Price: ${stock_data['current_price']:.2f}\n"
        f"- Previous Close: ${stock_data['previous_close']:.2f}\n"
        f"- Change: {stock_data['change_percent']:.2f}%\n"
        f"- Open: ${stock_data['open_price']:.2f}\n"
        f"- High: ${stock_data['high_price']:.2f}\n"
        f"- Low: ${stock_data['low_price']:.2f}\n"
        f"- Volume: {stock_data['volume']:,}"
    )

def technical_indicators_tool(ticker: str) -> str:
    """Synchronous wrapper for technical indicators using cached data."""
    logger.debug(f"Starting technical_indicators_tool for {ticker}, current cache: {_cache['technical_indicators'].get(ticker)}")
    if ticker not in _cache['technical_indicators']:
        logger.info(f"Ticker {ticker} not in cache, preloading now")
        asyncio.run(preload_data(ticker, retries=2))
    logger.debug(f"Cache in technical_indicators_tool for {ticker} after preload: {_cache['technical_indicators'].get(ticker)}")
    indicators = _cache['technical_indicators'].get(ticker, {'error': f"No indicators for {ticker}"})
    if 'error' in indicators:
        logger.error(f"Technical indicators retrieval failed for {ticker}: {indicators['error']}")
        return f"Error fetching technical indicators: {indicators['error']}"
    return (
        f"**Technical Indicators for {ticker}**\n"
        f"- RSI: {indicators['rsi']:.2f}\n"
        f"- MACD: {indicators['macd']:.2f}\n"
        f"- MACD Signal: {indicators['macd_signal']:.2f}\n"
        f"- MACD Histogram: {indicators['macd_histogram']:.2f}\n"
        f"- Stochastic %K: {indicators['stochastic_k']:.2f}\n"
        f"- Stochastic %D: {indicators['stochastic_d']:.2f}\n"
        f"- SMA 9: ${indicators['sma_9']:.2f}\n"
        f"- SMA 20: ${indicators['sma_20']:.2f}\n"
        f"- Bollinger Upper: ${indicators['bb_upper']:.2f}\n"
        f"- Bollinger Middle: ${indicators['bb_middle']:.2f}\n"
        f"- Bollinger Lower: ${indicators['bb_lower']:.2f}"
    )

def price_prediction_tool(ticker: str) -> str:
    """Synchronous wrapper for price prediction using cached data with fallback."""
    logger.debug(f"Starting price_prediction_tool for {ticker}, current cache: {_cache['predictions'].get(ticker)}")
    if ticker not in _cache['predictions']:
        logger.info(f"Ticker {ticker} not in cache, preloading now")
        asyncio.run(preload_data(ticker, retries=2))
    logger.debug(f"Cache in price_prediction_tool for {ticker} after preload: {_cache['predictions'].get(ticker)}")
    prediction = _cache['predictions'].get(ticker, {'error': f"No prediction for {ticker}"})
    if 'error' in prediction or not prediction.get('prediction'):
        logger.warning(f"Price prediction failed for {ticker}, using rule-based fallback")
        stock_data = _cache['stock_price'].get(ticker, {})
        current_price = stock_data.get('current_price', 200.85)
        rsi = _cache['technical_indicators'].get(ticker, {}).get('rsi', 45.35)
        if rsi < 30:
            trend = "upward (oversold, potential reversal)"
            predicted_price = current_price * 1.05
        elif rsi > 70:
            trend = "downward (overbought, potential correction)"
            predicted_price = current_price * 0.95
        else:
            trend = "sideways (neutral momentum)"
            predicted_price = current_price * 1.01
        return f"**Price Prediction for {ticker} (Rule-Based Fallback)**\n- Predicted Trend: {trend}\n- Estimated Price: ${predicted_price:.2f}"
    return f"**Price Prediction for {ticker}**\n{prediction['prediction']}"

# Define Agents with SQLite storage for UI persistence
agent_storage = "tmp/agents.db"

web_search_agent = Agent(
    name="WebSearchAgent",
    model=OpenAIChat(
        id='accounts/fireworks/models/llama-v3p1-70b-instruct',
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=FIREWORKS_API_KEY,
    ),
    description="Web search agent for financial news",
    instructions=dedent("""\
        You are a financial news expert. Your job is to retrieve recent news articles from CNBC about a specified stock ticker using the web_search_tool.
       
        1. Use the web_search_tool with the ticker to fetch relevant news.
        2. If the tool fails or returns no data, acknowledge the failure and provide a brief market context based on general trends (e.g., trade tensions, tech sector performance).
        3. Summarize the news articles concisely, including article titles and URLs.
        4. Format the output in markdown for clarity.
    """),
    tools=[web_search_tool],
    storage=SqliteStorage(table_name="web_search_agent", db_file=agent_storage),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
)

stock_data_agent = Agent(
    name="StockDataAgent",
    model=OpenAIChat(
        id='accounts/fireworks/models/llama-v3p1-70b-instruct',
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=FIREWORKS_API_KEY,
    ),
    description="Stock data retrieval agent",
    instructions=dedent("""\
        You are a stock data analyst. Your job is to retrieve current stock price and technical indicators for a specified ticker using the provided tools.
       
        1. Use stock_price_tool to fetch current price, previous close, change percent, open, high, low, and volume.
        2. Use technical_indicators_tool to fetch RSI, MACD, Stochastic, SMA, and Bollinger Bands.
        3. Format the output in markdown with clear sections for price data and indicators.
    """),
    tools=[stock_price_tool, technical_indicators_tool],
    storage=SqliteStorage(table_name="stock_data_agent", db_file=agent_storage),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
)

prediction_agent = Agent(
    name="PredictionAgent",
    model=OpenAIChat(
        id='accounts/fireworks/models/llama-v3p1-70b-instruct',
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=FIREWORKS_API_KEY,
    ),
    description="Stock price prediction agent",
    instructions=dedent("""\
        You are a stock price prediction expert. Your job is to generate AI-driven price predictions for a specified ticker.
       
        1. Use price_prediction_tool to fetch the prediction results.
        2. If the tool fails, acknowledge the failure and indicate that a fallback rule-based prediction was used.
        3. Return the prediction results in markdown format.
    """),
    tools=[price_prediction_tool],
    storage=SqliteStorage(table_name="prediction_agent", db_file=agent_storage),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
)

# Define the Financial Analysis Workflow
class FinancialAnalysisWorkflow(Workflow):
    """Workflow for comprehensive financial analysis of a stock ticker, consolidating outputs from multiple agents."""

    description: str = "A workflow that orchestrates agents to provide a detailed financial analysis with a recommendation."

    # Add agents as attributes
    web_search_agent = web_search_agent
    stock_data_agent = stock_data_agent
    prediction_agent = prediction_agent

    def run(self, ticker: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(f"Starting financial analysis workflow for ticker: {ticker}")

        # Check if the full analysis is cached
        if use_cache:
            cached_analysis = self.session_state.get("analyses", {}).get(ticker)
            if cached_analysis:
                logger.info(f"Cache hit for ticker {ticker}")
                yield RunResponse(run_id=self.run_id, content=cached_analysis)
                return

        # Step 1: Run WebSearchAgent to get news
        logger.info(f"Fetching news for {ticker}")
        news_response = self.web_search_agent.run(ticker)
        if not news_response or not news_response.content:
            logger.error(f"Failed to fetch news for {ticker}")
            yield RunResponse(run_id=self.run_id, content=f"**Error**: Failed to fetch news for {ticker}.")
            return
        news_content = news_response.content

        # Step 2: Run StockDataAgent to get stock price and technical indicators
        logger.info(f"Fetching stock data for {ticker}")
        stock_data_response = self.stock_data_agent.run(ticker)
        if not stock_data_response or not stock_data_response.content:
            logger.error(f"Failed to fetch stock data for {ticker}")
            yield RunResponse(run_id=self.run_id, content=f"**Error**: Failed to fetch stock data for {ticker}.")
            return
        stock_data_content = stock_data_response.content

        # Step 3: Run PredictionAgent to get price prediction
        logger.info(f"Fetching price prediction for {ticker}")
        prediction_response = self.prediction_agent.run(ticker)
        if not prediction_response or not prediction_response.content:
            logger.error(f"Failed to fetch price prediction for {ticker}")
            yield RunResponse(run_id=self.run_id, content=f"**Error**: Failed to fetch price prediction for {ticker}.")
            return
        prediction_content = prediction_response.content

        # Step 4: Consolidate outputs and generate recommendation
        logger.info(f"Consolidating analysis for {ticker}")

        # Separate price data and technical indicators
        stock_data_lines = stock_data_content.split('\n')
        price_data_lines = [line for line in stock_data_lines if not line.startswith('-') or "Stock Price Data" in line]
        technical_indicators_lines = [line for line in stock_data_lines if line.startswith('-')]

        # Extract current price and change
        current_price = None
        change_percent = None
        for line in stock_data_lines:
            if "Current Price:" in line:
                current_price = float(line.split('$')[1].split()[0])
            if "Change:" in line:
                change_part = line.split("Change:")[1].strip()
                change_percent = float(change_part.replace('%', ''))

        # Extract technical indicators
        rsi = None
        macd = None
        for line in technical_indicators_lines:
            if "RSI:" in line:
                rsi_part = line.split("RSI:")[1].strip()
                rsi = float(rsi_part)
            if "MACD:" in line and "Signal" not in line and "Histogram" not in line:
                macd_part = line.split("MACD:")[1].strip()
                macd = float(macd_part)

        # Extract predicted trend and price
        predicted_trend = "sideways"
        predicted_price = current_price if current_price else 0
        for line in prediction_content.split('\n'):
            if "Predicted Trend:" in line:
                predicted_trend = line.split(":", 1)[1].strip()
            if "Estimated Price:" in line:
                predicted_price = float(line.split('$')[1].split()[0])

        # Determine news sentiment
        news_sentiment = "neutral"
        if "positive" in news_content.lower():
            news_sentiment = "positive"
        elif "negative" in news_content.lower():
            news_sentiment = "negative"

        # Generate recommendation and reasoning
        recommendation = "hold"
        confidence = 50
        reasoning_points = []

        # Price movement analysis
        if change_percent is not None:
            if change_percent > 0:
                reasoning_points.append(f"The stock is up {change_percent:.2f}%, indicating positive momentum.")
            else:
                reasoning_points.append(f"The stock is down {change_percent:.2f}%, suggesting potential caution.")

        # RSI analysis
        if rsi is not None:
            if rsi < 30:
                reasoning_points.append("The RSI at {:.2f} indicates an oversold condition, suggesting a potential upward reversal.".format(rsi))
                recommendation = "buy"
                confidence += 20
            elif rsi > 70:
                reasoning_points.append("The RSI at {:.2f} indicates an overbought condition, suggesting a potential correction.".format(rsi))
                recommendation = "sell"
                confidence += 20
            else:
                reasoning_points.append("The RSI at {:.2f} is neutral, indicating no strong overbought or oversold signals.".format(rsi))

        # MACD analysis
        if macd is not None:
            if macd > 0:
                reasoning_points.append("The MACD at {:.2f} shows bullish momentum.".format(macd))
                if recommendation == "buy":
                    confidence += 10
                elif recommendation != "sell":
                    recommendation = "buy"
                    confidence += 10
            else:
                reasoning_points.append("The MACD at {:.2f} shows bearish momentum.".format(macd))
                if recommendation == "sell":
                    confidence += 10
                elif recommendation != "buy":
                    recommendation = "sell"
                    confidence += 10

        # Prediction analysis
        if predicted_price and current_price:
            price_change = ((predicted_price - current_price) / current_price) * 100
            reasoning_points.append(f"The AI model predicts a {predicted_trend} trend with a target price of ${predicted_price:.2f}, a {price_change:.2f}% change.")
            if "upward" in predicted_trend:
                if recommendation == "buy":
                    confidence += 20
                elif recommendation != "sell":
                    recommendation = "buy"
                    confidence += 20
            elif "downward" in predicted_trend:
                if recommendation == "sell":
                    confidence += 20
                elif recommendation != "buy":
                    recommendation = "sell"
                    confidence += 20

        # News sentiment analysis
        reasoning_points.append(f"Recent news reflects {news_sentiment} sentiment.")
        if news_sentiment == "positive":
            if recommendation == "buy":
                confidence += 10
            elif recommendation != "sell":
                recommendation = "buy"
                confidence += 10
        elif news_sentiment == "negative":
            if recommendation == "sell":
                confidence += 10
            elif recommendation != "buy":
                recommendation = "sell"
                confidence += 10

        # Cap confidence at 90%
        confidence = min(confidence, 90)

        # Format the final analysis
        final_analysis = f"## Financial Analysis for {ticker}\n\n"
        final_analysis += f"**Current Price Analysis**\n{chr(10).join(price_data_lines)}\n\n"
        final_analysis += f"**Technical Indicators**\n{chr(10).join(technical_indicators_lines)}\n\n"
        final_analysis += f"**Price Prediction**\n{prediction_content}\n\n"
        final_analysis += f"**News Summary**\n{news_content}\n\n"
        final_analysis += f"**Recommendation and Reasoning**\n"
        final_analysis += f"We recommend **{recommendation}** with a confidence level of {confidence}%.\n"
        for point in reasoning_points:
            final_analysis += f"- {point}\n"

        # Cache the final analysis
        self.session_state.setdefault("analyses", {})
        self.session_state["analyses"][ticker] = final_analysis

        # Yield the final response
        yield RunResponse(run_id=self.run_id, content=final_analysis)

# Define a tool to run the workflow
def run_financial_analysis(ticker: str) -> str:
    """Run the financial analysis workflow for the given ticker."""
    workflow = FinancialAnalysisWorkflow(
        session_id=f"financial-analysis-{ticker}",
        storage=SqliteStorage(table_name="financial_analysis_workflows", db_file=agent_storage),
        debug_mode=True
    )
    response = workflow.run(ticker=ticker, use_cache=True)
    for resp in response:
        return resp.content
    return f"**Error**: Failed to generate financial analysis for {ticker}."

# Define a new Agent to wrap the workflow for UI compatibility
financial_analysis_agent = Agent(
    name="FinancialAnalysisAgent",
    model=OpenAIChat(
        id='accounts/fireworks/models/llama-v3p1-70b-instruct',
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=FIREWORKS_API_KEY,
    ),
    description="Agent for comprehensive financial analysis of a stock ticker using a workflow",
    instructions=dedent("""\
        You are a financial analysis agent that uses a workflow to provide a detailed analysis.
       
        1. Extract the stock ticker from the user's request (e.g., "Perform a detailed financial analysis for AAPL stock" should extract "AAPL"). The ticker is typically a 1-5 character uppercase string (e.g., AAPL, MSFT).
        2. Use the run_financial_analysis tool to execute the FinancialAnalysisWorkflow with the extracted ticker.
        3. Return the workflow's response directly to the user in markdown format.
        4. If the ticker cannot be extracted, return an error message: "**Error**: Please specify a valid stock ticker (e.g., AAPL) in your request."
    """),
    tools=[run_financial_analysis],
    storage=SqliteStorage(table_name="financial_analysis_agent", db_file=agent_storage),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
)

# Create Playground app with the agents
app = Playground(agents=[web_search_agent, stock_data_agent, prediction_agent, financial_analysis_agent]).get_app()

async def main():
    parser = argparse.ArgumentParser(description="Run Agno Playground or financial analysis workflow")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--run-workflow", action="store_true", help="Run workflow directly instead of Playground")
    args = parser.parse_args()

    logger.debug(f"Preloading data for {args.ticker}")
    await preload_data(args.ticker, retries=2)
    logger.debug(f"Cache after preload: {_cache}")

    if args.run_workflow:
        print(f"Running financial analysis workflow for {args.ticker}...")
        workflow = FinancialAnalysisWorkflow(
            session_id=f"financial-analysis-{args.ticker}",
            storage=SqliteStorage(table_name="financial_analysis_workflows", db_file=agent_storage),
            debug_mode=True
        )
        response = workflow.run(ticker=args.ticker, use_cache=True)
        pprint_run_response(response, markdown=True, show_time=True)
    else:
        print("Starting Agno Playground server...")
        serve_playground_app("main:app", reload=True)

if __name__ == "__main__":
    print("Initializing financial analysis agents...")
    asyncio.run(main())
    print("Playground server running. Access at http://localhost:8000")