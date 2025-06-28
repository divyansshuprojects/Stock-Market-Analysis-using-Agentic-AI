import os
import asyncio
import sys
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
 
os.environ["FIREWORKS_API_KEY"] = ""
# Load environment variables
load_dotenv(override=True)
 
# Configure model
stock_model = OpenAIModel(
    'accounts/fireworks/models/llama-v3p1-70b-instruct',
    provider=OpenAIProvider(
        base_url='https://api.fireworks.ai/inference/v1',
        api_key=os.environ["FIREWORKS_API_KEY"],
    ),
)
 
# Define the MCP Servers
web_search_server = MCPServerStdio(
    'python',
    ['web_search_mcp.py']
)
 
# Consolidated ML prediction server (includes stock data + ML prediction)
model_prediction_server = MCPServerStdio(
    'python',
    ['ml_prediction_mcp.py']
)
 
stock_data_server = MCPServerStdio(
    'python',
    ['stock_data_mcp.py']
)
 
# Define the Agent with all MCP servers
agent = Agent(
    stock_model,
    mcp_servers=[web_search_server, model_prediction_server, stock_data_server],
    retries=3
)
 
# Main async function
async def main():
    # Parse command line arguments for ticker symbol
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1 and not sys.argv[1].endswith(".py"):
        ticker = sys.argv[1]
   
    # Run the agent with the MCP servers
    async with agent.run_mcp_servers():
        result = await agent.run(f"""
        I want a detailed financial analysis for {ticker} stock. Please follow these steps exactly:
 
        1. First, use the get_price_prediction and get_technical_indicators to get the data and pass it to the  get_price_prediction tool to get comprehensive data for {ticker}, which will include:
           - Current stock price and basic information
           - Technical indicators
           - AI-driven predictions from the MLflow model
       
        2. Search for recent news about {ticker} stock using the search_cnbc_news tool.
       
        3. Finally, analyze all the data and provide a recommendation including:
           - Current price analysis
           - Technical indicator interpretation
           - AI model prediction results (directly from MLflow)
           - Recent news summary with article links
           - Buy/sell/hold recommendation with confidence level and reasoning
       
        Make sure to include the AI model's prediction and expected return value in your analysis.
        Present everything in clear, professional English.
        """)
       
        print(result)
 
if __name__ == "__main__":
    print("Starting financial analysis...")
    asyncio.run(main())
    print("Analysis complete!")
