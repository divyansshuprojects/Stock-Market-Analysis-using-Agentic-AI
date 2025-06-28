# Financial Stock Analysis Platform

A comprehensive platform for stock market analysis that combines real-time stock data, technical indicators, machine learning predictions, and latest news to provide investment recommendations.

## Overview

This system integrates several components:
- Real-time stock data retrieval from Financial Modeling Prep API
- Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
- Machine learning predictions using MLflow models
- Latest news aggregation from CNBC
- AI-driven analysis and recommendations using LLaMa v3 model

The platform uses a microservices architecture with Multi-Call Protocol (MCP) servers to handle different aspects of data retrieval and processing.

## Theory and Architecture

### Multi-Call Protocol (MCP)
The system uses MCP servers to separate concerns:
- `stock_data_mcp.py`: Retrieves stock price data and calculates technical indicators
- `ml_prediction_mcp.py`: Handles communication with the ML prediction API
- `web_search_mcp.py`: Searches for recent news articles

### Machine Learning Pipeline
The system uses MLflow for model management:
- Train models using historical stock data
- Register trained models in MLflow
- Serve models through FastAPI wrapper
- Make predictions based on current technical indicators

### LLM Integration
The platform uses Fireworks LLaMa v3 through the OpenAI-compatible API to:
- Interpret technical indicators
- Analyze ML model predictions
- Summarize recent news
- Generate final investment recommendations

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- MLflow
- FastAPI
- Minio (for model storage)
- Tavily API key (for news search)
- Financial Modeling Prep API key

## Installation and Setup

### Step 1: Clone the Repository

### Step 2: Set Environment Variables
Create a `.env` file with the following:
```
FMP_API_KEY=your_financial_modeling_prep_api_key
TAVILY_API_KEY=your_tavily_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
```

### Step 3: Start MLflow and Minio with Docker
Navigate to the MLflow development environment:
```bash
cd mlflow-dev-env
docker-compose up -d
```

This starts:
- MLflow server on port 5001
- Minio server on port 9000 (with credentials minioadmin/minioadmin)

### Step 4: Train and Register the ML Model
```bash
cd train
python train_ml_model.py
```

This script:
- Loads historical stock data
- Trains a machine learning model
- Registers the model in MLflow as "sample_model_ml" version "1"

### Step 5: Start the FastAPI Server
```bash
python wrap_fastapi.py
```

This starts the FastAPI server on port 8000, which:
- Loads the trained model from MLflow
- Provides HTTP endpoints for making predictions

## Usage

### Step 1: Run the Main Application
```bash
python main.py [TICKER]
```

Replace `[TICKER]` with the stock symbol you want to analyze (defaults to MSFT if not provided).

### Step 2: Interpret the Results
The system will:
1. Retrieve current stock price data and technical indicators
2. Generate machine learning predictions based on technical indicators
3. Search for recent news about the stock
4. Analyze all data and provide an investment recommendation

## Example Output

```
Starting financial analysis...
Based on analysis of the current stock price and basic information, technical indicators, AI-driven predictions from the MLflow model, and recent news about AAPL stock, here is a comprehensive financial analysis:

**Current Price Analysis:**
The current stock price of AAPL is $192.35, which is a decrease of 2.58445% from the previous close. The open price was $197.15, and the high and low prices were $193.71 and $191.38, respectively. The volume was 8216524.

**Technical Indicator Interpretation:**
The technical indicators suggest that AAPL is currently in a bearish trend. The RSI (Relative Strength Index) is at 41.57646823584666, which indicates an oversold condition. The MACD (Moving Average Convergence Divergence) is at -0.33039232237975, which indicates a negative signal. The Stochastic oscillator is at 30.670399558494456, which indicates a neutral signal...

**AI Model Prediction:**
The AI model predicts a negative return of -0.017 (or -1.7%) for AAPL in the short term.

**Recent News:**
"Apple stock jumps on Trump's tariff exemption: How Jim Cramer says to play it" (https://www.cnbc.com/2025/04/15/apple-stock-jumps-on-trumps-tariff-exemption-how-cramer-says-to-play-it.html)
"Apple's 3-day loss in market cap swells to almost $640 billion" (https://www.cnbc.com/2025/04/07/apples-3-day-loss-in-market-cap-swells-to-almost-640-billion.html)
"Apple has worst day since August following reports of weak iPhone sales" (https://www.cnbc.com/2025/04/16/apple-has-worst-day-since-august-poor-iphones-spur-dec-downgrade.html)

**Buy/Sell/Hold Recommendation with Confidence Level and Reasoning:**
Based on the analysis, I recommend selling AAPL with a confidence level of 60%. The reason for this recommendation is that the technical indicators suggest a bearish trend, and the AI model predicts a negative return. Additionally, the recent news articles suggest that AAPL is facing challenges, including weaker iPhone sales and tariffs. However, it's important to note that the stock market can be unpredictable, and there is always a risk of losses when selling or buying.

Analysis complete!
```

## File Descriptions

- `main.py`: Main application that orchestrates all components
- `stock_data_mcp.py`: MCP server for retrieving stock data and technical indicators
- `ml_prediction_mcp.py`: MCP server for making ML model predictions
- `web_search_mcp.py`: MCP server for searching recent news
- `wrap_fastapi.py`: FastAPI wrapper for the MLflow model

## Troubleshooting

### Common Issues:
1. **API Connection Errors**: Ensure your API keys are valid and correctly set in the .env file
2. **MLflow Connection**: Check that MLflow server is running and accessible at http://127.0.0.1:5001
3. **Model Not Found**: Ensure you've completed the model training step and the model is registered in MLflow
4. **Minio Connection**: Verify Minio is running and credentials are correct

## Advanced Configuration

### Customizing the Model
To use a different ML model:
1. Modify `train_ml_model.py` with your desired algorithm
2. Update the model name and version in `wrap_fastapi.py`

### Adding More Data Sources
Additional MCP servers can be created for:
- Social media sentiment analysis
- Economic indicator data
- Industry-specific metrics

