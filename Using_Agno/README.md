# Financial Analysis System

A comprehensive multi-agent financial analysis system built with the Agno framework that provides detailed stock analysis, technical indicators, news sentiment, and AI-driven price predictions.

## Overview

This system orchestrates multiple specialized agents to deliver comprehensive financial analysis:

- **Web Search Agent**: Retrieves recent financial news from CNBC
- **Stock Data Agent**: Fetches real-time stock prices and technical indicators
- **Prediction Agent**: Generates AI-driven price predictions using machine learning
- **Financial Analysis Agent**: Consolidates all data into actionable investment recommendations

## Features

- 🔍 **Real-time Stock Data**: Current prices, volume, daily changes
- 📊 **Technical Analysis**: RSI, MACD, Bollinger Bands, SMA, Stochastic indicators
- 📰 **News Sentiment Analysis**: Latest financial news with sentiment evaluation
- 🤖 **AI Price Predictions**: Machine learning-based price forecasting
- 💡 **Investment Recommendations**: Buy/Hold/Sell recommendations with confidence scores
- 🎯 **Interactive Playground**: Web-based UI for easy interaction
- 💾 **Intelligent Caching**: Persistent data storage with file-based caching
- 🔄 **Retry Logic**: Robust error handling with automatic retries

## Architecture

### Agents

1. **WebSearchAgent**
   - Retrieves financial news using CNBC search
   - Provides market sentiment analysis
   - Fallback to general market context if news unavailable

2. **StockDataAgent**
   - Fetches real-time stock prices from FMP API
   - Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Formats data for analysis

3. **PredictionAgent**
   - Uses ML models for price prediction
   - Fallback to rule-based predictions using RSI signals
   - Provides trend analysis and target prices

4. **FinancialAnalysisAgent**
   - Orchestrates all other agents through a workflow
   - Generates comprehensive analysis reports
   - Provides buy/hold/sell recommendations with reasoning

### Workflow System

The `FinancialAnalysisWorkflow` coordinates agent interactions:

1. **Data Collection**: Parallel execution of news, price, and indicator retrieval
2. **Analysis Integration**: Combines all data sources
3. **Recommendation Engine**: Generates actionable insights based on:
   - Technical indicator signals
   - Price momentum analysis
   - News sentiment
   - AI prediction confidence
4. **Caching**: Stores results for improved performance

## Installation

### Prerequisites

- Python 3.8+
- MLflow server running on `http://127.0.0.1:5001`
- Minio server running on `http://localhost:9000` (for MLflow model storage)
- Required API keys (see Configuration section)

### Dependencies

```bash
# Core dependencies
pip install agno
pip install filelock
pip install asyncio

# MLflow prediction service dependencies
pip install mlflow
pip install fastapi
pip install uvicorn
pip install pandas
```

### Required Packages

The system uses these MCP (Model Context Protocol) tools:
- `web_search_mcp`: For CNBC news search
- `stock_data_mcp`: For stock price and technical indicators
- `ml_prediction_mcp`: For AI price predictions

### MLflow Model Setup

The system requires an MLflow model named `sample_model_ml` version `1` to be registered in your MLflow server for price predictions.

## Configuration

### API Keys Required

The system requires the following API keys (currently hardcoded in the script):

```python
AGNO_API_KEY = "your_agno_api_key"
FIREWORKS_API_KEY = "your_fireworks_api_key"  # For LLM inference
TAVILY_API_KEY = "your_tavily_api_key"        # For web search
FMP_API_KEY = "your_fmp_api_key"              # For financial data
```

### Environment Setup

The system automatically sets environment variables for MCP tools:

```bash
export AGNO_API_KEY="your_agno_api_key"
export FIREWORKS_API_KEY="your_fireworks_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export FMP_API_KEY="your_fmp_api_key"
```

### MLflow Configuration

The MLflow prediction service (`wrap_fastapi.py`) uses these settings:

```python
# MLflow and Minio configuration
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # Minio endpoint
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"                  # Minio username
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"              # Minio password
mlflow.set_tracking_uri("http://127.0.0.1:5001")               # MLflow server
```

## Usage

### Step 1: Start MLflow Prediction Service (Required)

**IMPORTANT**: Before running the main application, you must start the MLflow prediction service:

```bash
python wrap_fastapi.py
```

This starts the MLflow model prediction API on `http://0.0.0.0:8000` which provides:
- `/predict/csv`: Accept CSV files for batch predictions
- `/predict/json`: Accept JSON data for single predictions

The service requires:
- MLflow server running on `http://127.0.0.1:5001`
- Minio server running on `http://localhost:9000`
- Registered model `sample_model_ml` version `1` in MLflow

### Step 2: Run the Financial Analysis System

#### Running the Playground (Recommended)

```bash
python main.py
```

This starts an interactive web interface at `http://localhost:8000` where you can:
- Chat with individual agents
- Run comprehensive financial analysis
- View historical analysis results

**Note**: The main application will run on port 8001 if port 8000 is occupied by the MLflow service.

#### Direct Workflow Execution

```bash
python main.py AAPL --run-workflow
```

This runs the analysis workflow directly for a specific ticker and outputs results to the console.

### Supported Commands

In the Playground interface, you can use natural language commands:

- "Analyze AAPL stock"
- "Get technical indicators for MSFT"
- "What's the latest news on TSLA?"
- "Predict GOOGL stock price"
- "Perform a detailed financial analysis for NVDA"

## Data Sources

### Stock Data (Financial Modeling Prep API)
- Real-time stock prices
- Historical price data
- Volume and market cap information
- Technical indicators calculation

### News Data (CNBC via Tavily)
- Latest financial news articles
- Company-specific news
- Market sentiment indicators
- Article URLs for reference

### ML Predictions (MLflow Model Service)
- AI-driven price predictions via `wrap_fastapi.py`
- Machine learning model hosted on MLflow
- Supports both CSV and JSON input formats
- Model: `sample_model_ml` version `1`

### Technical Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **SMA**: Simple Moving Averages (9-day, 20-day)
- **Stochastic Oscillator**: %K and %D values

## Output Format

### Sample Analysis Report

```markdown
## Financial Analysis for AAPL

**Current Price Analysis**
- Current Price: $185.25
- Previous Close: $182.50
- Change: +1.51%
- Volume: 45,234,567

**Technical Indicators**
- RSI: 45.35 (Neutral)
- MACD: 0.85 (Bullish)
- Bollinger Upper: $190.45
- Bollinger Lower: $175.30

**Price Prediction**
- Predicted Trend: upward momentum
- Target Price: $192.50
- Confidence: 78%

**News Summary**
Recent positive earnings report boosts investor confidence...

**Recommendation**
We recommend **BUY** with 85% confidence.
- Strong technical momentum indicated by MACD
- RSI shows room for upward movement
- Positive news sentiment supports bullish outlook
```

## Caching System

The system implements intelligent caching to improve performance:

### Cache Structure
```json
{
  "stock_price": {"AAPL": {...}},
  "technical_indicators": {"AAPL": {...}},
  "news": {"AAPL": {...}},
  "predictions": {"AAPL": {...}}
}
```

### Cache Features
- **File-based persistence**: Data survives system restarts
- **Thread-safe**: Uses FileLock for concurrent access
- **Automatic preloading**: Data loaded before analysis
- **Fallback handling**: Graceful degradation when APIs fail

## Error Handling

### Retry Mechanism
- Automatic retries for failed API calls (up to 2 retries)
- 5-second delays between retry attempts
- Graceful fallbacks for each data source

### Fallback Strategies
- **News**: Generic market context when CNBC unavailable
- **Predictions**: Rule-based analysis using RSI when ML fails
- **Stock Data**: Error messages with guidance for resolution

## Logging

Comprehensive logging system with different levels:

```python
# Debug level logging enabled
logging.basicConfig(level=logging.DEBUG)
```

### Log Categories
- **INFO**: Major workflow steps and successful operations
- **WARNING**: Fallback activations and data quality issues
- **ERROR**: API failures and system errors
- **DEBUG**: Detailed cache operations and data flow

## Database Storage

### SQLite Integration
- **Agent Storage**: `tmp/agents.db`
- **Conversation History**: Persistent across sessions
- **Workflow State**: Session management and caching

### Tables
- `web_search_agent`: News search history
- `stock_data_agent`: Price and indicator queries
- `prediction_agent`: ML prediction requests
- `financial_analysis_agent`: Complete analysis sessions
- `financial_analysis_workflows`: Workflow execution logs

## Customization

### Adding New Tickers
The system supports any valid stock ticker:

```python
# Preload data for multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
for ticker in tickers:
    await preload_data(ticker)
```

### Modifying Recommendation Logic
Edit the recommendation engine in `FinancialAnalysisWorkflow.run()`:

```python
# Custom recommendation logic
if rsi < 25:  # More aggressive oversold threshold
    recommendation = "strong_buy"
    confidence += 30
```

### Adding New Indicators
Extend the technical indicators in `stock_data_mcp`:

```python
# Add new indicators
indicators.update({
    'roc': rate_of_change,
    'williams_r': williams_percent_r,
    'cci': commodity_channel_index
})
```

## Performance Optimization

### Caching Strategy
- **Preload Data**: Load all data before analysis starts
- **Session Persistence**: Reuse data across multiple queries
- **Intelligent Updates**: Only refresh stale data

### Concurrent Processing
- **Async Operations**: Non-blocking API calls
- **Parallel Execution**: Multiple agents run simultaneously
- **Resource Management**: Connection pooling and rate limiting

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Invalid API key for FMP
   Solution: Verify API keys in configuration section
   ```

2. **MLflow Service Not Running**
   ```
   Error: Connection refused to MLflow prediction service
   Solution: Start wrap_fastapi.py before running main.py
   ```

3. **MLflow Model Not Found**
   ```
   Error: Model sample_model_ml version 1 not found
   Solution: Register the model in MLflow server first
   ```

4. **Cache Lock Issues**
   ```
   Error: Could not acquire cache lock
   Solution: Delete cache.json.lock file and restart
   ```

5. **Missing Dependencies**
   ```
   Error: No module named 'agno'
   Solution: pip install agno filelock mlflow fastapi uvicorn pandas
   ```

6. **Port Conflicts**
   ```
   Error: Port 8000 already in use
   Solution: MLflow service runs on 8000, main app will use 8001
   ```

### Debug Mode
Enable detailed logging:

```python
workflow = FinancialAnalysisWorkflow(debug_mode=True)
```

## API Limits and Costs

### Financial Modeling Prep (FMP)
- Free tier: 250 requests/day
- Paid plans available for higher limits

### Tavily Search API
- Free tier: 1,000 searches/month
- Commercial plans for production use

### Fireworks AI
- Pay-per-token pricing
- Llama models typically cost-effective

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install agno filelock mlflow fastapi uvicorn pandas
```

### 2. Set Up Infrastructure
```bash
# Start MLflow server (in separate terminal)
mlflow server --host 127.0.0.1 --port 5001

# Start Minio server (in separate terminal)
minio server /path/to/data --address ":9000"
```

### 3. Register ML Model
```python
# Register your trained model in MLflow
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
# ... register model as "sample_model_ml" version "1"
```

### 4. Start Prediction Service
```bash
python wrap_fastapi.py
```

### 5. Run Financial Analysis System
```bash
python main.py
```

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up MLflow server and Minio
4. Register model `sample_model_ml` version `1` in MLflow
5. Start MLflow prediction service: `python wrap_fastapi.py`
6. Configure API keys in main.py
7. Run the main application: `python main.py`
8. Run tests: `python -m pytest tests/`

### Code Structure
```
main.py                  # Main application file
wrap_fastapi.py          # MLflow prediction service
├── Cache Management     # File-based caching system
├── Agent Definitions    # Individual agent configurations
├── Workflow Engine      # Multi-agent orchestration
├── Tool Wrappers        # Synchronous API interfaces
└── Playground Setup     # Web UI configuration
```

## MLflow Prediction Service (`wrap_fastapi.py`)

### Features
- **FastAPI Integration**: RESTful API for ML predictions
- **Multiple Input Formats**: Supports CSV files and JSON data
- **MLflow Integration**: Seamless model loading and versioning
- **Minio Storage**: S3-compatible storage for model artifacts

### API Endpoints

#### Root Endpoint
```bash
GET / 
# Returns: {"message": "MLflow Model Prediction API"}
```

#### CSV Predictions
```bash
POST /predict/csv
# Upload a CSV file for batch predictions
# Content-Type: multipart/form-data
```

#### JSON Predictions
```bash
POST /predict/json
# Content-Type: application/json
# Body: {"feature1": [value1], "feature2": [value2], ...}
```

### Model Configuration
- **Model Name**: `sample_model_ml`
- **Model Version**: `1`
- **Storage**: Minio (S3-compatible)
- **Framework**: MLflow PyFunc

## License

This project is built on the Agno framework. Please refer to the Agno license for usage terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Verify API key configurations
4. Ensure all dependencies are installed

## Roadmap

### Planned Features
- [ ] Portfolio analysis across multiple stocks
- [ ] Real-time streaming data updates
- [ ] Custom alert system for price targets
- [ ] Integration with additional news sources
- [ ] Advanced ML models for prediction
- [ ] Mobile-responsive UI improvements
- [ ] Export functionality for reports
- [ ] Backtesting capabilities

### Technical Improvements
- [ ] Redis caching for better performance
- [ ] Kubernetes deployment configurations
- [ ] API rate limiting and queuing
- [ ] Enhanced error recovery mechanisms
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline setup