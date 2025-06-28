# Financial Analysis System

## Overview

The Financial Analysis System is a comprehensive financial analysis tool that combines technical analysis, sentiment analysis, and machine learning to provide investment recommendations for stocks. The system leverages multiple agents to process different aspects of financial data:

1. **Price Analysis Agent**: Analyzes stock prices and technical indicators to generate trading recommendations
2. **Market News Sentiment Agent**: Analyzes news articles to extract market sentiment
3. **Consolidated Analysis Agent**: Combines inputs from the other agents to provide holistic investment recommendations

## Features

- Technical analysis using multiple indicators (RSI, MACD, Stochastic, Bollinger Bands)
- Machine learning prediction for expected returns
- News sentiment analysis from financial news sources
- Consolidated analysis with actionable investment recommendations
- Support for any publicly traded stock ticker

## Requirements

- Python 3.8+
- Docker and Docker Compose
- Access to the following APIs:
  - [Fireworks AI](https://fireworks.ai/) (LLM provider)
  - [Tavily](https://tavily.com/) (Search API)
  - [Financial Modeling Prep](https://financialmodelingprep.com/) (Financial data)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-analysis-system.git
   cd financial-analysis-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install pydantic httpx pandas numpy talib-binary mlflow pydantic-ai tavily-python dataclasses
   ```

4. Install additional TA-Lib requirements (may vary by platform):
   
   **For macOS:**
   ```bash
   brew install ta-lib
   ```
   
   **For Windows:**
   Download prebuilt wheels from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install with pip:
   ```bash
   pip install path/to/downloaded/wheel.whl
   ```
   
   **For Linux:**
   ```bash
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   cd ..
   pip install ta-lib
   ```

## Configuration

1. Set up your API keys:

   You can either replace them directly in the code or set them as environment variables:
   
   ```python
   # In your terminal or .env file
   export FIREWORKS_API_KEY="your_fireworks_api_key"
   export TAVILY_API_KEY="your_tavily_api_key"
   export FMP_API_KEY="your_fmp_api_key"
   ```

2. Start the MLflow environment using Docker:

   ```bash
   cd mlflow-dev-env
   docker-compose up -d
   ```

   This will start all required services including MLflow, MinIO, and any other dependencies in Docker containers. No manual port configuration is needed as everything is set up in the Docker Compose file.

   After starting the containers, make sure that you have a model registered in MLflow with the name "sample_model_ml" and version "1" that accepts the technical indicators as input and outputs price predictions. You can access the MLflow UI in your browser to verify this.

## Usage

### Running the Analysis

To run the system for a specific stock ticker:

```python
from financial_analysis import run_financial_analysis_system_sync

# Analyze Microsoft stock
result = run_financial_analysis_system_sync("MSFT")
print(result)
```

### Command Line Usage

You can run the script directly from the command line:

```bash
python financial_analysis.py
```

By default, it will analyze MSFT, but you can modify the ticker in the main block:

```python
if __name__ == "__main__":
    result = run_financial_analysis_system_sync("AAPL")  # Change to your preferred ticker
    print(json.dumps(result, indent=2))
```

## Output Structure

The system generates a JSON output with the following structure:

```json
{
  "ticker": "MSFT",
  "price_analysis": {
    "direction": "BUY/SELL/HOLD",
    "confidence": 1-10,
    "target_price": 123.45,
    "recommended_allocation": 0-100,
    "reasoning": "Reasoning text...",
    "ml_prediction_return": 0.05
  },
  "sentiment_analysis": {
    "overall_market_sentiment": "positive/negative/neutral",
    "fear_index": 1-100,
    "optimism_score": 1-100,
    "key_stocks_mentioned": [...],
    "market_trends": [...],
    "summary": "Summary text...",
    "insights": "Insights text...",
    "source_urls": [...]
  },
  "consolidated_analysis": {
    "ticker": "MSFT",
    "direction": "BUY/SELL/HOLD",
    "confidence": 1-10,
    "exposure_percentage": 0-100,
    "price_analysis_summary": "Summary text...",
    "sentiment_analysis_summary": "Summary text...",
    "expected_return_5d": 0.05,
    "risk_level": "LOW/MEDIUM/HIGH",
    "reasoning": "Reasoning text...",
    "action_steps": [...]
  }
}
```

## Troubleshooting

### Docker and MLflow Issues

If you encounter issues with the MLflow environment:

1. Check if all Docker containers are running: `docker ps`
2. View container logs if needed: `docker logs <container_name>`
3. Ensure your model is registered correctly in MLflow UI (typically available at http://localhost:5000)
4. If you need to restart the environment: `docker-compose down && docker-compose up -d`

### API Errors

If you encounter API errors:

1. Verify your API keys are correct
2. Check API rate limits
3. Confirm your internet connection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Pydantic AI, Fireworks LLM, and TA-Lib
- Uses Financial Modeling Prep API for financial data
- Uses Tavily API for news search capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
