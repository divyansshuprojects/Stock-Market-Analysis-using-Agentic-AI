# Financial Analysis System

A comprehensive stock analysis tool that combines technical analysis, machine learning predictions, and market sentiment analysis to provide holistic investment recommendations.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Theory and Methodology](#theory-and-methodology)
- [Features](#features)
- [Installation](#installation)
- [Setting Up MLflow Server](#setting-up-mlflow-server)
- [Training the ML Model](#training-the-ml-model)
- [Running the Financial Analysis](#running-the-financial-analysis)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This Financial Analysis System leverages multiple AI agents to provide comprehensive investment recommendations by analyzing stocks from three distinct perspectives:

1. **Price Analysis**: Technical indicators and machine learning predictions for price forecasting
2. **Market Sentiment**: News and social media sentiment analysis for market perception
3. **Consolidated Analysis**: Integration of technical and sentiment data for final recommendations

The system uses advanced techniques including technical analysis, natural language processing, and machine learning to provide actionable trading insights.

## System Architecture

The system employs a multi-agent architecture:

1. **Price Analysis Agent**: Processes stock price data and technical indicators to generate trading signals
2. **Market News Sentiment Agent**: Analyzes financial news articles from CNBC to evaluate market sentiment
3. **Consolidated Analysis Agent**: Combines insights from both agents to produce final recommendations

These agents communicate through a structured workflow, ensuring comprehensive analysis of each stock.

## Theory and Methodology

The system incorporates several financial analysis methodologies:

- **Technical Analysis**: Utilizes indicators like RSI, MACD, Stochastic oscillators, and Bollinger Bands to identify price patterns and momentum
- **Machine Learning Forecasting**: Employs a Random Forest model to predict 5-day price movements
- **Sentiment Analysis**: Uses NLP techniques to extract sentiment from financial news
- **Portfolio Theory**: Incorporates allocation recommendations based on risk-reward profiles

## Features

- Technical analysis using common indicators (RSI, MACD, Stochastic, Bollinger Bands)
- Machine learning predictions for 5-day returns
- Market news sentiment analysis from CNBC
- Consolidated investment recommendations with confidence scores
- Actionable investment insights with specific allocation percentages
- Risk assessment and categorization

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- API keys for:
  - Fireworks AI
  - Tavily (for web search)
  - Financial Modeling Prep (FMP)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/financial-analysis-system.git
cd financial-analysis-system
```

### Step 2: Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
pydantic>=2.0.0
pydantic-ai>=0.1.0
pandas>=1.5.0
numpy>=1.23.0
httpx>=0.24.0
mlflow>=2.7.0
tavily-python>=0.1.4
python-dotenv>=1.0.0
ta-lib>=0.4.26
```

### Step 3: Installing TA-Lib

TA-Lib can be challenging to install. Follow these platform-specific instructions:

**On Ubuntu/Debian:**
```bash
apt-get install build-essential
apt-get install libta-lib-dev
pip install ta-lib
```

**On macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**On Windows:**
Download and install the pre-built binary from [here](https://github.com/mrjbq7/ta-lib/releases), then:
```bash
pip install ta-lib
```

### Step 4: Set Up Environment Variables

Create a `.env` file with your API keys:

```
FIREWORKS_API_KEY=your_fireworks_api_key
TAVILY_API_KEY=your_tavily_api_key
FMP_API_KEY=your_fmp_api_key
```

Or set them directly in your environment:

```bash
export FIREWORKS_API_KEY="your_fireworks_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export FMP_API_KEY="your_fmp_api_key"
```

## Setting Up MLflow Server

The system requires an MLflow server with MinIO for artifact storage. We use Docker for easy setup.

### Step 1: Start Docker Services

Navigate to the MLflow development environment directory and start the Docker containers:

```bash
cd mlflow-dev-env
docker-compose up -d
```

This starts:
- MLflow tracking server (available at http://127.0.0.1:5001)
- MinIO object storage server
- PostgreSQL database (for experiment tracking)

### Step 2: Verify Services are Running

Check that all containers are running:

```bash
docker ps
```

You should see containers for MLflow, MinIO, and PostgreSQL.

### Step 3: Access the MLflow UI

Open your browser and navigate to:

```
http://127.0.0.1:5001
```

You should see the MLflow UI. Initially, there will be no experiments or models.

## Training the ML Model

### Step 1: Navigate to Train Directory

```bash
cd mlflow-dev-env/train
```

### Step 2: Execute the Training Script

```bash
python train_ml_model.py
```

This script:
1. Loads historical price data
2. Calculates technical indicators
3. Trains a Random Forest model to predict 5-day returns
4. Registers the model in MLflow as "sample_model_ml" (version 1)
5. Stores model artifacts in MinIO

### Step 3: Verify Model Registration

In the MLflow UI (http://127.0.0.1:5001):
1. Navigate to the "Models" section
2. Check that "sample_model_ml" is registered
3. Verify it has a version "1"

## Running the Financial Analysis

### Step 1: Execute the Analysis Tool

Run the main analysis script with a stock ticker:

```bash
python tool_enhanced.py --ticker MSFT
```

For programmatic use:

```python
from financial_analysis import run_financial_analysis_system_sync

# Analyze Microsoft stock
result = run_financial_analysis_system_sync("MSFT")
print(result)
```

### Step 2: Review the Analysis

The system will output a comprehensive analysis in JSON format with three sections:
- Price Analysis
- Sentiment Analysis
- Consolidated Analysis

## Understanding the Output

The output includes:

### Price Analysis
- Direction: Buy, Sell, or Hold recommendation
- Confidence: Level from 1-10
- Target Price: Price forecast
- Recommended Allocation: Portfolio percentage
- ML Prediction: Expected 5-day return

### Sentiment Analysis
- Overall Market Sentiment: Positive, Negative, or Neutral
- Fear Index: Sentiment score from 1-100
- Stock Mentions: Specific context and sentiment of stock mentions
- Market Trends: Identified trends affecting the stock
- Insights: Investment recommendations based on sentiment

### Consolidated Analysis
- Final Recommendation: Buy, Sell, or Hold
- Confidence Level: Combined confidence from both analyses
- Exposure Percentage: Recommended allocation
- Risk Level: Low, Medium, or High
- Action Steps: Specific actions to take (entry points, stop-loss levels)

## Troubleshooting

### Docker Issues
- If containers fail to start: `docker-compose down && docker-compose up -d`
- Check logs: `docker logs mlflow-server`
- Ensure ports 5001 (MLflow) and 9000 (MinIO) are available

### MLflow Connection Issues
- Verify tracking URI is correct (http://127.0.0.1:5001)
- Check if model is registered correctly in the MLflow UI
- Ensure model name matches exactly: "sample_model_ml"

### API Errors
- Verify API keys are correctly set
- Check for rate limiting on Financial Modeling Prep API
- Ensure correct capitalization in stock tickers

### TA-Lib Issues
- If installation fails, try using pre-built binaries or Docker
- For Windows users, ensure you have Visual C++ build tools installed

## License
L
