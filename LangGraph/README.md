# Financial Analysis Multi-Agent System

A sophisticated multi-agent financial analysis system that combines AI-powered planning, data collection, technical analysis, machine learning predictions, and trading recommendations using the Model Context Protocol (MCP) architecture.

## 🏗️ System Architecture

### Multi-Agent Design

The system employs a **multi-agent architecture** with three specialized agents:

1. **Planning Agent** - Creates comprehensive analysis plans
2. **Analysis Agent** - Executes data collection and analysis steps
3. **Trading Agent** - Generates actionable trading recommendations

### Agent Workflow

```
User Input → Planning Agent → Analysis Agent → Trading Agent → Final Report
                ↓              ↓ (iterative)      ↓
            Analysis Plan   → Data Collection → Trading Plan
                           → Technical Analysis
                           → ML Predictions
                           → News Sentiment
                           → Synthesis
```

## 🧠 Theoretical Foundation

### Agent-Based Systems Theory

**Multi-Agent Systems (MAS)** are computational systems where multiple intelligent agents interact to solve problems that are beyond the capabilities of individual agents. Key principles implemented:

- **Autonomy**: Each agent operates independently with specialized functions
- **Social Ability**: Agents communicate through structured state sharing
- **Reactivity**: Agents respond to environmental changes (market data)
- **Pro-activeness**: Agents take initiative to achieve goals

### Model Context Protocol (MCP)

**MCP** is a protocol for connecting AI models with external tools and data sources:
- Standardizes tool integration
- Enables secure, controlled access to external resources
- Facilitates modular architecture

### LangGraph State Management

**StateGraph** provides:
- Deterministic workflow execution
- State persistence across agent transitions
- Conditional branching based on analysis results
- Error handling and retry mechanisms

## 🔧 Technical Components

### Core Technologies

- **LangChain**: LLM orchestration and prompt management
- **LangGraph**: Workflow state management and agent coordination
- **Pydantic**: Data validation and structured outputs
- **Fireworks AI**: High-performance LLM inference
- **MCP**: Tool integration protocol

### Data Sources

1. **Stock Data API**: Real-time price and volume data
2. **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
3. **ML Prediction Model**: MLflow-hosted prediction model
4. **News API**: CNBC financial news and sentiment

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- API keys for external services

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd financial-analysis-system
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv financial_env
source financial_env/bin/activate  # On Windows: financial_env\Scripts\activate

# Or using conda
conda create -n financial_env python=3.9
conda activate financial_env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Requirements File

Create `requirements.txt` with the following dependencies:

```txt
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-mcp-adapters>=0.1.0
langgraph>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
asyncio
typing-extensions
requests
pandas
numpy
```

### Step 5: MCP Server Setup

Create the following MCP server files in your project directory:

#### `web_search_mcp.py`
```python
# MCP server for web search functionality
# Implementation depends on your news API choice
```

#### `ml_prediction_mcp.py`
```python
# MCP server for ML model predictions
# Connect to your MLflow model endpoint
```

#### `stock_data_mcp.py`
```python
# MCP server for stock data retrieval
# Implementation for stock price and technical indicators
```

### Step 6: Environment Configuration

Create `.env` file in the project root:

```env
# Fireworks AI API Key
FIREWORKS_API_KEY=your_fireworks_api_key_here

# Optional: Additional API keys for data sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
MLFLOW_TRACKING_URI=your_mlflow_server_url
```

### Step 7: API Key Setup

#### Get Fireworks AI API Key:
1. Visit [Fireworks AI](https://fireworks.ai/)
2. Sign up for an account
3. Navigate to API settings
4. Generate an API key
5. Add to `.env` file

#### Optional Additional APIs:
- **Alpha Vantage**: For stock data ([alphavantage.co](https://www.alphavantage.co/))
- **News API**: For news data ([newsapi.org](https://newsapi.org/))
- **MLflow**: For ML model hosting

## 📊 Usage

### Basic Usage

```bash
python main.py "Analyze AAPL stock for investment decision"
```

### Advanced Usage Examples

```bash
# Technical analysis focus
python main.py "Perform technical analysis on TSLA with RSI and MACD indicators"

# News sentiment analysis
python main.py "Analyze NVDA stock with recent news sentiment for day trading"

# Comprehensive analysis
python main.py "Full analysis of MSFT including ML predictions and risk assessment"
```

### Programmatic Usage

```python
import asyncio
from main import main

# Run analysis programmatically
async def analyze_stock():
    await main("Analyze GOOGL stock for long-term investment")

asyncio.run(analyze_stock())
```

## 📈 Output Structure

### Analysis Results

The system provides structured outputs including:

```json
{
  "planning_result": {
    "objective": "Comprehensive analysis of AAPL stock",
    "target_symbol": "AAPL",
    "steps": [...],
    "estimated_confidence": 0.85
  },
  "analysis_results": [
    {
      "analysis_type": "Stock Data Collection",
      "findings": {...},
      "confidence": 0.9,
      "quality_assessment": {...}
    }
  ],
  "trading_plan": {
    "primary_recommendation": "BUY",
    "confidence_score": 0.82,
    "actions": [...],
    "risk_assessment": "Medium risk",
    "position_sizing": "2-5% of portfolio"
  }
}
```

## 🛠️ Configuration

### Model Configuration

Adjust model parameters in the code:

```python
# Planning model (high precision)
planning_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    temperature=0.0,  # Deterministic planning
    parallel_tool_calls=True
)

# Analysis model (balanced)
analysis_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    temperature=0.1  # Low creativity for analysis
)

# Trading model (slightly creative)
trading_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    temperature=0.2  # Moderate creativity for recommendations
)
```

### Workflow Customization

Modify the workflow by adjusting the state graph:

```python
workflow = StateGraph(AgentState)
workflow.add_node("custom_agent", custom_agent_node)
workflow.add_edge("planning_agent", "custom_agent")
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: FIREWORKS_API_KEY is not set
   Solution: Check .env file and ensure API key is valid
   ```

2. **MCP Server Connection Issues**
   ```
   Error: MCP server not responding
   Solution: Ensure MCP server files are properly implemented and accessible
   ```

3. **Tool Integration Failures**
   ```
   Error: Tool not found
   Solution: Verify MCP server implementations and tool registrations
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Use smaller models for faster responses
- Implement caching for repeated API calls
- Optimize MCP server implementations

## 🧪 Testing

### Unit Tests

```bash
python -m pytest tests/
```

### Integration Tests

```bash
python test_integration.py
```

### Manual Testing

```bash
# Test with known stock symbol
python main.py "Quick analysis of AAPL"

# Test error handling
python main.py "Analyze invalid stock XYZ123"
```

## 📚 Advanced Features

### Custom Analysis Steps

Add custom analysis steps by extending the `AnalysisStep` model:

```python
class CustomAnalysisStep(AnalysisStep):
    custom_parameter: str = Field(description="Custom parameter")
```

### Integration with Other Services

The modular MCP architecture allows easy integration with:
- Bloomberg Terminal API
- Reuters data feeds
- Custom ML models
- Portfolio management systems

### Scalability Considerations

- Implement async processing for multiple stocks
- Use connection pooling for API calls
- Consider containerization with Docker
- Implement rate limiting for API usage

