import os
import asyncio
import sys
import json
import re
from typing import Dict, List, Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(override=True)
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = os.environ["FIREWORKS_API_KEY"]
# Set API keys
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY is not set")
os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
os.environ["OPENAI_API_KEY"] = FIREWORKS_API_KEY

# Simplified Pydantic models
class AnalysisStep(BaseModel):
    step_name: str
    description: str
    priority: int = 3

class AnalysisPlan(BaseModel):
    objective: str
    target_symbol: Optional[str] = None
    steps: List[AnalysisStep]

class AnalysisResult(BaseModel):
    analysis_type: str
    findings: Dict
    confidence: float = 0.8
    recommendations: List[str] = []

class TradingPlan(BaseModel):
    primary_recommendation: str
    confidence_score: float
    reasoning: str
    risk_level: str = "MEDIUM"
    time_horizon: str = "MEDIUM_TERM"

# Configure models
planning_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    openai_api_base='https://api.fireworks.ai/inference/v1',
    openai_api_key=os.environ["FIREWORKS_API_KEY"],
    temperature=0.0
).with_structured_output(AnalysisPlan)

analysis_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    openai_api_base='https://api.fireworks.ai/inference/v1',
    openai_api_key=os.environ["FIREWORKS_API_KEY"],
    temperature=0.1
)

trading_model = ChatOpenAI(
    model='accounts/fireworks/models/llama-v3p1-70b-instruct',
    openai_api_base='https://api.fireworks.ai/inference/v1',
    openai_api_key=os.environ["FIREWORKS_API_KEY"],
    temperature=0.2
).with_structured_output(TradingPlan)

# Simplified state
class AgentState(TypedDict):
    user_prompt: str
    identified_symbols: List[str]
    planning_result: Optional[AnalysisPlan]
    analysis_results: List[AnalysisResult]
    trading_plan: Optional[TradingPlan]
    raw_data: Dict[str, Dict]
    current_step: int
    messages: Annotated[List[BaseMessage], add_messages]
    tools: List

def safe_json_parse(text: str) -> Dict:
    """Safely parse JSON from text"""
    if not text or not isinstance(text, str):
        return {"error": "Empty or invalid input"}
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Extract JSON from code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```[\w\s]*\s*(\{.*?\})\s*```',
        r'\s*(\{.*?\})\s*'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    return {"content": text, "parsed": False}

async def setup_mcp_client():
    """Setup MCP client"""
    try:
        client = MultiServerMCPClient({
            "web_search": {"command": "python", "args": ["web_search_mcp.py"], "transport": "stdio"},
            "model_prediction": {"command": "python", "args": ["ml_prediction_mcp.py"], "transport": "stdio"},
            "stock_data": {"command": "python", "args": ["stock_data_mcp.py"], "transport": "stdio"}
        })
        return client
    except Exception as e:
        print(f"MCP client setup failed: {e}")
        return None

async def planning_agent_node(state: AgentState) -> AgentState:
    """Planning Agent creates analysis plan with detailed diagnostics"""
    user_prompt = state.get("user_prompt", "")
    if not user_prompt:
        if state.get("messages"):
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    user_prompt = msg.content
                    break
    
    print(f"🔍 PLANNING DIAGNOSTICS:")
    print(f"   User prompt: {user_prompt[:100]}...")
    print(f"   Planning model configured: {planning_model is not None}")
    
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Financial Planning Agent. Create an analysis plan based on the user's request.
        Extract any stock symbols and create 3-5 specific analysis steps.
        Return a structured plan."""),
        ("user", "{user_prompt}")
    ])

    try:
        print("🔄 Attempting to call LLM planning model...")
        
        # Test the model call
        response = await planning_model.ainvoke(
            planning_prompt.format_messages(user_prompt=user_prompt)
        )
        
        print("✅ LLM planning model call successful!")
        print(f"   Response type: {type(response)}")
        print(f"   Response has target_symbol: {hasattr(response, 'target_symbol')}")
        print(f"   Response has steps: {hasattr(response, 'steps')}")
        
        if hasattr(response, 'steps'):
            print(f"   Number of steps: {len(response.steps)}")
        
        # Extract symbol from user prompt if not in response
        if not response.target_symbol:
            symbol_match = re.search(r'\b([A-Z]{1,5})\b', user_prompt.upper())
            if symbol_match:
                response.target_symbol = symbol_match.group(1)
                print(f"   Extracted symbol from prompt: {symbol_match.group(1)}")
        
        state["planning_result"] = response
        state["identified_symbols"] = [response.target_symbol] if response.target_symbol else ["AAPL"]
        state["current_step"] = 0
        
        # Print successful plan
        print("\n" + "="*60)
        print("✅ SUCCESSFUL LLM-GENERATED ANALYSIS PLAN")
        print("="*60)
        print(f"🎯 Objective: {response.objective}")
        print(f"📈 Target Symbol: {response.target_symbol or 'Not specified'}")
        print(f"📊 Total Steps: {len(response.steps)}")
        print("\n📝 LLM-Generated Analysis Steps:")
        
        for i, step in enumerate(response.steps, 1):
            print(f"  {i}. {step.step_name.upper()}")
            print(f"     Description: {step.description}")
            print(f"     Priority: {step.priority}/5")
            print()
        
        print("="*60)
        
        state["messages"].append(AIMessage(content=f"✅ Created LLM-generated analysis plan for {response.target_symbol or 'stock'} with {len(response.steps)} steps"))
        
        return state
        
    except Exception as e:
        print(f"❌ LLM Planning failed! Error details:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Try to get more specific error info
        if hasattr(e, 'response'):
            print(f"   HTTP Response: {e.response}")
        
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()
        
        print("\n🔄 Falling back to hardcoded plan...")
        
        # Fallback plan
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', user_prompt.upper())
        symbol = symbol_match.group(1) if symbol_match else "AAPL"
        
        fallback_plan = AnalysisPlan(
            objective=f"Analyze {symbol} stock for investment decision",
            target_symbol=symbol,
            steps=[
                AnalysisStep(step_name="stock_data", description="Get current stock data", priority=5),
                AnalysisStep(step_name="technical_analysis", description="Analyze technical indicators", priority=4),
                AnalysisStep(step_name="ml_prediction", description="Get ML predictions", priority=4),
                AnalysisStep(step_name="news_analysis", description="Analyze recent news", priority=3)
            ]
        )
        
        print("\n" + "="*60)
        print("⚠️  FALLBACK ANALYSIS PLAN ACTIVATED")
        print("="*60)
        print(f"🎯 Objective: {fallback_plan.objective}")
        print(f"📈 Target Symbol: {fallback_plan.target_symbol}")
        print(f"📊 Total Steps: {len(fallback_plan.steps)}")
        print("\n📝 Fallback Analysis Steps:")
        
        for i, step in enumerate(fallback_plan.steps, 1):
            print(f"  {i}. {step.step_name.upper()}")
            print(f"     Description: {step.description}")
            print(f"     Priority: {step.priority}/5")
            print()
        
        print(f"❌ Reason for fallback: {str(e)}")
        print("="*60)
        
        state["planning_result"] = fallback_plan
        state["identified_symbols"] = [symbol]
        state["current_step"] = 0
        state["messages"].append(AIMessage(content=f"⚠️ Created fallback plan for {symbol} due to error: {str(e)}"))

    return state

# Analysis Agent
async def analysis_agent_node(state: AgentState) -> AgentState:
    """Analysis Agent executes analysis steps"""
    planning_result = state.get("planning_result")
    if not planning_result or not planning_result.steps:
        state["messages"].append(AIMessage(content="No analysis plan available"))
        return state

    current_step = state.get("current_step", 0)
    if current_step >= len(planning_result.steps):
        state["messages"].append(AIMessage(content="All analysis steps completed"))
        return state

    step = planning_result.steps[current_step]
    symbol = state["identified_symbols"][0] if state["identified_symbols"] else "AAPL"
    tools = state.get("tools", [])

    try:
        if "stock" in step.step_name.lower():
            result = await gather_stock_data(tools, symbol)
        elif "technical" in step.step_name.lower():
            result = await perform_technical_analysis(tools, symbol)
        elif "ml" in step.step_name.lower() or "prediction" in step.step_name.lower():
            result = await generate_ml_predictions(tools, state)
        elif "news" in step.step_name.lower():
            result = await analyze_news_sentiment(tools, symbol)
        else:
            result = AnalysisResult(
                analysis_type=step.step_name,
                findings={"status": "completed"},
                recommendations=[f"Completed {step.step_name}"]
            )

        state["analysis_results"] = state.get("analysis_results", []) + [result]
        state["raw_data"][step.step_name] = result.findings
        state["current_step"] = current_step + 1
        state["messages"].append(AIMessage(content=f"Completed {step.step_name}"))

    except Exception as e:
        error_result = AnalysisResult(
            analysis_type=step.step_name,
            findings={"error": str(e)},
            confidence=0.0,
            recommendations=[f"Failed to complete {step.step_name}"]
        )
        state["analysis_results"] = state.get("analysis_results", []) + [error_result]
        state["current_step"] = current_step + 1
        state["messages"].append(AIMessage(content=f"Error in {step.step_name}: {str(e)}"))

    return state

# Trading Agent
async def trading_agent_node(state: AgentState) -> AgentState:
    """Trading Agent creates trading plan"""
    user_prompt = state.get("user_prompt", "")
    analysis_results = state.get("analysis_results", [])
    
    trading_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Trading Strategy Agent. Create a trading plan based on analysis results.
        Provide a clear BUY/SELL/HOLD recommendation with reasoning."""),
        ("user", """User Request: {user_prompt}
        Analysis Results: {analysis_results}
        Provide a trading recommendation.""")
    ])

    try:
        response = await trading_model.ainvoke(
            trading_prompt.format_messages(
                user_prompt=user_prompt,
                analysis_results=json.dumps([r.dict() for r in analysis_results], indent=2)
            )
        )
        
        state["trading_plan"] = response
        state["messages"].append(AIMessage(content=f"Trading recommendation: {response.primary_recommendation}"))
        
    except Exception as e:
        # Fallback trading plan
        fallback_plan = TradingPlan(
            primary_recommendation="HOLD",
            confidence_score=0.5,
            reasoning="Insufficient data for confident recommendation",
            risk_level="MEDIUM"
        )
        
        state["trading_plan"] = fallback_plan
        state["messages"].append(AIMessage(content=f"Error creating trading plan: {str(e)}"))

    return state

# Analysis helper functions
async def gather_stock_data(tools: List, symbol: str) -> AnalysisResult:
    """Gather stock data"""
    stock_tool = next((tool for tool in tools if "get_stock_price" in tool.name), None)
    if stock_tool:
        try:
            result = await stock_tool.ainvoke({"ticker": symbol})
            data = parse_tool_result(result)
            return AnalysisResult(
                analysis_type="Stock Data",
                findings=data,
                recommendations=[f"Current data for {symbol} retrieved"]
            )
        except Exception as e:
            return AnalysisResult(
                analysis_type="Stock Data",
                findings={"error": str(e)},
                confidence=0.0
            )
    
    return AnalysisResult(
        analysis_type="Stock Data",
        findings={"error": "Stock tool not available"},
        confidence=0.0
    )

async def perform_technical_analysis(tools: List, symbol: str) -> AnalysisResult:
    """Perform technical analysis"""
    tech_tool = next((tool for tool in tools if "get_technical_indicators" in tool.name), None)
    if tech_tool:
        try:
            result = await tech_tool.ainvoke({"ticker": symbol})
            data = parse_tool_result(result)
            return AnalysisResult(
                analysis_type="Technical Analysis",
                findings=data,
                recommendations=[f"Technical indicators for {symbol} analyzed"]
            )
        except Exception as e:
            return AnalysisResult(
                analysis_type="Technical Analysis",
                findings={"error": str(e)},
                confidence=0.0
            )
    
    return AnalysisResult(
        analysis_type="Technical Analysis",
        findings={"error": "Technical tool not available"},
        confidence=0.0
    )

async def generate_ml_predictions(tools: List, state: AgentState) -> AnalysisResult:
    """Generate ML predictions"""
    pred_tool = next((tool for tool in tools if "get_price_prediction" in tool.name), None)
    if pred_tool:
        try:
            # Use data from previous steps
            stock_data = state.get("raw_data", {}).get("stock_data", {})
            tech_data = state.get("raw_data", {}).get("technical_analysis", {})
            
            prediction_params = {
                "open": stock_data.get("open_price", 0),
                "high": stock_data.get("high_price", 0),
                "low": stock_data.get("low_price", 0),
                "close": stock_data.get("current_price", 0),
                "volume": stock_data.get("volume", 0),
                "vwap": stock_data.get("current_price", 0),
                "rsi": tech_data.get("rsi", 50),
                "stochastic_k": tech_data.get("stochastic_k", 50),
                "stochastic_d": tech_data.get("stochastic_d", 50),
                "bb_upper": tech_data.get("bb_upper", 0),
                "bb_middle": tech_data.get("bb_middle", 0),
                "bb_lower": tech_data.get("bb_lower", 0),
                "bb_width": tech_data.get("bb_upper", 0) - tech_data.get("bb_lower", 0),
                "macd": tech_data.get("macd", 0),
                "macd_signal": tech_data.get("macd_signal", 0),
                "macd_hist": tech_data.get("macd_histogram", 0),
                "sma_9": tech_data.get("sma_9", 0),
                "sma_20": tech_data.get("sma_20", 0)
            }
            
            result = await pred_tool.ainvoke(prediction_params)
            data = parse_tool_result(result)
            return AnalysisResult(
                analysis_type="ML Prediction",
                findings=data,
                recommendations=["ML prediction completed"]
            )
        except Exception as e:
            return AnalysisResult(
                analysis_type="ML Prediction",
                findings={"error": str(e)},
                confidence=0.0
            )
    
    return AnalysisResult(
        analysis_type="ML Prediction",
        findings={"error": "ML tool not available"},
        confidence=0.0
    )

async def analyze_news_sentiment(tools: List, symbol: str) -> AnalysisResult:
    """Analyze news sentiment"""
    news_tool = next((tool for tool in tools if "search_cnbc_news" in tool.name), None)
    if news_tool:
        try:
            result = await news_tool.ainvoke({"query": f"{symbol} stock news"})
            data = parse_tool_result(result)
            return AnalysisResult(
                analysis_type="News Analysis",
                findings=data,
                recommendations=["News sentiment analyzed"]
            )
        except Exception as e:
            return AnalysisResult(
                analysis_type="News Analysis",
                findings={"error": str(e)},
                confidence=0.0
            )
    
    return AnalysisResult(
        analysis_type="News Analysis",
        findings={"error": "News tool not available"},
        confidence=0.0
    )

def parse_tool_result(result):
    """Parse tool result"""
    if isinstance(result, str):
        parsed = safe_json_parse(result)
        return parsed if parsed.get("parsed", True) else {"content": result}
    elif isinstance(result, dict):
        return result
    return {"content": str(result)}

def should_continue(state: AgentState) -> Literal["analysis_agent", "trading_agent", "__end__"]:
    """Determine next step"""
    if state.get("trading_plan"):
        return "__end__"
    
    planning_result = state.get("planning_result")
    current_step = state.get("current_step", 0)
    
    if planning_result and current_step < len(planning_result.steps):
        return "analysis_agent"
    elif current_step >= len(planning_result.steps if planning_result else []):
        return "trading_agent"
    
    return "__end__"

async def create_analysis_workflow():
    """Create the analysis workflow"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planning_agent", planning_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("trading_agent", trading_agent_node)
    
    workflow.add_edge(START, "planning_agent")
    workflow.add_edge("planning_agent", "analysis_agent")
    workflow.add_conditional_edges(
        "analysis_agent",
        should_continue,
        {
            "analysis_agent": "analysis_agent",
            "trading_agent": "trading_agent",
            "__end__": END
        }
    )
    workflow.add_edge("trading_agent", END)
    
    return workflow.compile()

# Create graph for LangGraph deployment
async def create_graph():
    """Create graph for deployment"""
    client = await setup_mcp_client()
    tools = await client.get_tools() if client else []
    
    workflow = await create_analysis_workflow()
    return workflow

# Export graph
try:
    graph = asyncio.run(create_graph())
except Exception as e:
    print(f"Error creating graph: {e}")
    # Create a simple fallback graph
    async def fallback_graph():
        return await create_analysis_workflow()
    graph = asyncio.run(fallback_graph())

# Main function for local testing
async def main(prompt: str = "Analyze AAPL stock for investment decision"):
    """Main function for local testing"""
    print(f"Starting analysis: {prompt}")
    
    try:
        client = await setup_mcp_client()
        tools = await client.get_tools() if client else []
        workflow = await create_analysis_workflow()
        
        initial_state: AgentState = {
            "user_prompt": prompt,
            "identified_symbols": [],
            "planning_result": None,
            "analysis_results": [],
            "trading_plan": None,
            "raw_data": {},
            "current_step": 0,
            "messages": [HumanMessage(content=prompt)],
            "tools": tools
        }
        
        final_state = await workflow.ainvoke(initial_state)
        
        print("\n" + "="*60)
        print("FINANCIAL ANALYSIS RESULTS")
        print("="*60)
        
        for result in final_state.get("analysis_results", []):
            print(f"\n{result.analysis_type}:")
            print(json.dumps(result.dict(), indent=2))
        
        if final_state.get("trading_plan"):
            print(f"\nTRADING RECOMMENDATION:")
            print(json.dumps(final_state["trading_plan"].dict(), indent=2))
        
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    prompt = """I want a detailed financial analysis for MSFT stock. Please follow these steps exactly:
        1. First, use the stock_data to get the stock data and technical data and pass it to the model_prediction tool to get ml predictions data for AAPL, which will include:
           - Current stock price and basic information
           - Technical indicators
           - ml model predictions from the MLflow model
        2. Search for recent news about the stock using the search_cnbc_news tool.
        3. Finally, analyze all the data and provide a recommendation including:
           - Current price analysis
           - Technical indicator interpretation
           - ml model prediction results (directly from MLflow)
           - Recent news summary with article links
           - Buy/sell/hold recommendation with confidence level and reasoning"""
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    asyncio.run(main(prompt))
