import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv(override=True)

mcp = FastMCP(
    name="web_search",
    version="1.0.0",
    description="Web search tool for retrieving the latest financial news from CNBC about specific stocks, market trends, and economic events",
    host="127.0.0.1",  # Host address (localhost)
    port=8004,  # Port number for the server
)


# Set API key
tavily_api_key = os.getenv("TAVILY_API_KEY", "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")

@mcp.tool()
async def search_cnbc_news(query: str) -> dict:
    """
    Search for market news on CNBC related to specific stocks or financial information.
    """
    try:
        from tavily import TavilyClient
        
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        search_query = f"{query} stock news site:www.cnbc.com"
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
        
        return {
            "content": content,
            "urls": urls,
            "status": "success" 
        }
    except Exception as e:
        return {
            "content": f"Error searching news: {str(e)}",
            "urls": [],
            "status": "error"
        }

if __name__ == "__main__":
    mcp.run(transport="sse")