"""
News retrieval agent for financial analysis.

Handles news search and query generation for financial topics.
Uses SerpAPI to find recent financial news articles and LLM to
generate targeted search queries based on user questions.

Functions:
- get_news_retriever: Creates news search client
- generate_news_queries: Creates smart search queries for news
"""

from typing import List
from clients.serpapi_client import SerpNewsRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from utils.llm import get_fast_llm

def get_news_retriever() -> SerpNewsRetriever:
    """
    Get news retriever client for financial news searches.
    
    Returns:
        SerpNewsRetriever: LangChain-compatible news search client
    """
    return SerpNewsRetriever()

def generate_news_queries(question: str, ticker: str = None) -> List[str]:
    """
    Generate targeted news search queries using LLM.
    
    Takes a user question and optional ticker symbol to create
    1-3 focused search queries for finding recent financial news.
    
    Args:
        question: User's financial question or topic
        ticker: Optional stock symbol (e.g., "AAPL", "TSLA")
        
    Returns:
        List of 1-3 search query strings for news APIs
        
    Example:
        question="What's happening with Tesla?"
        ticker="TSLA" 
        Returns: ["Tesla earnings Q3 2024", "TSLA stock news", "Tesla partnership news"]
    """
    llm = get_fast_llm()  # Use fast model for query generation
    
    # Create prompt for generating focused news queries
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial research assistant. Generate 1–3 search queries to find **recent news articles** (last 7 days) about company events, earnings, regulations, partnerships, or market sentiment — NOT technical charts or paywalled analysis."),
        ("human", "User question: {question}\nTicker (if any): {ticker}\n\nGenerate 1 to 3 short search queries (one per line) for recent financial news.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        # Generate queries using LLM
        response = chain.invoke({"question": question, "ticker": ticker or "N/A"})
        
        # Clean and filter the response lines
        queries = [q.strip() for q in response.split("\n") if q.strip() and not q.strip().startswith("•")]
        
        return queries[:3]  # Limit to maximum 3 queries
    except Exception as e:
        print(f"⚠️ Failed to generate news queries: {e}")
        return []