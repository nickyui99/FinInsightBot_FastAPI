"""
Data fetching module for financial agent.

Contains functions for:
- Fetching fundamental analysis data (P/E, earnings, etc.)
- Fetching technical analysis data (charts, indicators)
- Fetching recent news articles for stocks
- Retrieving SEC filing documents from database
"""

import logging
from typing import Dict, Any
from agents.financial_agent_state import FinancialAgentState
from clients.yfinance_client import get_fundamental_data, get_technical_data
from agents.news_retrieval_agent import get_news_retriever, generate_news_queries
from utils.retriever import get_retriever

# Configure logger
logger = logging.getLogger(__name__)

def fetch_fundamental_data(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Fetch fundamental analysis data for stocks.
    
    Gets valuation metrics, earnings, revenue, P/E ratios, etc.
    
    Args:
        state: Agent state with tickers and requirements
        
    Returns:
        Dict with fundamental_data keyed by ticker
    """
    if not state.is_financial or not state.needs_fundamental or not state.ticker:
        return {"fundamental_data": {}}
    
    all_data = {}
    for ticker in state.ticker:
        data = get_fundamental_data(ticker)
        all_data[ticker] = data

    print("Fundamental data fetched:", all_data)
    
    return {"fundamental_data": all_data}

def fetch_technical_data(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Fetch technical analysis data for stocks.
    
    Gets charts, RSI, MACD, price trends, support/resistance levels.
    
    Args:
        state: Agent state with tickers and requirements
        
    Returns:
        Dict with technical_data keyed by ticker
    """
    if not state.is_financial or not state.needs_technical or not state.ticker:
        return {"technical_data": {}}
    
    all_data = {}
    for ticker in state.ticker:
        data = get_technical_data(ticker)
        all_data[ticker] = data
    
    return {"technical_data": all_data}

def fetch_news(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Fetch recent news articles for stocks.
    
    Gets latest announcements, market updates, and company news.
    Deduplicates results and returns top 5 unique articles.
    
    Args:
        state: Agent state with tickers and requirements
        
    Returns:
        Dict with news_articles list (max 5 items)
    """

    print(state)
    
    if not state.is_financial or not state.needs_news:
        logger.warning(f"Skipping news fetch - is_financial: {state.is_financial}, needs_news: {state.needs_news}, ticker: {state.ticker}")
        return {"news_articles": []}
    
    logger.info(f"Starting news fetch for tickers: {state.ticker}")
    
    try:
        all_news = []
        news_retriever = get_news_retriever()
        logger.debug(f"News retriever initialized successfully")
        
        # Generate and execute news queries for each ticker or general query
        if state.ticker and len(state.ticker) > 0:
            # Ticker-specific news queries
            for ticker in state.ticker:
                logger.debug(f"Generating news queries for ticker: {ticker}")
                queries = generate_news_queries(state.query, ticker)
                logger.debug(f"Generated {len(queries)} queries for {ticker}: {queries}")
                
                for i, q in enumerate(queries):
                    logger.debug(f"Executing query {i+1}/{len(queries)} for {ticker}: {q}")
                    results = news_retriever.invoke(q)
                    logger.debug(f"Query returned {len(results)} results")
                    all_news.extend(results)
        else:
            # General financial news query without specific ticker
            logger.debug("No specific ticker found, generating general financial news queries")
            queries = generate_news_queries(state.query, None)
            logger.debug(f"Generated {len(queries)} general queries: {queries}")
            
            for i, q in enumerate(queries):
                logger.debug(f"Executing general query {i+1}/{len(queries)}: {q}")
                results = news_retriever.invoke(q)
                logger.debug(f"Query returned {len(results)} results")
                all_news.extend(results)
        
        # Remove duplicates by source URL
        seen = set()
        unique_news = []
        for doc in all_news:
            src = doc.metadata.get("source")
            if src and src not in seen:
                seen.add(src)
                unique_news.append(doc)
                logger.debug(f"Added unique article: {src}")
            else:
                logger.debug(f"Skipped duplicate article: {src}")
        
        final_count = min(len(unique_news), 5)
        logger.info(f"Returning {final_count} unique news articles for {state.ticker}")
        
        # Log article titles for debugging
        for i, article in enumerate(unique_news[:5]):
            title = article.metadata.get("title", "No title")
            logger.debug(f"Article {i+1}: {title}")
        
        return {"news_articles": unique_news[:5]}
    except Exception as e:
        logger.error(f"News retrieval failed: {str(e)}", exc_info=True)
        return {"news_articles": []}

def retrieve_documents(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Retrieve SEC filing documents from database.
    
    Gets 10-K, 10-Q reports and other financial disclosures
    filtered by requested tickers.
    
    Args:
        state: Agent state with tickers and requirements
        
    Returns:
        Dict with retrieved_docs list
    """
    if not state.is_financial or not state.needs_secfiling or not state.ticker:
        return {"retrieved_docs": []}
    
    retriever = get_retriever()
    
    # Filter documents by requested tickers
    ticker_filter = {"company_ticker": {"$in": state.ticker}}
    
    try:
        docs = retriever.invoke(state.query, filter=ticker_filter)
    except Exception:
        # Fallback without filter if it fails
        docs = retriever.invoke(state.query)
        docs = [d for d in docs if d.metadata.get("company_ticker") in state.ticker]
    
    print(f"Retrieved {len(docs)} documents for {state.ticker}")
    return {"retrieved_docs": docs}