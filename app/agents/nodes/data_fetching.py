"""
Data fetching module for financial agent.

Contains functions for:
- Fetching fundamental analysis data (P/E, earnings, etc.)
- Fetching technical analysis data (charts, indicators)
- Fetching recent news articles for stocks
- Retrieving SEC filing documents from database
"""

import logging
from typing import Dict, Any, List
import re
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

def _extract_year(text: str) -> str | None:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group(1) if match else None


def _section_score(title: str) -> int:
    """Heuristic scores to prefer informative 10-K sections."""
    if not title:
        return 0
    t = title.lower()
    # High-value sections
    if any(key in t for key in [
        "item 7", "management's discussion", "managements discussion", "md&a",
        "item 1a", "risk factors",
        "item 1:", "item 1.", "business",
        "item 7a", "quantitative and qualitative",
        "item 8", "financial statements"
    ]):
        return 3
    # Neutral
    if any(key in t for key in ["item 2", "properties", "legal proceedings", "market for"]):
        return 1
    # Low-value (signatures, index only, exhibits)
    if any(key in t for key in ["signatures", "index", "exhibit", "controls and procedures"]):
        return -2
    # Default
    return 0


def _rerank_docs(docs: List[Any]) -> List[Any]:
    return sorted(docs, key=lambda d: _section_score(d.metadata.get("section_title", "")), reverse=True)


def retrieve_documents(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Retrieve SEC filing documents from database using the user's query.
    
    Args:
        state: Agent state with tickers and requirements
        
    Returns:
        Dict with retrieved_docs list
    """
    if not state.is_financial or not state.needs_secfiling:
        logger.info(f"Skipping document retrieval - is_financial: {state.is_financial}, needs_secfiling: {state.needs_secfiling}")
        return {"retrieved_docs": []}
    
    try:
        # Build retrieval filters and strategy
        search_kwargs: Dict[str, Any] = {"k": 8, "fetch_k": 40, "lambda_mult": 0.35}
        # Metadata filter by ticker if provided
        if state.ticker:
            tickers_upper = [t.upper() for t in state.ticker]
            if len(tickers_upper) == 1:
                search_kwargs["filter"] = {"company_ticker": tickers_upper[0]}
            else:
                search_kwargs["filter"] = {"company_ticker": {"$in": tickers_upper}}

        retriever = get_retriever(search_type="mmr", search_kwargs=search_kwargs)
        logger.info(f"Starting document retrieval with query: {state.query}")
        
        # Enhanced query with ticker context if available
        query = state.query
        if state.ticker:
            # Add ticker names to help with retrieval
            ticker_str = " ".join(state.ticker)
            query = f"{state.query} {ticker_str}"
            logger.info(f"Enhanced query with tickers: {query}")
        
        # Retrieve documents using standard LangChain API (prefer invoke per deprecation notice)
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            raise AttributeError("Retriever has no supported retrieval method (missing invoke and get_relevant_documents)")

        # Ensure docs is a list
        if docs is None:
            docs = []
        
        try:
            docs = list(docs)
        except TypeError:
            docs = [docs] if docs else []

        logger.info(f"Retrieved {len(docs)} documents from retriever")
        
        # Optional: Filter by ticker/year if specified and results exist
        if docs:
            filtered = docs
            if state.ticker:
                tickers_upper = [t.upper() for t in state.ticker]
                filtered = [d for d in filtered if d.metadata.get('company_ticker', '').upper() in tickers_upper]
                if filtered:
                    docs = filtered

            # Year filter from query (e.g., 2024)
            year = _extract_year(state.query)
            if year:
                filtered = [d for d in docs if year in str(d.metadata.get('period_end', ''))]
                if filtered:
                    docs = filtered

            # Rerank to prefer informative sections
            docs = _rerank_docs(docs)

            # If top doc still looks uninformative, try a refined query
            top_title = docs[0].metadata.get('section_title', '').lower() if docs else ''
            if docs and any(key in top_title for key in ["signatures", "item 16"]):
                refined = f"{state.query} Item 7 Management's Discussion and Analysis OR Item 1 Business OR Item 8 Financial Statements"
                logger.info("Top result looked uninformative (Signatures/Item 16). Running refined retrieval.")
                more = retriever.invoke(refined) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(refined)
                if more:
                    more = _rerank_docs(list(more))
                    # Merge and keep unique by id if available
                    seen_ids = set()
                    merged: List[Any] = []
                    for d in list(docs) + list(more):
                        did = getattr(d, 'id', None) or d.metadata.get('source_file') or id(d)
                        if did in seen_ids:
                            continue
                        seen_ids.add(did)
                        merged.append(d)
                    docs = merged[:8]
        
        # Log sample results for debugging
        if docs:
            sample = docs[0]
            logger.info(f"Sample doc metadata: {sample.metadata}")
            logger.info(f"Sample content preview: {sample.page_content[:200]}...")
        else:
            logger.warning("No documents retrieved - check if ChromaDB is populated")
        
        return {"retrieved_docs": docs}
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}", exc_info=True)
        return {"retrieved_docs": []}
    