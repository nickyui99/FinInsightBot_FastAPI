"""
SerpAPI client for financial news retrieval.

Provides LangChain-compatible news retrieval using SerpAPI's Google News search.
Fetches recent financial news articles without full-page scraping, using only
the clean snippets provided by SerpAPI for fast and reliable news access.

Key Features:
- LangChain BaseRetriever compatibility
- Google News search via SerpAPI
- Clean snippet-based content (no scraping needed)
- Configurable result limits
- Proper error handling and logging
- Both sync and async retrieval support
"""

import os
import logging
import requests
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

logger = logging.getLogger(__name__)
MAX_NEWS_RESULTS = 5  # Maximum number of news articles to retrieve

class SerpNewsRetriever(BaseRetriever, BaseModel):
    """
    LangChain-compatible news retriever using SerpAPI.
    
    Fetches recent financial news from Google News using SerpAPI without
    requiring full-page scraping. Uses clean snippets provided by SerpAPI
    for fast, reliable news content retrieval.
    
    Features:
    - Inherits from LangChain's BaseRetriever for seamless integration
    - Google News search with US localization
    - Clean snippet-based content (no HTML parsing needed)
    - Structured metadata (title, date, source, link)
    - Error handling with fallback to empty results
    - Both sync and async retrieval methods
    
    Requirements:
    - SERP_API_KEY_1 environment variable must be set
    - Internet connection for API calls
    """
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant news documents for a given query.
        
        Searches Google News via SerpAPI and returns news articles as
        LangChain Document objects with clean snippet content.
        
        Args:
            query: Search query for financial news (e.g., "Tesla earnings", "AAPL stock")
            
        Returns:
            List of Document objects with news content and metadata
            
        Example:
            retriever = SerpNewsRetriever()
            docs = retriever._get_relevant_documents("Apple earnings Q3")
            # Returns: [Document(page_content="Apple reported...", metadata={...})]
        """
        # Check for required API key
        api_key = os.getenv("SERP_API_KEY_1")
        if not api_key:
            logger.error("SERP_API_KEY not set. Skipping news retrieval.")
            return []

        # Configure SerpAPI request for Google News search
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",           # Use Google search engine
            "api_key": api_key,          # SerpAPI authentication
            "q": query,                  # Search query
            "tbm": "nws",               # Search news specifically  
            "num": MAX_NEWS_RESULTS,     # Limit number of results
            "gl": "us",                 # Geographic location (US)
            "hl": "en"                  # Language (English)
        }

        try:
            # Make API request with timeout
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract news results from API response
            items = data.get("news_results", [])
            
            # Convert news items to LangChain Documents
            docs = []
            for item in items[:MAX_NEWS_RESULTS]:
                # Extract article information with fallbacks
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No summary available.")
                link = item.get("link", "")
                date = item.get("date", "Unknown date")
                source = item.get("source", "Unknown source")

                # Create Document with snippet as content (clean and factual)
                docs.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "source": link,              # Article URL
                            "title": title,              # Article headline
                            "date": date,                # Publication date
                            "original_source": source    # News outlet name
                        }
                    )
                )
            return docs

        except Exception as e:
            logger.error(f"News retrieval failed for query '{query}': {e}")
            return []  # Return empty list on failure

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of news document retrieval.
        
        Currently wraps the sync method. Can be enhanced with async HTTP
        client (like aiohttp) for better async performance if needed.
        
        Args:
            query: Search query for financial news
            
        Returns:
            List of Document objects with news content and metadata
        """
        return self._get_relevant_documents(query)