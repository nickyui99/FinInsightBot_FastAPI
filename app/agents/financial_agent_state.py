"""
Financial agent state management.

Defines the FinancialAgentState class that manages all data and context
throughout the financial analysis workflow. This state is passed between
all workflow nodes and accumulates data as the analysis progresses.

The state tracks:
- Conversation history and current query
- Analysis requirements (what data is needed)
- Collected financial data (fundamental, technical, news, documents)
- Final analysis result
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FinancialAgentState(BaseModel):
    """
    State container for the financial agent workflow.
    
    Manages all data and context throughout the financial analysis process.
    State is updated by each workflow node and passed to the next stage.
    
    Attributes:
        # Conversation Context
        messages: Chat history for multi-turn conversations
        query: Current resolved query to analyze
        
        # Analysis Requirements  
        ticker: Stock symbols to analyze (e.g., ["AAPL", "TSLA"])
        is_financial: Whether query is about financial topics
        needs_fundamental: Whether fundamental data is required
        needs_technical: Whether technical analysis is needed
        needs_news: Whether recent news is requested
        needs_secfiling: Whether SEC filings are needed
        
        # Collected Data
        fundamental_data: P/E ratios, earnings, financial metrics by ticker
        technical_data: Charts, RSI, MACD indicators by ticker  
        news_articles: Recent news and announcements
        retrieved_docs: SEC filings and financial documents
        
        # Final Result
        answer: Comprehensive financial analysis response
    """
    
    # Conversation context
    messages: List[Dict[str, str]] = []  # Chat history
    query: str = ""                       # Current query

    # Analysis requirements
    ticker: List[str] = []               # Stock symbols to analyze
    is_financial: bool = False           # Is this a financial query?
    needs_fundamental: bool = False      # Need P/E, earnings, etc.?
    needs_technical: bool = False        # Need charts, indicators?
    needs_news: bool = False            # Need recent news?
    needs_secfiling: bool = False       # Need SEC documents?
    
    # Collected financial data
    fundamental_data: Dict[str, Any] = Field(default_factory=dict)  # Market metrics
    technical_data: Dict[str, Any] = Field(default_factory=dict)    # Technical indicators
    news_articles: List[Any] = Field(default_factory=list)         # Recent news
    retrieved_docs: List[Any] = Field(default_factory=list)        # SEC filings
    
    # Final analysis result
    answer: str = ""  # Comprehensive response