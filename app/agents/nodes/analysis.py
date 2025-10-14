"""
Analysis module for financial agent query processing.

Contains functions for:
- Resolving ambiguous queries using conversation context
- Analyzing queries to determine required financial data types
"""

from typing import Dict, Any, Optional, List
from agents.financial_agent_state import FinancialAgentState
from agents.request_analyzer_agent import analyze_request
from utils.llm import get_fast_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def resolve_query_with_history(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Resolve ambiguous queries using conversation history.
    
    Rewrites the latest user query into a clear, standalone question by 
    incorporating context from previous messages.
    
    Args:
        state: Agent state with conversation messages
            
    Returns:
        Dict with resolved query string
        
    Example:
        Previous: "Tell me about Apple stock"
        Current: "What's its P/E ratio?"
        Resolved: "What's Apple's P/E ratio?"
    """
    print("Messages:", state.messages)
    
    if not state.messages:
        return {"query": "", "is_financial": False}
    
    # Get latest user message
    latest_user_msg: Optional[str] = None
    for msg in reversed(state.messages):
        if msg["role"] == "user":
            latest_user_msg = msg["content"]
            break
    
    if not latest_user_msg:
        return {"query": "", "is_financial": False}
    
    # Single message needs no resolution
    if len(state.messages) == 1:
        return {"query": latest_user_msg}
    
    # Use LLM to rewrite query with context
    llm = get_fast_llm()
    # Create prompt template for query resolution
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            """You are an expert query rewriter for financial analysis. 
            Your task is to rewrite the user's current query into a clear, standalone, and unambiguous question by incorporating ONLY the necessary context from the conversation history.

            **CRITICAL RULES:**
            1. **Pronoun Resolution**: If the current query uses pronouns (it, its, they, them, these, those), replace them with the company/ticker from the MOST RECENT relevant mention.
            2. **Company Switching**: If the user mentions a NEW company name, this is a NEW query topic. DO NOT mix context from previous companies.
            3. **Multiple Companies**: ONLY preserve multiple companies if the current query uses plural pronouns (they, them, these, those) AND refers to companies mentioned together in the same previous query.
            4. **Complete Queries**: If the current query is already complete and specific (mentions company names explicitly), return it as-is or with minimal clarification.
            5. **Context Boundary**: Don't mix attributes from different queries. If the previous query asked about "Tesla's 10K" and current asks "What about Alibaba?", rewrite as "What about Alibaba?" NOT "What about Alibaba's 10K?".
            """
        ),
        (
            "human", 
            "Conversation history:\n{history}\n\nCurrent query: {current_query}\n\nRewritten query:"
        )
    ])
    
    # Build conversation history (excluding current query)
    history_str = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in state.messages[:-1]
    ])
    
    # Execute LLM chain
    chain = prompt | llm | StrOutputParser()
    try:
        resolved_query = chain.invoke({
            "history": history_str,
            "current_query": latest_user_msg
        })
    except Exception as e:
        print(f"Query resolution failed: {e}")
        resolved_query = latest_user_msg

    print(f"Resolved Query: {resolved_query}")
    
    return {"query": resolved_query}

def analyze_user_query(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Analyze query to determine required financial data types.
    
    Args:
        state: Agent state containing the query to analyze
            
    Returns:
        Dict with analysis flags:
        - is_financial: Whether query is financial
        - ticker: Stock symbols mentioned
        - needs_fundamental: Earnings, P/E ratios, etc.
        - needs_technical: Charts, RSI, MACD, etc.
        - needs_news: Recent announcements, events
        - needs_secfiling: 10-K, 10-Q reports, etc.
    """

    print("Analyzing query:", state.query)

    # Classify query intent and extract requirements
    analysis = analyze_request(state.query)
    
    print("Analyzer Result:", analysis)
    
    # Return analysis results with proper key mapping
    return {
        "is_financial": analysis.get("is_financial", False),
        "ticker": analysis.get("ticker", []),  # Get ticker directly - analyzer returns "ticker" not "ticker_symbols"
        "needs_fundamental": analysis.get("needs_fundamental", False),
        "needs_technical": analysis.get("needs_technical", False),
        "needs_news": analysis.get("needs_news", False),
        "needs_secfiling": analysis.get("needs_secfiling", False),
    }