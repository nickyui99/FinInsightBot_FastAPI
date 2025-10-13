"""
Financial agent workflow graph using LangGraph.

Defines the complete workflow for processing financial queries:
1. Resolve query with conversation context
2. Analyze query to determine data requirements  
3. Fetch required data (fundamental, technical, news, documents)
4. Generate comprehensive financial analysis response

The graph handles parallel data fetching and ensures all data is
collected before generating the final answer.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

# Import node functions
from agents.nodes.analysis import analyze_user_query, resolve_query_with_history
from agents.nodes.data_fetching import fetch_fundamental_data, fetch_technical_data, fetch_news, retrieve_documents
from agents.nodes.response import generate_final_answer
from agents.financial_agent_state import FinancialAgentState

def create_financial_agent_graph():
    """
    Create and configure the financial agent workflow graph.
    
    Builds a LangGraph workflow that processes financial queries through
    multiple stages: query resolution, analysis, data fetching, and response 
    generation. All data fetching operations run in parallel.
    
    Returns:
        Compiled LangGraph workflow ready for execution
        
    Workflow Steps:
        1. resolve_query: Clean up query using conversation context
        2. analyze: Determine what financial data is needed
        3. Data fetching (parallel): Get fundamental, technical, news, SEC data
        4. generate_answer: Create comprehensive response from all data
    """
    workflow = StateGraph(FinancialAgentState)
    
    # Add workflow nodes
    workflow.add_node("resolve_query", resolve_query_with_history)     # Clean query with context
    workflow.add_node("analyze", analyze_user_query)                   # Determine data needs
    workflow.add_node("retrieve_docs", retrieve_documents)             # Get SEC filings
    workflow.add_node("fetch_fundamental", fetch_fundamental_data)     # Get P/E, earnings, etc.
    workflow.add_node("fetch_technical", fetch_technical_data)         # Get charts, indicators
    workflow.add_node("fetch_news", fetch_news)                       # Get recent news
    workflow.add_node("generate_answer", generate_final_answer)        # Create final response
    
    # Set workflow entry point
    workflow.set_entry_point("resolve_query")
    
    # Define workflow connections
    workflow.add_edge("resolve_query", "analyze")
    
    # Parallel data fetching (each node has internal requirement checks)
    workflow.add_edge("analyze", "retrieve_docs")
    workflow.add_edge("analyze", "fetch_fundamental")
    workflow.add_edge("analyze", "fetch_technical")
    workflow.add_edge("analyze", "fetch_news")

    # All data nodes feed into final response generation
    workflow.add_edge("retrieve_docs", "generate_answer")
    workflow.add_edge("fetch_fundamental", "generate_answer")
    workflow.add_edge("fetch_technical", "generate_answer")
    workflow.add_edge("fetch_news", "generate_answer")

    # Complete workflow
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# Create the global workflow instance
financial_graph = create_financial_agent_graph()
"""
Global financial agent workflow instance.

This compiled workflow can be invoked with a FinancialAgentState
to process financial queries through the complete analysis pipeline.

Usage:
    result = financial_graph.invoke(initial_state)
"""
