"""
Financial agent workflow graph using LangGraph.

Defines the complete workflow for processing financial queries:
1. Resolve query with conversation context (rewrite query)
2. Analyze query to determine data requirements (analyze intent)
3. Fetch required data conditionally based on intent
4. Generate comprehensive financial analysis response

The graph uses conditional edges to only execute necessary data fetching nodes.
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
    multiple stages with conditional routing based on query intent.
    
    Returns:
        Compiled LangGraph workflow ready for execution
        
    Workflow Steps:
        1. resolve_query: Rewrite query using conversation context
        2. analyze: Analyze intent and determine what data is needed
        3. Data fetching (conditional): Only fetch what's required
        4. generate_answer: Create comprehensive response from collected data
    """
    workflow = StateGraph(FinancialAgentState)
    
    # Step 1: Add node to rewrite/resolve query with conversation history
    workflow.add_node("resolve_query", resolve_query_with_history)
    
    # Step 2: Add node to analyze query intent
    workflow.add_node("analyze", analyze_user_query)
    
    # Step 3: Add conditional data fetching nodes
    workflow.add_node("fetch_fundamental", fetch_fundamental_data)
    workflow.add_node("fetch_technical", fetch_technical_data)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("retrieve_docs", retrieve_documents)
    
    # Step 4: Add final answer generation node
    workflow.add_node("generate_answer", generate_final_answer)
    
    # === ROUTING LOGIC ===
    
    def route_after_analysis(state: FinancialAgentState) -> str:
        """
        Route to appropriate data fetching node after analyzing intent.
        Priority order: fundamental → technical → news → secfiling → answer
        """
        if state.needs_fundamental:
            return "fetch_fundamental"
        elif state.needs_technical:
            return "fetch_technical"
        elif state.needs_news:
            return "fetch_news"
        elif state.needs_secfiling:
            return "retrieve_docs"
        else:
            # No data needed, go straight to answer generation
            return "generate_answer"
    
    def route_after_fundamental(state: FinancialAgentState) -> str:
        """Route after fetching fundamental data."""
        if state.needs_technical:
            return "fetch_technical"
        elif state.needs_news:
            return "fetch_news"
        elif state.needs_secfiling:
            return "retrieve_docs"
        else:
            return "generate_answer"
    
    def route_after_technical(state: FinancialAgentState) -> str:
        """Route after fetching technical data."""
        if state.needs_news:
            return "fetch_news"
        elif state.needs_secfiling:
            return "retrieve_docs"
        else:
            return "generate_answer"
    
    def route_after_news(state: FinancialAgentState) -> str:
        """Route after fetching news."""
        if state.needs_secfiling:
            return "retrieve_docs"
        else:
            return "generate_answer"
    
    # === WORKFLOW EDGES ===
    
    # Set entry point
    workflow.set_entry_point("resolve_query")
    
    # Step 1 → Step 2: Always go from resolve to analyze
    workflow.add_edge("resolve_query", "analyze")
    
    # Step 2 → Step 3: Conditional routing based on analysis
    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "fetch_fundamental": "fetch_fundamental",
            "fetch_technical": "fetch_technical",
            "fetch_news": "fetch_news",
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer"
        }
    )
    
    # Step 3: Chain data fetching nodes conditionally
    workflow.add_conditional_edges(
        "fetch_fundamental",
        route_after_fundamental,
        {
            "fetch_technical": "fetch_technical",
            "fetch_news": "fetch_news",
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_conditional_edges(
        "fetch_technical",
        route_after_technical,
        {
            "fetch_news": "fetch_news",
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_conditional_edges(
        "fetch_news",
        route_after_news,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer"
        }
    )
    
    # Last data fetching node goes to answer generation
    workflow.add_edge("retrieve_docs", "generate_answer")
    
    # Step 4: End after generating answer
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()


# Create the global workflow instance
financial_graph = create_financial_agent_graph()
"""
Global financial agent workflow instance.

This compiled workflow can be invoked with a FinancialAgentState
to process financial queries through the complete analysis pipeline.

Workflow Flow:
    User Query 
    → resolve_query (rewrite with context)
    → analyze (determine intent & requirements)
    → [conditional] fetch_fundamental (if needs_fundamental=True)
    → [conditional] fetch_technical (if needs_technical=True)
    → [conditional] fetch_news (if needs_news=True)
    → [conditional] retrieve_docs (if needs_secfiling=True)
    → generate_answer (create final response)
    → END

Usage:
    # Single invocation
    result = financial_graph.invoke(initial_state)
    
    # Streaming
    for chunk in financial_graph.stream(initial_state):
        print(chunk)
"""