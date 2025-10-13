"""
Response generation module for financial agent.

Contains functions for:
- Generating final answers using all collected data
- Formatting market data, news, and document context
- Creating comprehensive financial analysis responses
"""

from typing import Dict, Any
from pydantic import BaseModel
from agents.financial_agent_state import FinancialAgentState
from utils.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_final_answer(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Generate comprehensive financial analysis response.
    
    Combines all collected data (fundamental, technical, news, documents) 
    with conversation history to create a professional financial answer.
    
    Args:
        state: Agent state with all collected financial data
        
    Returns:
        Dict with final answer string
    """
    
    # Build market context from fundamental and technical data
    market_parts = []
    for ticker in state.ticker:
        f_data = state.fundamental_data.get(ticker, {})
        t_data = state.technical_data.get(ticker, {})

        print("t_data", t_data)
        
        # Format fundamental analysis data
        if f_data and "error" not in str(f_data.get("error", "")):
            lines = [
                f"📊 FUNDAMENTAL ANALYSIS for {ticker} (Retrieved from Yahoo Finance):",
                f"• Price: ${f_data.get('current_price', 'N/A')}",
                f"• Market Cap: ${f_data['market_cap']:,.0f}" if f_data.get('market_cap') else "• Market Cap: N/A",
                f"• P/E Ratio: {f_data.get('pe_ratio', 'N/A')}",
                f"• EPS: ${f_data.get('eps', 'N/A')}",
                f"• 52W Range: ${f_data.get('52w_low', 'N/A')} – ${f_data.get('52w_high', 'N/A')}",
            ]
            market_parts.append("\n".join(lines))
        
        # Format technical analysis data
        if t_data and "error" not in str(t_data.get("error", "")):
            # Format all technical indicators
            formatted_lines = []
            for key, value in t_data.items():
                if isinstance(value, (int, float)):
                    name = key.replace("_", " ").upper()
                    formatted_lines.append(f"• {name}: {value:.2f}")
                elif isinstance(value, str):
                    formatted_lines.append(f"• {key.replace('_', ' ').title()}: {value}")
            
            market_parts.append(f"📈 TECHNICAL ANALYSIS for {ticker} (Retrieved from Yahoo Finance):\n" + "\n".join(formatted_lines))

    
    market_context = "\n\n".join(market_parts)
    print("Market", market_context)
    
    # Build news context from articles
    news_context = ""
    if state.news_articles:
        snippets = [
            f"[{doc.metadata.get('source', 'News')} - {doc.metadata.get('date', 'Recent')}]: {doc.page_content[:500]}..."
            for doc in state.news_articles
        ]
        news_context = "📰 RECENT NEWS:\n" + "\n\n".join(snippets)
    
    # Build document context from SEC filings
    def format_doc_source(metadata: dict) -> str:
        """Format document metadata into readable source description."""
        ticker = metadata.get("company_ticker", "Unknown")
        company = metadata.get("company_name", ticker)
        form = metadata.get("form_type", "Filing")
        period = metadata.get("period_end", "N/A")
        section = metadata.get("section_title", "Document")
        return f"{company} ({ticker}) {form} (Period: {period}) - {section}"

    doc_context = "\n\n".join([
        f"[{format_doc_source(d.metadata)}]: {d.page_content[:800]}..."
        for d in state.retrieved_docs
    ])

    # Build conversation history for context
    history_str = ""
    if len(state.messages) > 1:
        history_msgs = state.messages[:-1]  # Exclude current message
        history_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history_msgs
        ])
    
    # Generate comprehensive answer using LLM
    llm = get_llm(temperature=0.3)  # Balanced creativity for analysis

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a FinInsightBot, a professional AI financial analyst. Use the document excerpts, live market data, and recent news to provide a precise, professional, and well-grounded response."
         "If financial metrics like P/E ratio or EPS are already provided in the data, report them directly — do not attempt to recalculate unless explicitly asked."),
        ("human", "Conversation History:\n{history}\n\nQuestion: {question}\n\nRelevant Documents:\n{doc_context}\n\n{market_context}\n\n{news_context}\n\nAnswer:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "history": history_str or "None",
            "question": state.query,
            "doc_context": doc_context,
            "market_context": market_context.strip(),
            "news_context": news_context.strip()
        })
    except Exception as e:
        answer = f"An error occurred: {str(e)}"
    
    return {"answer": answer}