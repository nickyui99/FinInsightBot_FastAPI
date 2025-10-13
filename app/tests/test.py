# test_agent_direct.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path so 'agents' is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import your agent
from app.agents.financial_agent_graph import financial_rag_agent


def test_financial_agent():
    print("🔍 Testing Financial RAG Agent (with memory support)...\n")

    test_queries = [
        "Is the stock market overvalued?",
        # "What is Apple's current P/E ratio?",
        # "How is Tesla trading today?",
        # "What are the main risks in Microsoft's latest 10-K?",
        # "What's the technical outlook for NVIDIA stock?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"🧪 Test {i}: {query}\n")
        try:
            # ✅ Pass as list of message dicts (required by LangGraph state)
            result = financial_rag_agent([
                {"role": "user", "content": query}
            ])
            
            print(f"✅ Ticker: {result.get('ticker')}")
            print(f"💬 Answer:\n{result.get('answer')}\n")
            
            market_data = result.get("market_data", {})
            if market_data and not any("error" in str(v) for v in market_data.values()):
                print(f"📈 Market Data: {market_data}\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
        print("-" * 80 + "\n")


if __name__ == "__main__":
    test_financial_agent()