from pathlib import Path
from app.agents.financial_agent_graph import financial_rag_agent
from dotenv import load_dotenv
import sys

# Add project root to Python path so 'agents' is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

def test_multi_turn():
    print("\n🔄 Testing Multi-Turn Memory...\n")
    
    # Turn 1
    resp1 = financial_rag_agent([{"role": "user", "content": "What is Apple's market cap?"}])
    print("💬 Turn 1:", resp1["answer"][:100] + "...")
    
    # Turn 2 — follow-up
    resp2 = financial_rag_agent([
        {"role": "user", "content": "What is Apple's market cap?"},
        {"role": "assistant", "content": resp1["answer"]},
        {"role": "user", "content": "And its P/E ratio?"}  # should resolve to Apple
    ])
    print("💬 Turn 2:", resp2["answer"][:100] + "...")

if __name__ == "__main__":
    test_multi_turn()