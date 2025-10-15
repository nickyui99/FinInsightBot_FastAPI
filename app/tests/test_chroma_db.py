"""
Focused ChromaDB test script to diagnose retrieval issues.

Tests:
1. ChromaDB connection and collection status
2. Document count and metadata inspection
3. Direct similarity search
4. Retriever interface (as used by your agent)
5. Embedding verification
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
PERSIST_DIR = "./app/app/chroma_db"
COLLECTION_NAME = "magnificent7_filings"

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def test_chromadb_connection():
    """Test 1: Basic ChromaDB connection and collection info."""
    print_section("TEST 1: ChromaDB Connection & Collection Info")
    
    try:
        # Load environment
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in environment")
            return None
        
        print(f"✅ API Key loaded: {api_key[:10]}...")
        
        # Check if directory exists
        if not os.path.exists(PERSIST_DIR):
            print(f"❌ ChromaDB directory not found: {PERSIST_DIR}")
            print("💡 Run 'python app/scripts/ingest.py' first to create the database")
            return None
        
        print(f"✅ ChromaDB directory exists: {PERSIST_DIR}")
        
        # Initialize ChromaDB client directly
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        print("✅ ChromaDB client initialized")
        
        # List all collections
        collections = client.list_collections()
        print(f"\n📚 Available collections ({len(collections)}):")
        for coll in collections:
            print(f"   - {coll.name} (count: {coll.count()})")
        
        # Get target collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            print(f"\n✅ Target collection '{COLLECTION_NAME}' found")
            print(f"📊 Total documents: {doc_count}")
            
            if doc_count == 0:
                print("⚠️  WARNING: Collection is empty!")
                print("💡 Run 'python app/scripts/ingest.py' to populate the database")
                return None
            
            # Sample metadata from first 3 documents
            print(f"\n📄 Sample document metadata:")
            sample = collection.get(limit=3, include=["metadatas", "documents"])
            
            for i, (doc_id, metadata) in enumerate(zip(sample['ids'], sample['metadatas']), 1):
                print(f"\n   Document {i} (ID: {doc_id}):")
                print(f"      Ticker: {metadata.get('company_ticker', 'N/A')}")
                print(f"      Company: {metadata.get('company_name', 'N/A')}")
                print(f"      Form: {metadata.get('form_type', 'N/A')}")
                print(f"      Period: {metadata.get('period_end', 'N/A')}")
                print(f"      Section: {metadata.get('section_title', 'N/A')[:60]}...")
            
            return api_key
            
        except Exception as e:
            print(f"❌ Collection '{COLLECTION_NAME}' not found: {e}")
            print(f"💡 Available collections: {[c.name for c in collections]}")
            return None
            
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_langchain_vectorstore(api_key: str):
    """Test 2: LangChain Chroma wrapper initialization."""
    print_section("TEST 2: LangChain Chroma Wrapper")
    
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        print("✅ Google embeddings initialized")
        
        # Initialize Chroma vectorstore (LangChain wrapper)
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("✅ LangChain Chroma vectorstore initialized")
        
        # Check document count via LangChain interface
        try:
            count = vectorstore._collection.count()
            print(f"📊 Document count via LangChain: {count}")
        except Exception as e:
            print(f"⚠️  Could not get count via LangChain: {e}")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ LangChain vectorstore initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_similarity_search(vectorstore):
    """Test 3: Direct similarity search."""
    print_section("TEST 3: Similarity Search")
    
    test_queries = [
        "What are Apple's main risk factors?",
        "Tesla revenue and earnings",
        "Microsoft cloud business performance",
        "NVDA AI chip sales"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        try:
            results = vectorstore.similarity_search(query, k=3)
            
            if not results:
                print("   ⚠️  No results returned")
                continue
            
            print(f"   ✅ Found {len(results)} results:")
            
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                ticker = metadata.get('company_ticker', 'N/A')
                company = metadata.get('company_name', 'N/A')
                section = metadata.get('section_title', 'N/A')[:50]
                content_preview = doc.page_content[:150].replace('\n', ' ')
                
                print(f"\n   Result {i}:")
                print(f"      Company: {company} ({ticker})")
                print(f"      Section: {section}...")
                print(f"      Content: {content_preview}...")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()

def test_retriever_interface(vectorstore):
    """Test 4: Retriever interface (as used by your agent)."""
    print_section("TEST 4: Retriever Interface (Agent Method)")
    
    try:
        # Create retriever with same config as your agent
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ Retriever created with k=4")
        
        # Test query
        query = "What are Apple's revenue trends?"
        print(f"\n🔎 Testing query: '{query}'")
        
        # Test different retrieval methods
        print("\n📋 Testing retrieval methods:")
        
        # Method 1: get_relevant_documents (standard)
        if hasattr(retriever, "get_relevant_documents"):
            print("\n   Method: get_relevant_documents()")
            try:
                docs = retriever.get_relevant_documents(query)
                print(f"   ✅ Retrieved {len(docs)} documents")
                
                if docs:
                    print(f"\n   First result:")
                    print(f"      Ticker: {docs[0].metadata.get('company_ticker', 'N/A')}")
                    print(f"      Section: {docs[0].metadata.get('section_title', 'N/A')[:60]}...")
                    print(f"      Content preview: {docs[0].page_content[:200]}...")
                else:
                    print("   ⚠️  Empty results returned")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Method 2: invoke (alternative)
        if hasattr(retriever, "invoke"):
            print("\n   Method: invoke()")
            try:
                docs = retriever.invoke(query)
                print(f"   ✅ Retrieved {len(docs)} documents")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ Retriever creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_embedding_functionality(api_key: str):
    """Test 5: Verify embedding generation."""
    print_section("TEST 5: Embedding Generation Test")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Test embedding a simple query
        test_text = "What is Apple's P/E ratio?"
        print(f"🔎 Generating embedding for: '{test_text}'")
        
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding generated successfully")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Verify it's not all zeros
        if all(v == 0 for v in embedding):
            print("   ⚠️  WARNING: Embedding is all zeros!")
        else:
            print("   ✅ Embedding contains non-zero values")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all ChromaDB diagnostic tests."""
    print("\n" + "="*80)
    print("🚀 ChromaDB Diagnostic Test Suite")
    print("="*80)
    
    # Test 1: Connection
    api_key = test_chromadb_connection()
    if not api_key:
        print("\n❌ Cannot proceed - ChromaDB connection failed")
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("   1. Ensure .env file exists with GOOGLE_API_KEY")
        print("   2. Run: python app/scripts/ingest.py")
        print("   3. Verify ./app/chroma_db directory exists")
        return
    
    # Test 2: LangChain wrapper
    vectorstore = test_langchain_vectorstore(api_key)
    if not vectorstore:
        print("\n❌ Cannot proceed - Vectorstore initialization failed")
        return
    
    # Test 3: Similarity search
    test_similarity_search(vectorstore)
    
    # Test 4: Retriever interface
    test_retriever_interface(vectorstore)
    
    # Test 5: Embeddings
    test_embedding_functionality(api_key)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
    
    print("\n📊 SUMMARY:")
    print("   If all tests passed, your ChromaDB is working correctly.")
    print("   If retrieve_documents() still returns empty in your agent:")
    print("   1. Check if state.needs_secfiling is True")
    print("   2. Check if state.is_financial is True")
    print("   3. Verify state.query is being passed correctly")
    print("   4. Check logs in app/agents/nodes/data_fetching.py")

if __name__ == "__main__":
    run_all_tests()