# test_chroma_simple.py
import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CONFIG ---
PERSIST_DIR = "./app/chroma_db"
COLLECTION_NAME = "magnificent7_filings"

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: Missing GOOGLE_API_KEY in .env")
        sys.exit(1)

    print("🔍 Loading ChromaDB...")
    
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )
        
        # 1. Check document count
        count = vectorstore._collection.count()
        print(f"✅ DB loaded. Total documents: {count}")
        
        if count == 0:
            print("⚠️  WARNING: No documents found. Did ingestion run?")
            return

        # 2. Test a simple query
        print("\n🔍 Running test query: 'Apple risk factors'")
        results = vectorstore.similarity_search("Apple risk factors", k=2)
        
        if not results:
            print("❌ No results retrieved.")
            return

        print(f"✅ Retrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            ticker = meta.get("company_ticker", "N/A")
            section = meta.get("section_title", "N/A")[:50]
            preview = doc.page_content[:120].replace('\n', ' ')
            print(f"  {i}. [{ticker}] {section} → {preview}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Check:")
        print(f"   - Does '{PERSIST_DIR}' exist?")
        print(f"   - Was DB created with SAME collection_name: '{COLLECTION_NAME}'?")
        print("   - Is GOOGLE_API_KEY valid?")

if __name__ == "__main__":
    main()