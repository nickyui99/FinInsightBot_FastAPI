# ingest_summarized.py
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain.docstore.document import Document
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local imports
from utils.doc_summarizer import extract_summarize_and_chunk_pdf

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()

# Validate API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

PERSIST_DIR = "./app/chroma_db"
DATA_DIR = "./documents"
COLLECTION_NAME = "magnificent7_filings"

# Ticker to company name mapping (Magnificent 7)
TICKER_TO_NAME = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc."
}

# ----------------------------
# METADATA PARSER
# ----------------------------
def parse_metadata_from_filename(filename: str) -> dict:
    """
    Parse metadata from filename like 'AAPL_10K_2023.pdf' or 'NVDA_10Q_2023Q3.pdf'
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")
    
    ticker = parts[0].upper()
    company_name = TICKER_TO_NAME.get(ticker, ticker)
    
    # Determine form type
    form_raw = parts[1].upper()
    if "10K" in form_raw:
        form_type = "10-K"
    elif "10Q" in form_raw:
        form_type = "10-Q"
    else:
        form_type = "Filing"
    
    # Extract period (e.g., "2023" or "2023Q3")
    period_end = parts[2] if len(parts) > 2 else "unknown"
    
    return {
        "company_ticker": ticker,
        "company_name": company_name,
        "form_type": form_type,
        "period_end": period_end,
        "source_file": filename
    }

# ----------------------------
# MAIN INGESTION
# ----------------------------
def main():
    print("🚀 Starting ingestion with Gemini embeddings...")
    
    # Optional: Clear old DB (uncomment during development)
    # if os.path.exists(PERSIST_DIR):
    #     shutil.rmtree(PERSIST_DIR)
    #     print(f"🗑️  Cleared old DB at {PERSIST_DIR}")

    # Initialize embedding model
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialize Chroma with PersistentClient (fixes socket error)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )
    
    total_chunks = 0
    processed_files = 0
    
    # Walk through all PDFs
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".pdf"):
                continue
                
            pdf_path = os.path.join(root, file)
            print(f"\n📄 Processing: {file}")
            
            try:
                # Parse metadata
                base_meta = parse_metadata_from_filename(file)
                print(f"   → Ticker: {base_meta['company_ticker']}, Form: {base_meta['form_type']}")
                
                # Extract and summarize
                summarized_sections = extract_summarize_and_chunk_pdf(pdf_path)
                if not summarized_sections:
                    print(f"   ⚠️  No sections extracted from {file}")
                    continue
                
                # Create LangChain documents
                documents = []
                for sec in summarized_sections:
                    meta = {
                        **base_meta,
                        "section_title": sec["section_title"],
                        "original_length": sec["original_length"]
                    }
                    doc = Document(page_content=sec["summary"], metadata=meta)
                    documents.append(doc)
                
                if documents:
                    vectorstore.add_documents(documents)
                    added = len(documents)
                    total_chunks += added
                    processed_files += 1
                    print(f"   ✅ Added {added} summarized sections")
                else:
                    print(f"   ⚠️  No valid documents from {file}")
                    
            except Exception as e:
                print(f"   ❌ Failed on {file}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n🎉 Ingestion complete!")
    print(f"   Files processed: {processed_files}")
    print(f"   Total chunks added: {total_chunks}")
    print(f"   DB location: {PERSIST_DIR}")

if __name__ == "__main__":
    main()