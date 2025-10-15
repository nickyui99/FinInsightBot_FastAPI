# ingest_summarized.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.docstore.document import Document
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Local imports
from app.utils.doc_summarizer import extract_summarize_and_chunk_pdf

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()

# Validate API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

PERSIST_DIR = "./app/chroma_db"
DATA_DIR = "./app/documents"
COLLECTION_NAME = "sec_filings"

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
    print("🚀 Starting FAST ingestion with batch processing...")
    print(f"📂 Data directory: {DATA_DIR}")
    print(f"💾 Persist directory: {PERSIST_DIR}")
    
    # Optional: Clear old DB (uncomment during development)
    # if os.path.exists(PERSIST_DIR):
    #     import shutil
    #     shutil.rmtree(PERSIST_DIR)
    #     print(f"🗑️  Cleared old DB at {PERSIST_DIR}")

    # Initialize embedding model
    print("🔧 Initializing embeddings...")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialize Chroma with PersistentClient
    print("🔧 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )
    
    total_chunks = 0
    processed_files = 0
    failed_files = []
    
    # Collect all PDFs first
    pdf_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    
    print(f"\n📊 Found {len(pdf_files)} PDF files to process\n")
    
    # Process each PDF with batch insertion
    BATCH_SIZE = 50  # Insert in batches for speed
    document_batch = []
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        file = os.path.basename(pdf_path)
        print(f"[{idx}/{len(pdf_files)}] 📄 Processing: {file}")
        
        try:
            # Parse metadata
            base_meta = parse_metadata_from_filename(file)
            print(f"   → {base_meta['company_ticker']} | {base_meta['form_type']} | {base_meta['period_end']}")
            
            # Extract and chunk (NO summarization - much faster!)
            chunks = extract_summarize_and_chunk_pdf(pdf_path)
            if not chunks:
                print(f"   ⚠️  No chunks extracted from {file}")
                failed_files.append((file, "No chunks"))
                continue
            
            # Create LangChain documents
            for chunk_data in chunks:
                meta = {
                    **base_meta,
                    "section_title": chunk_data["section_title"],
                    "original_length": chunk_data["original_length"]
                }
                doc = Document(page_content=chunk_data["summary"], metadata=meta)
                document_batch.append(doc)
            
            print(f"   ✅ Prepared {len(chunks)} chunks (batch: {len(document_batch)})")
            
            # Batch insert when we hit BATCH_SIZE
            if len(document_batch) >= BATCH_SIZE:
                print(f"   💾 Inserting batch of {len(document_batch)} documents...")
                vectorstore.add_documents(document_batch)
                total_chunks += len(document_batch)
                document_batch = []  # Clear batch
                print(f"   ✅ Batch inserted (Total: {total_chunks})")
            
            processed_files += 1
            
        except Exception as e:
            print(f"   ❌ Failed on {file}: {e}")
            failed_files.append((file, str(e)))
            import traceback
            traceback.print_exc()
    
    # Insert remaining documents in batch
    if document_batch:
        print(f"\n💾 Inserting final batch of {len(document_batch)} documents...")
        vectorstore.add_documents(document_batch)
        total_chunks += len(document_batch)
    
    # Summary
    print("\n" + "="*80)
    print("✅ INGESTION COMPLETE")
    print("="*80)
    print(f"📊 Files processed: {processed_files}/{len(pdf_files)}")
    print(f"📦 Total chunks: {total_chunks}")
    print(f"💾 Stored in: {PERSIST_DIR}")
    print(f"🗂️  Collection: {COLLECTION_NAME}")
    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for fname, reason in failed_files:
            print(f"   - {fname}: {reason}")
    
    print("="*80 + "\n")
    
    print(f"\n🎉 Ingestion complete!")
    print(f"   Files processed: {processed_files}")
    print(f"   Total chunks added: {total_chunks}")
    print(f"   DB location: {PERSIST_DIR}")

if __name__ == "__main__":
    main()