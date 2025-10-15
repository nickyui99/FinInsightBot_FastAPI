# doc_summarizer.py
import os
import sys
from dotenv import load_dotenv
import fitz  # PyMuPDF - faster than unstructured
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Text splitter for chunking (no summarization for speed)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def extract_summarize_and_chunk_pdf(pdf_path: str) -> List[Dict]:
    """
    Fast extraction using PyMuPDF with intelligent section detection.
    No LLM summarization - direct chunking for speed.
    
    Returns list of chunks with section metadata.
    """
    print(f"   📖 Extracting text from PDF...")
    
    # Step 1: Fast text extraction with PyMuPDF
    doc = fitz.open(pdf_path)
    page_count = len(doc)  # Get page count before closing
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    
    if not full_text.strip():
        print(f"   ⚠️  PDF appears to be empty or image-based")
        return []
    
    print(f"   ✅ Extracted {len(full_text):,} characters from {page_count} pages")
    
    # Step 2: Detect SEC filing sections with flexible patterns
    section_patterns = [
        # Standard SEC patterns
        r"(?:^|\n)\s*Item\s+(\d+[A-Z]?)\.\s+([^\n]{5,100})",  # Item 1. Business
        r"(?:^|\n)\s*ITEM\s+(\d+[A-Z]?)\.\s+([^\n]{5,100})",  # ITEM 1. BUSINESS
        r"(?:^|\n)\s*Part\s+([IVX]+)\s*[:\-]\s*([^\n]{5,100})",  # Part I - Description
        r"(?:^|\n)\s*Table of Contents",  # TOC marker
    ]
    
    # Find all section markers
    section_matches = []
    for pattern in section_patterns[:2]:  # Focus on Item patterns
        matches = list(re.finditer(pattern, full_text, re.MULTILINE | re.IGNORECASE))
        if matches:
            for match in matches:
                section_matches.append({
                    'start': match.start(),
                    'item': match.group(1) if len(match.groups()) >= 1 else "0",
                    'title': match.group(2).strip() if len(match.groups()) >= 2 else "Section"
                })
    
    # Sort by position in document
    section_matches = sorted(section_matches, key=lambda x: x['start'])
    
    print(f"   ✅ Detected {len(section_matches)} sections")
    
    # Step 3: Split document by sections or chunk intelligently
    if len(section_matches) >= 2:
        # We have clear sections - split by them
        chunks_with_metadata = []
        
        for i, section in enumerate(section_matches):
            start_pos = section['start']
            end_pos = section_matches[i + 1]['start'] if i + 1 < len(section_matches) else len(full_text)
            
            section_text = full_text[start_pos:end_pos].strip()
            section_title = f"Item {section['item']}: {section['title']}"
            
            # Skip very short sections
            if len(section_text) < 300:
                continue
            
            # Chunk this section if too long
            if len(section_text) > 2000:
                sub_chunks = text_splitter.split_text(section_text)
                for j, chunk in enumerate(sub_chunks):
                    chunks_with_metadata.append({
                        "summary": chunk,
                        "original_length": len(section_text),
                        "section_title": f"{section_title} (Part {j+1}/{len(sub_chunks)})"
                    })
            else:
                chunks_with_metadata.append({
                    "summary": section_text,
                    "original_length": len(section_text),
                    "section_title": section_title
                })
        
        if chunks_with_metadata:
            print(f"   ✅ Created {len(chunks_with_metadata)} section-based chunks")
            return chunks_with_metadata
    
    # Fallback: No clear sections found - intelligent chunking
    print(f"   ⚠️  No clear sections detected, using intelligent chunking")
    chunks = text_splitter.split_text(full_text)
    
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        # Try to extract a title from first line
        first_line = chunk.split('\n')[0][:80].strip()
        title = first_line if len(first_line) > 10 else f"Section {i+1}"
        
        chunks_with_metadata.append({
            "summary": chunk,
            "original_length": len(chunk),
            "section_title": title
        })
    
    print(f"   ✅ Created {len(chunks_with_metadata)} intelligent chunks")
    return chunks_with_metadata

def parse_metadata_from_filename(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    ticker = parts[0].upper()
    form_raw = parts[1].upper()
    period = parts[2] if len(parts) > 2 else "unknown"
    form_type = "10-K" if "10K" in form_raw else "10-Q"
    return {
        "company_ticker": ticker,
        "form_type": form_type,
        "period": period,
        "source_file": filename
    }