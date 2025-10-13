# utils.py
import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from config import GEMINI_FLASH_LITE

load_dotenv()

# Initialize summarizer (Gemini 1.5 Flash is fast & cost-effective)
summarizer = ChatGoogleGenerativeAI(
    model= GEMINI_FLASH_LITE,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    max_tokens=500
)

# Summarization prompt tailored for SEC filings
summary_prompt = PromptTemplate.from_template(
    """You are a financial analyst. Summarize the following section from a SEC 10-K or 10-Q filing.
Focus on key facts, risks, trends, and material information. Be concise and professional.

Section title: {section_title}
Section content:
{content}

Summary (3-5 sentences):"""
)

summarize_chain = summary_prompt | summarizer | StrOutputParser()

def extract_summarize_and_chunk_pdf(pdf_path):
    """
    Returns list of summarized chunks with section-aware metadata
    """
    # Step 1: Extract structured elements
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        infer_table_structure=True
    )
    
    # Step 2: Group into semantic sections (e.g., "Item 1A. Risk Factors")
    sections = chunk_by_title(elements, max_characters=3000, combine_text_under_n_chars=800)
    
    summarized_chunks = []
    
    for section in sections:
        raw_text = str(section)
        if len(raw_text.strip()) < 200:  # Skip tiny sections
            continue
            
        # Try to get section title (Unstructured often captures this)
        section_title = getattr(section, 'title', 'Unknown Section')
        if not section_title or len(section_title) > 100:
            # Fallback: extract first line as title
            section_title = raw_text[:80].split('\n')[0][:60] + "..."

        try:
            # Step 3: Summarize the section
            summary = summarize_chain.invoke({
                "section_title": section_title,
                "content": raw_text[:4000]  # Cap to avoid token limits
            })
            
            summarized_chunks.append({
                "summary": summary,
                "original_length": len(raw_text),
                "section_title": section_title
            })
            
        except Exception as e:
            print(f"⚠️ Summarization failed for section in {pdf_path}: {e}")
            # Fallback: use original text if summarization fails
            summarized_chunks.append({
                "summary": raw_text[:1000],
                "original_length": len(raw_text),
                "section_title": section_title
            })
    
    return summarized_chunks

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