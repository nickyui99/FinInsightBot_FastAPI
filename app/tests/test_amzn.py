"""
Debug script to inspect PDF structure and test extraction
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.doc_summarizer import extract_summarize_and_chunk_pdf

def inspect_pdf_structure(pdf_path: str):
    """Inspect PDF structure to diagnose extraction issues."""
    print(f"\n{'='*80}")
    print(f"🔍 Inspecting: {Path(pdf_path).name}")
    print(f"{'='*80}\n")
    
    try:
        doc = fitz.open(pdf_path)
        
        print(f"📊 Basic Info:")
        print(f"   Pages: {len(doc)}")
        print(f"   Metadata: {doc.metadata}")
        
        # Check first few pages
        print(f"\n📄 First 3 pages content preview:")
        for i in range(min(3, len(doc))):
            page = doc[i]
            text = page.get_text()
            print(f"\n   Page {i+1} ({len(text)} chars):")
            print(f"   {text[:500]}...")
            
            # Check for common section headers
            common_headers = [
                "Item 1", "Item 2", "Item 3",
                "BUSINESS", "RISK FACTORS", "MANAGEMENT",
                "Part I", "Part II", "Part III"
            ]
            found_headers = [h for h in common_headers if h.lower() in text.lower()]
            if found_headers:
                print(f"   ✅ Found headers: {found_headers}")
            else:
                print(f"   ⚠️  No standard headers detected")
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Failed to inspect PDF: {e}")
        import traceback
        traceback.print_exc()

def test_extraction(pdf_path: str):
    """Test the actual extraction function."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing Extraction")
    print(f"{'='*80}\n")
    
    try:
        sections = extract_summarize_and_chunk_pdf(pdf_path)
        
        if not sections:
            print("❌ No sections extracted!")
            print("\n💡 Possible reasons:")
            print("   1. PDF text is encoded/encrypted")
            print("   2. Section headers not matching expected patterns")
            print("   3. PDF is image-based (needs OCR)")
            print("   4. Text extraction returning empty strings")
        else:
            print(f"✅ Extracted {len(sections)} sections:")
            for i, sec in enumerate(sections[:3], 1):
                print(f"\n   Section {i}:")
                print(f"      Title: {sec.get('section_title', 'N/A')}")
                print(f"      Original length: {sec.get('original_length', 0)} chars")
                print(f"      Summary length: {len(sec.get('summary', ''))} chars")
                print(f"      Summary preview: {sec.get('summary', '')[:200]}...")
    
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pdf_path = "./app/documents/AMZN_10K_2024.pdf"
    
    # First inspect PDF structure
    inspect_pdf_structure(pdf_path)
    
    # Then test extraction
    test_extraction(pdf_path)
    
    print(f"\n{'='*80}")
    print("📋 Next Steps:")
    print("='*80}")
    print("1. Check if PDF text is readable (not image-based)")
    print("2. Verify section header patterns in doc_summarizer.py")
    print("3. Check if PDF has unusual formatting/encryption")
    print("4. Try with a different PDF to isolate the issue")