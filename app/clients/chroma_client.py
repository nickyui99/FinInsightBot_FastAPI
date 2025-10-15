"""
ChromaDB client for vector storage and document retrieval.

Provides functionality for creating and managing vector databases
using ChromaDB with Google Generative AI embeddings. Used for
storing and retrieving SEC filings and financial documents.

Key Features:
- Google Generative AI embeddings for semantic search
- Persistent storage using ChromaDB
- Document vectorization and storage
- Semantic similarity search capabilities
"""

from typing import List, Optional
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from config import EMBEDDING_MODEL, PERSIST_DIR

def create_vectorstore(documents: List[Document], persist_dir: str = PERSIST_DIR) -> Chroma:
    """
    Create a ChromaDB vector store from documents.
    
    Converts documents into vector embeddings using Google's embedding model
    and stores them in a persistent ChromaDB database for semantic search.
    
    Args:
        documents: List of LangChain Document objects to vectorize
        persist_dir: Directory path for persistent storage (default from config)
        
    Returns:
        Chroma: Configured ChromaDB vector store instance
        
    Example:
        docs = [Document(page_content="Apple Q3 earnings...", metadata={"ticker": "AAPL"})]
        vectorstore = create_vectorstore(docs)
        
    Note:
        Requires GOOGLE_API_KEY environment variable for embeddings API
    """
    # Initialize Google embeddings with configured model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create vector store from documents with persistent storage
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vectorstore