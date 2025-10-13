from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_retriever():
    """Initialize and return a Chroma retriever with Google embeddings."""
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")  
    )
    vectorstore = Chroma(
        persist_directory="./app/chroma_db",
        embedding_function=embedding,
        collection_name="magnificent7_filings"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})