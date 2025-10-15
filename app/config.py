import os
from pathlib import Path
from typing import Set
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Document types
    SUPPORTED_DOC_TYPES: Set[str] = {
        "financial_report",
        "earnings_transcript", 
        "news_article",
        "research_paper",
        "general"
    }

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")

    # LLM Configuration
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"
    EMBEDDING_MODEL = "models/embedding-001"
    
    # LLM Parameters
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERP_API_KEY = os.getenv("SERP_API_KEY_2")
    
    # Session Configuration
    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))
    MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Caching
    ENABLE_LLM_CACHE = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Performance
    ASYNC_TIMEOUT_SECONDS = int(os.getenv("ASYNC_TIMEOUT_SECONDS", "30"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate configuration"""
        required_vars = ["GOOGLE_API_KEY"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return True

# Singleton instance
config = Config()
config.validate()