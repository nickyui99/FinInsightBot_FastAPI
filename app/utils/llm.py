# optimizations/optimized_llm.py
import os
import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache
from config import config

# Enable caching if configured
if config.ENABLE_LLM_CACHE:
    set_llm_cache(InMemoryCache())

class LLMPool:
    """Connection pool for LLM instances to avoid recreating them"""
    
    def __init__(self):
        self._pools: Dict[str, ChatGoogleGenerativeAI] = {}
        self._lock = asyncio.Lock()
    
    async def get_llm(
        self, 
        model: str = config.GEMINI_FLASH, 
        temperature: float = config.TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> ChatGoogleGenerativeAI:
        """Get or create an LLM instance with connection pooling"""
        
        # Create a key for the pool
        pool_key = f"{model}_{temperature}_{max_tokens}"
        
        async with self._lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = ChatGoogleGenerativeAI(
                    model=model,
                    api_key=config.GOOGLE_API_KEY,
                    temperature=temperature,
                    max_tokens=max_tokens or config.MAX_TOKENS,
                    timeout=config.ASYNC_TIMEOUT_SECONDS
                )
            
            return self._pools[pool_key]
    
    def clear_pool(self):
        """Clear the connection pool"""
        self._pools.clear()

# Global LLM pool instance
llm_pool = LLMPool()

# Backward compatibility function
@lru_cache(maxsize=10)
def get_llm(
    model: str = config.GEMINI_FLASH, 
    temperature: float = config.TEMPERATURE, 
    max_tokens: Optional[int] = None
) -> ChatGoogleGenerativeAI:
    """
    Get LLM instance with caching for synchronous usage.
    For async usage, prefer LLMPool.get_llm()
    """
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens or config.MAX_TOKENS,
        timeout=config.ASYNC_TIMEOUT_SECONDS
    )

# Async wrapper
async def get_llm_async(
    model: str = config.GEMINI_FLASH,
    temperature: float = config.TEMPERATURE,
    max_tokens: Optional[int] = None
) -> ChatGoogleGenerativeAI:
    """Async version using connection pool"""
    return await llm_pool.get_llm(model, temperature, max_tokens)

# Model-specific helpers
def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Get fast model for quick operations"""
    return get_llm(model=config.GEMINI_FLASH_LITE, temperature=0.0)

def get_smart_llm() -> ChatGoogleGenerativeAI:
    """Get smart model for complex operations"""  
    return get_llm(model=config.GEMINI_PRO, temperature=config.TEMPERATURE)

async def get_fast_llm_async() -> ChatGoogleGenerativeAI:
    """Get fast model for quick operations (async)"""
    return await get_llm_async(model=config.GEMINI_FLASH_LITE, temperature=0.0)

async def get_smart_llm_async() -> ChatGoogleGenerativeAI:
    """Get smart model for complex operations (async)"""
    return await get_llm_async(model=config.GEMINI_PRO, temperature=config.TEMPERATURE)