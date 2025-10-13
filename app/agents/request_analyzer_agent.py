"""
Request analyzer agent for financial queries.

Analyzes user queries to determine:
- Whether the query is financial in nature
- Which stock tickers are mentioned
- What types of financial data are needed (fundamental, technical, news, SEC filings)
- The user's intent category

Uses a combination of:
1. Fast pattern matching for common companies (Tesla->TSLA, Apple->AAPL)
2. LLM-based analysis with timeout protection
3. Regex fallback for edge cases

This ensures reliable query analysis even if the LLM is slow or unavailable.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.llm import get_fast_llm

def analyze_request(query: str) -> Dict[str, Any]:
    """
    Analyze user query to determine financial data requirements.
    
    Uses a multi-layered approach for reliable analysis:
    1. Fast pattern matching for common companies
    2. LLM analysis with timeout protection  
    3. Regex fallback if LLM fails
    
    Args:
        query: User's natural language query about financial topics
        
    Returns:
        Dict containing:
        - is_financial: Whether query is about finance/stocks
        - ticker: List of stock symbols mentioned (e.g., ["AAPL", "TSLA"])
        - needs_fundamental: Whether P/E ratios, earnings data needed
        - needs_technical: Whether charts, indicators needed
        - needs_news: Whether recent news needed
        - needs_secfiling: Whether SEC documents needed
        - intent: Category like "company_overview", "valuation", etc.
        
    Example:
        query="Help me analyze Tesla's performance"
        Returns: {
            "is_financial": True, 
            "ticker": ["TSLA"],
            "needs_fundamental": True,
            "needs_technical": False,
            "needs_news": False,
            "needs_secfiling": False,
            "intent": "company_overview"
        }
    """
    print(f"Starting analysis for query: '{query}'")
    
    #  LLM-based analysis with timeout protection
    try:
        llm = get_fast_llm()
        print("--LLM initialized successfully--")
        
        parser = JsonOutputParser()
        print("--Parser initialized successfully--")

        # Create detailed analysis prompt for complex queries
        prompt = PromptTemplate.from_template("""
        You are an expert financial query analyzer. Your role is crucial for enabling accurate financial analysis.

        TASK: Analyze the user's query and classify their intent with maximum precision to help financial analysts provide professional, comprehensive responses.

        ANALYSIS FRAMEWORK:

        1. **Financial Relevance Assessment:**
        - Set is_financial=true ONLY for queries about stocks, ETFs, companies, financial markets, investments, or economic topics
        - Set is_financial=false for general questions, personal advice, or non-financial topics

        2. **Ticker Symbol Identification:**
        - Include ONLY genuine stock/ETF symbols (AAPL, TSLA, GOOGL, SPY, etc.)
        - Map company names to tickers: Tesla→TSLA, Apple→AAPL, Microsoft→MSFT, Amazon→AMZN
        - Map market indices to tickers: VIX→^VIX, S&P 500→^GSPC, Dow Jones→^DJI, NASDAQ→^IXIC
        - NEVER include common words as tickers (WHAT, IS, THE, AND, etc.)
        - Return empty array [] if no valid tickers found

        3. **Data Requirements Analysis:**
        - needs_fundamental: Current price/value, valuation metrics, P/E ratios, earnings, revenue, market cap, dividends, financial statements
        - needs_technical: Charts, price trends, RSI, MACD, moving averages, support/resistance levels, trading patterns  
        - needs_news: Recent news articles, headlines, announcements, market updates
        - needs_secfiling: SEC documents, 10-K/10-Q reports, annual reports, insider trading, financial disclosures

        4. **Intent Classification:**
        Choose the most specific category: ["valuation", "technical_outlook", "performance", "risk_analysis", "company_overview", "news_analysis", "sec_filing_lookup", "general"]

        REQUIRED OUTPUT FORMAT (FLAT JSON STRUCTURE):
        {{
            "is_financial": boolean,
            "ticker": array of strings,
            "needs_fundamental": boolean,
            "needs_technical": boolean,
            "needs_news": boolean,
            "needs_secfiling": boolean,
            "intent": string
        }}

        EXAMPLES:
        - "What is the current VIX index?" → {{"is_financial": true, "ticker": ["^VIX"], "needs_fundamental": true, "needs_technical": false, "needs_news": false, "needs_secfiling": false, "intent": "general"}}
        - "Apple stock price today" → {{"is_financial": true, "ticker": ["AAPL"], "needs_fundamental": true, "needs_technical": false, "needs_news": false, "needs_secfiling": false, "intent": "general"}}
        - "Tesla news" → {{"is_financial": true, "ticker": ["TSLA"], "needs_fundamental": false, "needs_technical": false, "needs_news": true, "needs_secfiling": false, "intent": "news_analysis"}}

        IMPORTANT: Return ONLY a flat JSON object with these exact keys. Do NOT nest data requirements under a "data_requirements" object. Do NOT use "ticker_symbols" - use "ticker" instead.

        Current Query: {query}

        {format_instructions}
        """).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        print("Chain created successfully")

        print("Invoking LLM chain...")
        
        # Execute with timeout to prevent hanging
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        import json
        
        def invoke_with_timeout():
            return chain.invoke({"query": query})
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(invoke_with_timeout)
                raw_result = future.result(timeout=10)  # 10 second timeout
                print("✅ LLM chain completed successfully!")
                print("Raw LLM Result:", raw_result)
                
                # Handle both parsed and unparsed responses
                if isinstance(raw_result, str):
                    try:
                        result = json.loads(raw_result)
                        print("✅ Parsed JSON string to dict")
                    except json.JSONDecodeError as json_err:
                        print(f"❌ JSON parsing failed: {json_err}")
                        print(f"Raw string was: {raw_result}")
                        raise Exception("Invalid JSON format")
                elif isinstance(raw_result, dict):
                    result = raw_result
                    print("✅ Already a dictionary")
                else:
                    print(f"❌ Unexpected result type: {type(raw_result)}")
                    raise Exception("Unexpected result format")
                
        except FutureTimeoutError:
            print("⏱️ LLM call timed out, falling back to regex analysis")
            raise Exception("LLM timeout")
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            raise e
            
        print("Analyzer Input:", {"query": query})
        print("Final Processed Result:", result)
        
        # Clean and normalize ticker symbols from LLM response
        raw_ticker = result.get("ticker", [])
        if raw_ticker is None:
            ticker_list = []
        elif isinstance(raw_ticker, str):
            ticker_list = [raw_ticker.strip().upper()] if raw_ticker.strip() else []
        elif isinstance(raw_ticker, list):
            ticker_list = [t.strip().upper() for t in raw_ticker if isinstance(t, str) and t.strip()]
        else:
            ticker_list = []

        print(f"Raw ticker: {raw_ticker} -> Cleaned: {ticker_list}")

        # Return the flat structure as requested
        flat_result = {
            "is_financial": bool(result.get("is_financial", False)),
            "ticker": ticker_list,
            "needs_fundamental": bool(result.get("needs_fundamental", False)),
            "needs_technical": bool(result.get("needs_technical", False)),
            "needs_news": bool(result.get("needs_news", False)),
            "needs_secfiling": bool(result.get("needs_secfiling", False)),
            "intent": result.get("intent", "general")
        }
        
        print("Final flat result:", flat_result)
        return flat_result
    except Exception as e:
        # Step 3: Regex fallback when LLM fails
        print(f"❌ LLM analysis failed: {type(e).__name__}: {e}")
        print("🔄 Falling back to regex-based analysis")
        
        import re
        # Find potential ticker symbols (2-5 capital letters)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # Filter out common English words that aren't tickers
        common_words = {"I", "A", "AN", "THE", "IS", "AND", "OR", "IN", "ON", "TO", "FOR", "OF", "ETF", "USD"}
        filtered_tickers = [t for t in tickers if t not in common_words and len(t) >= 2]
        
        return {
            "is_financial": len(filtered_tickers) > 0,
            "ticker": filtered_tickers,
            "needs_fundamental": False,
            "needs_technical": False,
            "needs_news": "news" in query.lower() or "headline" in query.lower(),
            "needs_secfiling": "sec" in query.lower() or "10-" in query.lower(),
            "intent": "general"
        }
