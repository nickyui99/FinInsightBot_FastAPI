"""
Yahoo Finance client for financial data retrieval.

Provides functions to fetch fundamental and technical analysis data
for stocks using the yfinance library. Handles data processing,
technical indicator calculations, and error handling.

Key Features:
- Fundamental data: P/E ratios, market cap, earnings, dividends
- Technical analysis: RSI, MACD, moving averages, Bollinger bands
- Robust error handling with informative fallbacks
- Data validation and cleaning
- 1-year historical data for technical indicators

Dependencies:
- yfinance: Yahoo Finance API wrapper
- ta: Technical analysis library for indicators
"""

import yfinance as yf
from typing import Optional, Dict, Any
from ta.utils import dropna
from ta import add_all_ta_features

# ---------- FUNDAMENTAL DATA ----------
def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch fundamental analysis data for a stock.
    
    Retrieves key financial metrics including valuation ratios,
    market data, and company information from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        
    Returns:
        Dict containing fundamental metrics:
        - current_price: Current stock price
        - market_cap: Market capitalization
        - pe_ratio: Price-to-earnings ratio
        - eps: Earnings per share
        - dividend_yield: Annual dividend yield percentage
        - 52w_high/52w_low: 52-week price range
        - sector: Business sector
        - industry: Specific industry classification
        
        Returns {"error": "message"} if data retrieval fails
        
    Example:
        data = get_fundamental_data("AAPL")
        # Returns: {"current_price": 150.25, "pe_ratio": 28.5, ...}
    """
    try:
        # Fetch company information from Yahoo Finance
        info = yf.Ticker(symbol).info
        
        return {
            "current_price": info.get("currentPrice"),        # Latest stock price
            "market_cap": info.get("marketCap"),             # Market capitalization
            "pe_ratio": info.get("trailingPE"),              # Price-to-earnings ratio
            "eps": info.get("trailingEps"),                  # Earnings per share
            "dividend_yield": info.get("dividendYield"),      # Annual dividend yield
            "52w_high": info.get("fiftyTwoWeekHigh"),        # 52-week high price
            "52w_low": info.get("fiftyTwoWeekLow"),          # 52-week low price
            "sector": info.get("sector"),                    # Business sector
            "industry": info.get("industry")                 # Industry classification
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- TECHNICAL DATA ----------
def get_technical_data(symbol: str) -> Dict[str, Any]:
    """
    Calculate technical analysis indicators for a stock.
    
    Fetches 1-year price history and calculates various technical indicators
    including momentum, trend, and volatility metrics.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        
    Returns:
        Dict containing technical indicators:
        - rsi_14: 14-period Relative Strength Index
        - macd: MACD (Moving Average Convergence Divergence)
        - sma_50: 50-period Simple Moving Average
        - sma_200: 200-period Simple Moving Average  
        - price_vs_sma_200_pct: Price vs 200-day SMA percentage
        - bollinger_upper: Upper Bollinger Band
        - bollinger_lower: Lower Bollinger Band
        
        Returns {"error": "message"} if insufficient data or calculation fails
        
    Example:
        data = get_technical_data("AAPL")
        # Returns: {"rsi_14": 65.2, "macd": 1.25, "sma_50": 148.5, ...}
        
    Note:
        Requires at least 200 days of clean price history for accurate calculations
    """
    try:
        # Fetch 1-year price history from Yahoo Finance
        hist = yf.Ticker(symbol).history(period="1y")
        if hist.empty or len(hist) < 30:
            return {"error": "Insufficient price history"}
        
        # Prepare data for technical analysis
        df = hist.copy()
        df.columns = [c.lower() for c in df.columns]  # Standardize column names
        df = df.dropna()  # Remove rows with missing data
        
        # Ensure sufficient data for reliable technical indicators
        if len(df) < 200:
            return {"error": "Insufficient clean price history (<200 days)"}
        
        # Calculate all technical indicators using the 'ta' library
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
        
        # Extract the most recent values
        latest = df.iloc[-1]
        current_price = latest["close"]
        sma_200 = latest.get("trend_sma_slow")
        
        return {
            "rsi_14": latest.get("momentum_rsi"),             # RSI indicator
            "macd": latest.get("trend_macd"),                 # MACD line
            "sma_50": latest.get("trend_sma_fast"),           # 50-day SMA
            "sma_200": sma_200,                              # 200-day SMA
            "price_vs_sma_200_pct": (                       # Price vs 200-day SMA %
                (current_price / sma_200 - 1) * 100 if sma_200 else None
            ),
            "bollinger_upper": latest.get("volatility_bbh"), # Upper Bollinger Band
            "bollinger_lower": latest.get("volatility_bbl")  # Lower Bollinger Band
        }
    except Exception as e:
        return {"error": str(e)}
