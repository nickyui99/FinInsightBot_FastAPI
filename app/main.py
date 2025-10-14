import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uuid import uuid4
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your LangGraph graph
from agents.financial_agent_graph import financial_graph

# Initialize FastAPI app
app = FastAPI(
    title="FinInsightBot Agent API",
    description="LangGraph-powered streaming financial assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)

# In-memory session storage: { session_id: FinancialAgentState dict }
# ⚠️ For production, use Redis or a database with TTL
sessions: Dict[str, Dict[str, Any]] = {}

# Request model
class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "FinInsightBot is running!",
        "docs": "/docs"
    }

@app.post("/ask-session-stream")
async def ask_with_session_stream(request: SessionRequest):
    print(f"Incoming request: {request.session_id=}")

    session_id = request.session_id if request.session_id else str(uuid4())
    print(f"Using session_id: {session_id}")

    # Initialize session if needed
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "query": "",
            "ticker": [],
            "is_financial": False,
            "needs_fundamental": False,
            "needs_technical": False,
            "fundamental_data": {},
            "technical_data": {},
            "news_articles": [],
            "retrieved_docs": [],
            "answer": "",
        }

    # Append user message
    sessions[session_id]["messages"].append({"role": "user", "content": request.message})
    print(f"Session {session_id} Data:", sessions[session_id])

    inputs = sessions[session_id].copy()

    async def event_stream():
        try:
            current_state = sessions[session_id].copy()

            for step_output in financial_graph.stream(inputs):
                for node_name, updates in step_output.items():
                    current_state.update(updates)

                    if node_name == "analyze":
                        yield ' {"type": "status", "step": "analyzing_query"}\n\n'
                        ticker = updates.get("ticker")
                        if ticker:
                            # Send all tickers as a JSON array instead of just the first one
                            ticker_list_json = json.dumps(ticker if isinstance(ticker, list) else [ticker])
                            yield f' {{"type": "data", "ticker": {ticker_list_json}}}\n\n'
                    elif node_name == "retrieve_docs":
                        yield ' {"type": "status", "step": "retrieving_documents"}\n\n'
                    elif node_name == "fetch_fundamental":
                        print("Fundamental data fetched:", updates.get("fundamental_data"))
                        yield ' {"type": "status", "step": "fetching_fundamental_data"}\n\n'
                    elif node_name == "fetch_technical":
                        yield ' {"type": "status", "step": "fetching_technical_data"}\n\n'
                    elif node_name == "fetch_news":
                        yield ' {"type": "status", "step": "fetching_news"}\n\n'
                    elif node_name == "generate_answer":
                        yield ' {"type": "status", "step": "generating_answer"}\n\n'

            sessions[session_id].update(current_state)
            final_answer = current_state.get("answer", "").strip()
            ticker_list = current_state.get("ticker", [])
            # Send all tickers as an array instead of just the first one
            tickers = ticker_list if ticker_list else []

            fund_data = current_state.get("fundamental_data", {})
            tech_data = current_state.get("technical_data", {})
            market_data = {}
            if fund_data:
                market_data["fundamental"] = fund_data
            if tech_data:
                market_data["technical"] = tech_data

            yield f' {{"type": "done", "answer": {json.dumps(final_answer)}, "ticker": {json.dumps(tickers)}, "market_data": {json.dumps(market_data)}}}\n\n'

        except Exception as e:
            error_msg = str(e).replace('"', '\\"').replace('\n', ' ')
            yield f' {{"type": "error", "message": "{error_msg}"}}\n\n'

    headers = {
        "X-Session-ID": session_id,
        "Access-Control-Expose-Headers": "X-Session-ID"
    }

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
