# main.py
"""
FastAPI application.
  GET  /health  → readiness probe
  POST /chat    → stateless conversational agent
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Literal
from dotenv import load_dotenv

load_dotenv()   # reads .env file → sets GROQ_API_KEY in environment

from agent import run_agent
from catalog import get_catalog


# ── Startup: pre-load catalog so first request is fast ────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading catalog and building search index…")
    get_catalog()   # builds FAISS index if not cached, loads if cached
    print("Service ready.")
    yield           # app runs here
    # (cleanup code would go here if needed)


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {"status": "working"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class Message(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

    @validator("content")
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many messages")
        if v[-1].role != "user":
            raise ValueError("Last message must be from the user")
        return v


class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str


class ChatResponse(BaseModel):
    reply:               str
    recommendations:     list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness probe — returns 200 OK when service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversational agent.
    Client sends FULL conversation history on every call.
    Server stores NO state — any server instance can handle any request.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Enforce 8-turn cap (assignment requirement)
    if len(messages) > 8:
        return ChatResponse(
            reply="This conversation has reached its maximum length. Please start a new session.",
            recommendations=[],
            end_of_conversation=True,
        )
    
    try:
        result = run_agent(messages)
    
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("CHAT ERROR")
        print("=" * 60)
     
        traceback.print_exc()
        
        print("=" * 60)
        return ChatResponse(
        reply=f"ERROR: {str(e)}",
        recommendations=[],
        end_of_conversation=False
        )

    return ChatResponse(
    reply=result["reply"],
    recommendations=[Recommendation(**r) for r in result["recommendations"]],
    end_of_conversation=result["end_of_conversation"],
)


# ── Dev entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)