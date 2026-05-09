"""
SHL Assessment Recommender — FastAPI Service

Endpoints
---------
GET  /health   → {"status": "ok"}
POST /chat     → ChatResponse (reply + recommendations + end_of_conversation)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import run_agent
from catalog_scraper import scrape_full_catalog, save_catalog, DATA_PATH
from catalog_store import get_store
from models import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog (scrape if missing or forced)."""
    force_scrape = os.getenv("FORCE_SCRAPE", "false").lower() == "true"

    if force_scrape or not DATA_PATH.exists():
        logger.info("Scraping SHL catalog on startup...")
        products = scrape_full_catalog(enrich_descriptions=False)
        if products:
            save_catalog(products)
            logger.info(f"Catalog updated: {len(products)} assessments saved.")
        else:
            logger.warning("Scraping returned 0 products; will use bundled fallback catalog.")

    store = get_store()
    logger.info(f"Catalog ready — {len(store.get_all())} assessments indexed.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "based on role requirements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Readiness check — always returns 200 once the service is up."""
    return HealthResponse(status="ok")


# ── Chat endpoint ───────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.

    The caller must pass the FULL conversation history on every request.
    The service holds no per-session state.

    Returns:
    - reply          : agent's next message
    - recommendations: 0–10 SHL assessments (empty while gathering context)
    - end_of_conversation: true when the agent considers the task done
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty.")

    # Validate roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid message role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Ensure conversation starts with a user turn
    if request.messages[0].role != "user":
        raise HTTPException(
            status_code=422, detail="First message must have role 'user'."
        )

    # Turn cap: evaluator enforces max 8 turns; we enforce it here too
    MAX_TURNS = 8
    if len(request.messages) > MAX_TURNS:
        raise HTTPException(
            status_code=422,
            detail=f"Conversation exceeds maximum of {MAX_TURNS} turns.",
        )

    try:
        response = await run_agent(request.messages)
    except Exception as e:
        logger.exception(f"Unhandled agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal agent error.")

    return response


# ── Global exception handler ────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,
    )
