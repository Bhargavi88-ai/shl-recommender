# SHL Assessment Recommender

A conversational FastAPI agent that helps hiring managers find the right SHL Individual Test Solution assessments through natural dialogue.

---

## Architecture Overview

```
User → POST /chat (full history) → FastAPI
           ↓
      Prompt injection guard
           ↓
      TF-IDF retrieval (catalog_store.py)
           ↓
      Claude API call (grounded system prompt)
           ↓
      JSON parse + hallucination filter
           ↓
      ChatResponse { reply, recommendations[], end_of_conversation }
```

**Key design choices:**
- **Stateless API**: full conversation history on every request; no server-side session state.
- **TF-IDF retrieval**: lightweight, no GPU needed, runs on free-tier hosting. Fast enough for the 8-turn cap.
- **Grounded prompting**: Claude receives the top-20 retrieved assessments + the full catalog index in every call, preventing hallucination.
- **Hallucination filter**: every recommended assessment is looked up in the catalog after Claude responds; any invented names are silently stripped.
- **Bundled fallback catalog**: `data/catalog.json` ships with ~50 hand-verified SHL assessments so the service works even if scraping fails.

---

## Prerequisites

- **Python 3.10+**
- **Anthropic API key** (free tier works; get one at https://console.anthropic.com)
- Internet access (to call the Anthropic API)

---

## Local Setup — Step by Step

### Step 1: Clone / download the project

```bash
# If you have git:
git clone <your-repo-url>
cd shl-recommender

# Or just unzip the downloaded file:
unzip shl-recommender.zip
cd shl-recommender
```

### Step 2: Create a virtual environment

```bash
python -m venv venv

# Activate it:
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set your API key

```bash
# Copy the template:
cp .env.example .env

# Edit .env and paste your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

On Linux/macOS you can also export it directly:
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 5: (Optional) Refresh the SHL catalog

The repo ships with a pre-built `data/catalog.json`. To scrape the live SHL catalog:

```bash
python refresh_catalog.py
```

This takes ~1–2 minutes. Add `--descriptions` to also fetch product detail pages (slower).

### Step 6: Start the server

```bash
python main.py
```

The server starts on **http://localhost:8000**

You should see:
```
INFO — Catalog ready — 55 assessments indexed.
INFO — Application startup complete.
INFO — Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Verify it's running

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

### Step 8: Run the test suite

In a second terminal (with venv active):

```bash
python test_api.py
```

Expected output: `7/7 passed`.

---

## API Reference

### `GET /health`

```
200 OK
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "I am hiring a mid-level Java developer"},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "Around 4 years experience, works with stakeholders"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, here are 4 assessments that fit...",
  "recommendations": [
    {"name": "Java 8 (New)",  "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",        "url": "https://www.shl.com/...", "test_type": "P"},
    {"name": "Verify Verbal Reasoning", "url": "...", "test_type": "A"},
    {"name": "Verify Numerical Reasoning", "url": "...", "test_type": "A"}
  ],
  "end_of_conversation": false
}
```

**Rules:**
- `recommendations` is `[]` while the agent is gathering context.
- `recommendations` has 1–10 items once the agent commits to a shortlist.
- `end_of_conversation` is `true` only when the task is complete.
- Max 8 turns per conversation (user + assistant combined).
- Each call must time out within 30 seconds.

---

## Docker Deployment

```bash
# Build
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

---

## Free Cloud Deployment (Render)

1. Push code to a GitHub repo (ensure `.env` is in `.gitignore`).
2. Go to https://render.com → New Web Service.
3. Connect your repo.
4. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python main.py`
   - **Environment variable**: `ANTHROPIC_API_KEY=sk-ant-...`
5. Deploy. Your URL will be `https://your-service.onrender.com`.

**Important for Render free tier**: The first `/health` call after a cold start may take up to 2 minutes. This matches the evaluator's allowed warm-up time.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Your Anthropic API key |
| `FORCE_SCRAPE` | ❌ | `false` | Set `true` to re-scrape SHL catalog on startup |
| `PORT` | ❌ | `8000` | Port to listen on |

---

## Project Structure

```
shl-recommender/
├── main.py              # FastAPI app (endpoints, lifespan)
├── agent.py             # Conversational agent (retrieval + Claude call + validation)
├── catalog_store.py     # TF-IDF index over the catalog
├── catalog_scraper.py   # SHL website scraper
├── models.py            # Pydantic request/response models
├── refresh_catalog.py   # CLI to manually refresh catalog
├── test_api.py          # End-to-end API test suite
├── data/
│   └── catalog.json     # Pre-built SHL assessment catalog (fallback)
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Extending / Improving

- **Better retrieval**: swap TF-IDF for `sentence-transformers` + FAISS by replacing `catalog_store.py`. The rest of the code is unchanged.
- **Richer catalog**: run `python refresh_catalog.py --descriptions` to add product descriptions, which improves retrieval quality significantly.
- **Streaming**: FastAPI supports `StreamingResponse`; swap the `/chat` handler to stream Claude's reply token by token.
- **Caching**: add `functools.lru_cache` or Redis to avoid re-indexing on every worker restart.
