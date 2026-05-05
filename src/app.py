import os
import json
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.routes import memories, retrieve, agents, ui, graph_viz
from src.jobs.decay_job import run as run_decay
from src.db.migrate import migrate


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate()
    scheduler.add_job(run_decay, "interval", hours=24, id="decay_job")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="YourMemory", version="0.1.0", lifespan=lifespan)

app.include_router(memories.router)
app.include_router(retrieve.router)
app.include_router(agents.router)
app.include_router(ui.router)
app.include_router(graph_viz.router)


@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int = 3


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    import getpass
    from src.services.retrieve import retrieve as _retrieve

    OLLAMA_URL      = os.getenv("YOURMEMORY_OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL    = os.getenv("YOURMEMORY_OLLAMA_MODEL", "llama3.2:3b")
    MIN_SCORE       = 0.52   # direct cosine+BM25 matches (raised to cut false positives)
    MIN_GRAPH_SCORE = 0.20   # graph-expanded nodes (capped at 0.6×0.74≈0.444)

    user_id = req.user_id or os.getenv("YOURMEMORY_USER", "") or getpass.getuser()

    results  = _retrieve(user_id, req.query, top_k=req.top_k)
    memories = results.get("memories", [])

    direct = [m for m in memories if not m.get("via_graph")]
    if not direct or direct[0].get("score", 0) < MIN_SCORE:
        return {"answer": "Not enough memory context to answer without Claude.", "grounded": False}

    memory_lines = "\n".join(
        f"{i+1}. {m['content']}"
        for i, m in enumerate(memories)
        if m.get("score", 0) >= (MIN_GRAPH_SCORE if m.get("via_graph") else MIN_SCORE)
    )

    prompt = f"""You are a memory assistant. Answer ONLY using the provided memories below.
Be concise and direct. If the answer is not clearly supported by the memories, say exactly: "I don't know — ask Claude."

Memories:
{memory_lines}

Question: {req.query}
Answer:"""

    def stream_ollama():
        payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}).encode()
        try:
            ollama_req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(ollama_req, timeout=30) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            yield f"Ollama error: {exc}"

    return StreamingResponse(stream_ollama(), media_type="text/plain")
