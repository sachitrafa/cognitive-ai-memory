from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from src.routes import memories, retrieve, agents, ui
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


@app.get("/health")
def health():
    return {"status": "ok"}
