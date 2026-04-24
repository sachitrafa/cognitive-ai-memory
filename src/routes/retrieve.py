from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.services.retrieve import retrieve

router = APIRouter()


class RetrieveRequest(BaseModel):
    userId: str
    query: str
    topK: int = Field(5, ge=1, le=500)


@router.post("/retrieve")
def retrieve_memories(req: RetrieveRequest):
    return retrieve(req.userId, req.query, req.topK)
