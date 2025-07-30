from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.rag_service import RAGService
from datetime import date
router = APIRouter()
rag_svc = RAGService()

class RAGRequest(BaseModel):
    question: str
    user_id: int
    start_date: date | None = None
    end_date:   date | None = None

class RAGResponse(BaseModel):
    question: str
    answer: str
    source_chunks: list[str]

@router.post("/Question", response_model=RAGResponse)
async def raq_qa(request: RAGRequest):
    question = request.question
    user_id  = request.user_id
    start_date  = request.start_date
    end_date    = request.end_date
    try:
        answer, chunks = rag_svc.answer_question(user_id, question, start_date, end_date)
        #answer, chunks = rag_svc.answer_question(user_id, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found.")
    return RAGResponse(
        question=question,
        answer=answer,
        source_chunks=chunks
    )
