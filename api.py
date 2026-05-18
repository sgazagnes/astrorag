"""
AstroRAG FastAPI endpoint
Run with: uvicorn api:app --reload
Query with: curl -X POST http://localhost:8000/ask \
            -H "Content-Type: application/json" \
            -d '{"question": "What are Lyman-alpha escape mechanisms?"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retriever import load_retriever, build_qa_chain, query

# Load once at startup
retriever = load_retriever()
chain = build_qa_chain(retriever)

app = FastAPI(
    title="AstroRAG API",
    description="Query CLASSY survey research papers using RAG",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"message": "AstroRAG API is running. POST to /ask to query the papers."}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = query(chain, retriever, request.question)
    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"]
    )