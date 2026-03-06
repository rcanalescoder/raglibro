from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskRequest, AskResponse, SourceChunk
from .rag import RagService

app = FastAPI(title="RAG Libro (Ollama + LangChain)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RagService()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer, sources = rag.answer(req.question)
    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
    )