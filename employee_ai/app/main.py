import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.database import initialize_database
from app.graph import graph


app = FastAPI(
    title="Employee Intelligent QA",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=1000,
    )


class AskResponse(BaseModel):
    answer: str
    route: str
    reasoning: str
    sources: list[dict]
    processing_time_ms: float


@app.on_event("startup")
def startup():
    initialize_database()


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.perf_counter()

    try:
        result = graph.invoke(
            {
                "query": request.query
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return AskResponse(
        answer=result.get(
            "answer",
            "Unable to generate an answer.",
        ),
        route=result.get(
            "route",
            "unknown",
        ),
        reasoning=result.get(
            "reasoning",
            "",
        ),
        sources=result.get(
            "documents",
            [],
        ),
        processing_time_ms=round(elapsed, 2),
    )