"""REST API so the assistant can be integrated into existing applications.

Run with ``vrag serve`` and open http://localhost:8000 for the web UI or
http://localhost:8000/docs for the OpenAPI documentation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from versicherungs_rag import __version__
from versicherungs_rag.components import Components
from versicherungs_rag.config import RetrievalMode, get_settings
from versicherungs_rag.llm import LLMError

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Wie hoch ist der Selbstbehalt in der Vollkasko?"])
    model: str | None = Field(default=None, description="Name from config/models.yaml; default model if empty")
    retrieval_mode: RetrievalMode | None = None


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    models: list[str] = Field(min_length=1, max_length=4)
    retrieval_mode: RetrievalMode | None = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    description: str
    available: bool
    status: str


def create_app(components: Components | None = None) -> FastAPI:
    state: dict[str, Components] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        comps = components or Components(get_settings())
        comps.index  # fail fast at startup if the index is missing  # noqa: B018
        state["c"] = comps
        yield

    app = FastAPI(title="Versicherungs-RAG", version=__version__, lifespan=lifespan)

    def _answer(question: str, model: str, mode: RetrievalMode | None) -> dict:
        comps = state["c"]
        try:
            llm = comps.models.client(model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        available, status = llm.is_available()
        if not available:
            raise HTTPException(status_code=503, detail=f"{model}: {status}")
        try:
            return comps.pipeline(mode).answer(question, llm).to_dict()
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        index = state["c"].index
        return {"status": "ok", "chunks": len(index.chunks), "embedding_model": index.embedder_name}

    @app.get("/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        registry = state["c"].models
        infos = []
        for name in registry.names:
            spec = registry.spec(name)
            available, status = registry.client(name).is_available()
            infos.append(
                ModelInfo(
                    name=name, provider=spec.provider, description=spec.description, available=available, status=status
                )
            )
        return infos

    @app.post("/ask")
    async def ask(request: AskRequest) -> dict:
        model = request.model or state["c"].settings.default_model
        return await run_in_threadpool(_answer, request.question, model, request.retrieval_mode)

    @app.post("/compare")
    async def compare(request: CompareRequest) -> list[dict]:
        """Ask several models the same question in parallel, with identical retrieved context."""

        def run(model: str) -> dict:
            try:
                return _answer(request.question, model, request.retrieval_mode)
            except HTTPException as exc:
                return {"model": model, "error": exc.detail}

        def run_all() -> list[dict]:
            with ThreadPoolExecutor(max_workers=len(request.models)) as pool:
                return list(pool.map(run, request.models))

        return await run_in_threadpool(run_all)

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
