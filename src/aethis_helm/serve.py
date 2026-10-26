"""HTTP API: POST /rank and GET /health. One process, one model, one lock."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from . import __version__
from .config import HelmConfig
from .model import PromptTooLongError
from .ranker import Ranker
from .types import FINISH, FINISH_DESCRIPTION, Step, ToolSpec

log = logging.getLogger(__name__)

_DEFAULTS = HelmConfig()
MAX_NAME_CHARS = 128
MAX_DESCRIPTION_CHARS = 10_000  # accepted on input; only the first 200 reach the prompt
MAX_TASK_CHARS = 20_000
MAX_HISTORY = 100
MAX_RESULT_CHARS = 100_000
FINISH_NOTE = "'Finish' was added to the tool list because the model was trained with it present."


def _clean_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("tool name must not be empty")
    if any(ch in name for ch in "\r\n\t") or not name.isprintable():
        raise ValueError("tool name must be a single printable line")
    return name


class ToolIn(BaseModel):
    name: str = Field(max_length=MAX_NAME_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)

    _name = field_validator("name")(_clean_name)


class HistoryItem(BaseModel):
    tool: str = Field(max_length=MAX_NAME_CHARS)
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = Field(default="", max_length=MAX_RESULT_CHARS)

    _tool = field_validator("tool")(_clean_name)

    def to_step(self) -> Step:
        # indent=2 matches the pretty-printed JSON arguments the model saw in training
        return Step(action=self.tool, arguments=json.dumps(self.args, indent=2, ensure_ascii=False),
                    observation=self.result)


class RankRequest(BaseModel):
    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY)
    tools: list[ToolIn] = Field(min_length=_DEFAULTS.min_tools, max_length=_DEFAULTS.max_tools)

    @model_validator(mode="after")
    def _unique_names(self) -> "RankRequest":
        names = [t.name for t in self.tools]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"tool names must be unique; duplicated: {dupes}")
        return self


class RankedOut(BaseModel):
    name: str
    score: float
    prob: float


class RankResponse(BaseModel):
    ranked: list[RankedOut]
    finish_added: bool
    notes: list[str]
    latency_ms: float


def with_finish(tools: list[ToolIn]) -> tuple[list[ToolSpec], bool]:
    """The request's tools in order, with Finish appended (as in training) when the caller left it out."""
    specs = [ToolSpec(t.name, t.description) for t in tools]
    if any(t.name == FINISH for t in tools):
        return specs, False
    return [*specs, ToolSpec(FINISH, FINISH_DESCRIPTION)], True


def create_app(ranker_factory: Callable[[], Ranker] | None = None) -> FastAPI:
    """`ranker_factory` builds the Ranker at startup; defaults to loading the real model from the environment."""
    factory = ranker_factory or (lambda: Ranker.load(HelmConfig.from_env()))
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranker = factory()
        yield
        app.state.ranker = None

    app = FastAPI(title="Aethis Helm", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        ranker = getattr(request.app.state, "ranker", None)
        return {"status": "ok" if ranker else "loading", "model_loaded": ranker is not None,
                "device": ranker.device if ranker else None, "version": __version__}

    @app.post("/rank", response_model=RankResponse)
    def rank(body: RankRequest, request: Request) -> RankResponse:
        ranker = getattr(request.app.state, "ranker", None)
        if ranker is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        tools, finish_added = with_finish(body.tools)
        history = [h.to_step() for h in body.history]
        started = time.perf_counter()
        try:
            with lock:  # one forward pass at a time: GPU/MPS state is not safe to share across threads
                ranked = ranker.rank(body.task, history, tools)
        except PromptTooLongError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        latency_ms = 1000.0 * (time.perf_counter() - started)
        return RankResponse(ranked=[RankedOut(name=r.name, score=r.score, prob=r.prob) for r in ranked],
                            finish_added=finish_added, notes=[FINISH_NOTE] if finish_added else [],
                            latency_ms=round(latency_ms, 1))

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("HELM_LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s %(message)s")
    host = os.environ.get("HELM_HOST", "127.0.0.1")
    port = int(os.environ.get("HELM_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
