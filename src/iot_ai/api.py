"""FastAPI service: list events, explain one event, read the saved evaluation results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from iot_ai.ai.backends import make_backend
from iot_ai.ai.embed import HashingEmbedder
from iot_ai.ai.engine import Engine
from iot_ai.ai.knowledge import build_index, load_passages
from iot_ai.ai.tools import DeviceStore
from iot_ai.config import PROCESSED_DIR, RESULTS_DIR


class ExplainRequest(BaseModel):
    event_id: str


@dataclass
class State:
    engine: Engine | None
    events: list[dict[str, Any]]
    backend_name: str


def load_state(processed_dir: Path = PROCESSED_DIR) -> State:
    events_path = processed_dir / "events.json"
    if not events_path.exists() or not (processed_dir / "grid.csv.gz").exists():
        return State(engine=None, events=[], backend_name="none")
    events: list[dict[str, Any]] = json.loads(events_path.read_text(encoding="utf-8"))
    backend = make_backend(dict(os.environ))
    embedder = HashingEmbedder()
    passages = load_passages()
    engine = Engine(backend, embedder, build_index(passages, embedder), passages, DeviceStore.load(processed_dir))
    return State(engine=engine, events=events, backend_name=backend.name)


def create_app(state: State | None = None) -> FastAPI:
    """Application factory. Pass a State in tests; the server loads the data from processed/ at start."""
    app = FastAPI(title="IoT AI engine demo", version="0.1.0")
    app.state.iot = state if state is not None else load_state()

    @app.get("/health")
    def health() -> dict[str, Any]:
        s: State = app.state.iot
        return {"status": "ok", "data_loaded": s.engine is not None, "backend": s.backend_name, "events": len(s.events)}

    @app.get("/events")
    def events(device: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        s: State = app.state.iot
        rows = [e for e in s.events if device is None or e["device"] == device]
        return [{"event_id": f"{e['device']}@{e['start']}", **e} for e in rows[: max(1, min(limit, 200))]]

    @app.post("/explain")
    def explain(body: ExplainRequest) -> dict[str, Any]:
        s: State = app.state.iot
        if s.engine is None:
            raise HTTPException(status_code=503, detail="data not built: run `python -m iot_ai run` first")
        for e in s.events:
            if f"{e['device']}@{e['start']}" == body.event_id:
                return s.engine.explain(e).as_dict()
        raise HTTPException(status_code=404, detail="unknown event_id")

    @app.get("/results/detection")
    def detection() -> dict[str, Any]:
        path = RESULTS_DIR / "detection_metrics.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="run the pipeline first")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    return app
