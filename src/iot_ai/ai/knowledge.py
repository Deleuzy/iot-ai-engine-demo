"""Product and support passages. Each Markdown section (##) is one passage with the id `file#section-slug`.

The documents in knowledge/ are written for this demo. They are not vendor or Aurocon documentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from iot_ai.ai.embed import Embedder
from iot_ai.ai.store import VectorStore
from iot_ai.config import KNOWLEDGE_DIR


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def load_passages(directory: Path = KNOWLEDGE_DIR) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        current_title = ""
        body: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if current_title:
                    sections.append((current_title, body))
                current_title, body = line[3:].strip(), []
            elif current_title:
                body.append(line)
        if current_title:
            sections.append((current_title, body))
        for title, lines in sections:
            text = "\n".join(lines).strip()
            passages.append(Passage(id=f"{path.stem}#{_slug(title)}", title=title, text=text))
    return passages


def build_index(passages: list[Passage], embedder: Embedder) -> VectorStore:
    store = VectorStore(dim=embedder.embed(["probe"]).shape[1])
    vectors = embedder.embed([f"{p.title}. {p.text}" for p in passages])
    store.add([p.id for p in passages], vectors)
    return store
