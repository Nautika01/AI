"""로컬 문서 지식 베이스 (BM25 기반 검색).

`docs_dir`의 Markdown/텍스트 파일을 읽어 섹션 단위로 청크를 만들고 BM25로 검색한다.
망 분리 환경을 고려해 외부 임베딩 서비스나 벡터 DB에 의존하지 않는다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import tokenize

_MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    source: str

    @property
    def ref(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def _split_sections(body: str) -> list[tuple[str, str]]:
    """`## 제목` 기준으로 (섹션명, 본문) 목록을 만든다. 너무 긴 섹션은 문단 단위로 나눈다."""
    sections: list[tuple[str, str]] = []
    current_name = ""
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((current_name, "\n".join(buf).strip()))
            current_name = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current_name, "\n".join(buf).strip()))

    out: list[tuple[str, str]] = []
    for name, text in sections:
        if len(text) <= _MAX_CHUNK_CHARS:
            out.append((name, text))
            continue
        piece: list[str] = []
        size = 0
        part = 1
        for para in re.split(r"\n\s*\n", text):
            if size + len(para) > _MAX_CHUNK_CHARS and piece:
                out.append((f"{name} ({part})", "\n\n".join(piece)))
                part += 1
                piece, size = [], 0
            piece.append(para)
            size += len(para)
        if piece:
            out.append((f"{name} ({part})" if part > 1 else name, "\n\n".join(piece)))
    return out


class DocumentStore:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    # ---- 적재 ---------------------------------------------------------
    @classmethod
    def from_directory(cls, docs_dir: Path | str, **kwargs) -> "DocumentStore":
        store = cls(**kwargs)
        docs_dir = Path(docs_dir)
        if docs_dir.is_dir():
            for path in sorted(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt"))):
                store.add_document(path.read_text(encoding="utf-8"), source=str(path.relative_to(docs_dir)), doc_id=path.stem)
        return store

    def add_document(self, content: str, *, source: str, doc_id: str | None = None, title: str | None = None) -> int:
        doc_id = doc_id or source
        lines = content.strip().splitlines()
        if title is None:
            title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), doc_id)
        body = "\n".join(ln for ln in lines if not ln.startswith("# "))
        added = 0
        for section, text in _split_sections(body):
            chunk = Chunk(doc_id=doc_id, title=title, section=section, text=text, source=source)
            # 제목·섹션명은 두 번 넣어 가중치를 높인다.
            tokens = tokenize(f"{title} {section}") * 2 + tokenize(text)
            self.chunks.append(chunk)
            self._tf.append(Counter(tokens))
            self._doc_len.append(len(tokens))
            self._df.update(set(tokens))
            added += 1
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        return added

    # ---- 검색 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def titles(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.title, None)
        return list(seen)

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[SearchHit]:
        q_tokens = set(tokenize(query))
        if not q_tokens or not self.chunks:
            return []
        hits: list[SearchHit] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._doc_len[i]
            for term in q_tokens:
                f = tf.get(term)
                if not f:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                score += idf * f * (self.k1 + 1) / denom
            if score > min_score:
                hits.append(SearchHit(chunk=self.chunks[i], score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def format_hits(self, hits: list[SearchHit], max_chars: int = 900) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다."
        parts = []
        for n, h in enumerate(hits, 1):
            text = h.chunk.text
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + " …"
            parts.append(f"[{n}] {h.chunk.ref} (출처: {h.chunk.source}, 점수 {h.score:.2f})\n{text}")
        return "\n\n".join(parts)
