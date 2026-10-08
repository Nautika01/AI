"""로컬 문서 지식 베이스 (하이브리드 검색: BM25 + 의미 벡터).

`docs_dir`의 Markdown/텍스트 파일을 읽어 섹션 단위로 청크를 만들고,
1) 어휘 기반 BM25 점수와 2) 임베딩 코사인 유사도를 정규화해 가중합으로 합쳐 검색한다.
동의어 사전으로 질문을 확장하고, 머리말(front matter) 메타데이터(분류·시행일 등)로 필터링할 수 있다.
망 분리 환경을 고려해 기본 설정은 외부 서비스·모델 다운로드 없이 동작한다.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embeddings import Embedder, EmbeddingCache, HashEmbedder, VectorIndex
from .metadata import DocMeta, parse_front_matter
from .synonyms import SynonymMap
from .tokenizer import tokenize

log = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    source: str
    meta: DocMeta = field(default_factory=DocMeta, compare=False)

    @property
    def ref(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n{self.section}\n{self.text}"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


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
    def __init__(self, k1: float = 1.5, b: float = 0.75, *, synonyms: SynonymMap | None = None, embedder: Embedder | None | bool = None, cache_dir: Path | str | None = None):
        """`embedder=None` 이면 HashEmbedder(모델 불필요), `embedder=False` 면 BM25 만 사용한다."""
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0
        self.synonyms = synonyms or SynonymMap()
        self.embedder: Embedder | None = None if embedder is False else (embedder or HashEmbedder())
        self._cache = EmbeddingCache(cache_dir, self.embedder.name) if self.embedder else None
        self._vectors = VectorIndex()
        self.embedding_error: str | None = None  # 임베딩 서버 장애 시 사유 (BM25 만으로 계속 동작)

    # ---- 적재 ---------------------------------------------------------
    @classmethod
    def from_directory(cls, docs_dir: Path | str, **kwargs) -> "DocumentStore":
        store = cls(**kwargs)
        docs_dir = Path(docs_dir)
        if docs_dir.is_dir():
            for path in sorted(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt"))):
                try:
                    store.add_document(path.read_text(encoding="utf-8"), source=str(path.relative_to(docs_dir)), doc_id=path.stem)
                except UnicodeDecodeError:
                    log.warning("UTF-8 이 아닌 파일을 건너뜁니다: %s", path)
        store.ensure_vectors()
        return store

    def add_document(self, content: str, *, source: str, doc_id: str | None = None, title: str | None = None, meta: DocMeta | None = None) -> int:
        doc_id = doc_id or source
        parsed_meta, body_text = parse_front_matter(content)
        meta = meta or parsed_meta
        lines = body_text.strip().splitlines()
        if title is None:
            title = meta.title or next((ln[2:].strip() for ln in lines if ln.startswith("# ")), doc_id)
        body = "\n".join(ln for ln in lines if not ln.startswith("# "))
        added = 0
        for section, text in _split_sections(body):
            chunk = Chunk(doc_id=doc_id, title=title, section=section, text=text, source=source, meta=meta)
            # 제목·섹션명·태그는 두 번 넣어 가중치를 높인다.
            tokens = tokenize(f"{title} {section} {' '.join(meta.tags)} {meta.category}") * 2 + tokenize(text)
            self.chunks.append(chunk)
            self._tf.append(Counter(tokens))
            self._doc_len.append(len(tokens))
            self._df.update(set(tokens))
            added += 1
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        return added

    def ensure_vectors(self) -> None:
        """아직 벡터가 없는 청크의 임베딩을 계산한다 (캐시 활용). 검색 전에 자동 호출된다."""
        if self.embedder is None or self._cache is None:
            return
        pending = self.chunks[len(self._vectors):]
        if not pending:
            return
        try:
            vectors = self._cache.get_many([c.embed_text for c in pending], self.embedder)
        except Exception as e:  # noqa: BLE001 - 임베딩 서버 장애로 비서 전체가 멈추면 안 된다
            self.embedding_error = f"임베딩 계산 실패({self.embedder.name}): {type(e).__name__}: {e}. 어휘 검색(BM25)만 사용합니다."
            log.error(self.embedding_error)
            self.embedder = None
            self._vectors = VectorIndex()
            return
        self._vectors.add(vectors)
        self._cache.save()

    # ---- 조회 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def titles(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.title, None)
        return list(seen)

    @property
    def categories(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            if c.meta.category:
                seen.setdefault(c.meta.category, None)
        return list(seen)

    def documents(self) -> list[dict[str, Any]]:
        """문서 단위 요약 (제목, 출처, 청크 수, 메타데이터)."""
        out: dict[str, dict[str, Any]] = {}
        for c in self.chunks:
            d = out.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "source": c.source, "chunks": 0, "meta": c.meta.to_dict()})
            d["chunks"] += 1
        return list(out.values())

    # ---- 검색 ---------------------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._doc_len[i]
            for term in q_tokens:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                score += self._idf(term) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _allowed(self, chunk: Chunk, category: str | None, doc_id: str | None, tag: str | None) -> bool:
        if category and chunk.meta.category.lower() != category.lower():
            return False
        if doc_id and chunk.doc_id != doc_id and chunk.title != doc_id:
            return False
        if tag and tag.lower() not in {t.lower() for t in chunk.meta.tags}:
            return False
        return True

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0, *, category: str | None = None, doc_id: str | None = None, tag: str | None = None) -> list[SearchHit]:
        """하이브리드 검색. 점수는 정규화 가중합(0~10+)으로, 절대값보다 순위가 의미 있다."""
        query = query.strip()
        if not query or not self.chunks:
            return []
        self.ensure_vectors()
        expanded = self.synonyms.expand(query)
        candidates = max(top_k * 5, 20)

        def allowed(i: int) -> bool:
            return self._allowed(self.chunks[i], category, doc_id, tag)

        bm25 = [(i, s) for i, s in self._bm25(expanded) if allowed(i)][:candidates]
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25, 1)}
        bm25_max = bm25[0][1] if bm25 else 1.0
        bm25_norm = {i: s / bm25_max for i, s in bm25}

        vec_rank: dict[int, int] = {}
        vec_norm: dict[int, float] = {}
        weight = 0.0
        if self.embedder is not None and len(self._vectors):
            weight = self.embedder.weight
            qv = self.embedder.embed([query])[0]
            ranked = [(i, s) for i, s in self._vectors.top(qv, candidates * 3) if allowed(i)][:candidates]
            if ranked:
                hi, lo = ranked[0][1], ranked[-1][1]
                span = (hi - lo) or 1.0
                vec_rank = {i: r for r, (i, _) in enumerate(ranked, 1)}
                vec_norm = {i: (s - lo) / span for i, s in ranked}

        # 정규화 점수의 가중합: 어휘 점수가 기준이고, 벡터 점수는 가까운 후보 간 순위를 바로잡거나
        # 어휘 검색이 놓친 문서를 끌어올리는 보조 역할을 한다.
        fused: dict[int, float] = {}
        for i in set(bm25_norm) | set(vec_norm):
            fused[i] = bm25_norm.get(i, 0.0) + weight * vec_norm.get(i, 0.0)
        if not fused:
            return []
        order = sorted(fused.items(), key=lambda x: (-x[1], bm25_rank.get(x[0], 10**6), vec_rank.get(x[0], 10**6)))
        hits = [SearchHit(chunk=self.chunks[i], score=s * 10, bm25_rank=bm25_rank.get(i), vector_rank=vec_rank.get(i)) for i, s in order]
        return [h for h in hits if h.score > min_score][:top_k]

    def format_hits(self, hits: list[SearchHit], max_chars: int = 900) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다."
        parts = []
        for n, h in enumerate(hits, 1):
            text = h.chunk.text
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + " …"
            meta_bits = [b for b in (h.chunk.meta.category, h.chunk.meta.effective_date and f"시행 {h.chunk.meta.effective_date}", h.chunk.meta.version and f"v{h.chunk.meta.version}") if b]
            meta_str = f", {' · '.join(meta_bits)}" if meta_bits else ""
            parts.append(f"[{n}] {h.chunk.ref} (출처: {h.chunk.source}{meta_str}, 점수 {h.score:.2f})\n{text}")
        return "\n\n".join(parts)
