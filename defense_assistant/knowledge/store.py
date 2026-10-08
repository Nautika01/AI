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
import time
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
_EMBED_RETRY_SECONDS = 30.0  # 질의 임베딩 실패 후 다시 시도하기까지 BM25 만 쓰는 시간


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
    query: str = field(default="", compare=False)  # 결과 표시 때 질의어 주변을 보여 주기 위한 원 질의


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
        for para in _paragraphs(text):
            if size + len(para) > _MAX_CHUNK_CHARS and piece:
                out.append((f"{name} ({part})", "\n\n".join(piece)))
                part += 1
                piece, size = [], 0
            piece.append(para)
            size += len(para)
        if piece:
            out.append((f"{name} ({part})" if part > 1 else name, "\n\n".join(piece)))
    return out


def _paragraphs(text: str):
    """빈 줄 기준 문단을 돌려주되, 한도를 넘는 문단은 줄 단위로, 그래도 긴 줄은 고정 길이로 나눈다.

    빈 줄 없는 글머리표 목록처럼 긴 문단이 청크 하나로 남아 1,200자 한도가 무력화되는 것을 막는다.
    """
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= _MAX_CHUNK_CHARS:
            yield para
            continue
        buf: list[str] = []
        size = 0
        for line in para.splitlines():
            while len(line) > _MAX_CHUNK_CHARS:
                if buf:
                    yield "\n".join(buf)
                    buf, size = [], 0
                yield line[:_MAX_CHUNK_CHARS]
                line = line[_MAX_CHUNK_CHARS:]
            if not line:
                continue
            if buf and size + 1 + len(line) > _MAX_CHUNK_CHARS:
                yield "\n".join(buf)
                buf, size = [], 0
            size += len(line) + (1 if buf else 0)
            buf.append(line)
        if buf:
            yield "\n".join(buf)


def _best_window(text: str, query: str, size: int) -> int:
    """긴 청크를 잘라 보여 줄 때, 질의어가 가장 많이 들어가는 구간의 시작 위치(줄 시작 기준)를 고른다.

    질의어가 앞부분에 있거나 질의가 없으면 0 (기존처럼 앞에서부터).
    """
    terms = {t for t in tokenize(query) if len(t) >= 2}
    if not terms:
        return 0
    low = text.lower()
    terms = {t for t in terms if t in low}
    if not terms:
        return 0
    starts = [0] + [m.end() for m in re.finditer(r"\n", text) if m.end() < len(text)]
    best, best_count = 0, -1
    for st in starts:
        window = low[st : st + size]
        count = sum(1 for t in terms if t in window)
        if count > best_count:
            best, best_count = st, count
    return best


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
        self._query_embed_failed = False  # embedding_error 가 질의 임베딩 실패(일시 장애)에서 온 것인지
        self._embed_retry_at = 0.0  # 질의 임베딩 실패 후 이 시각(monotonic)까지는 벡터 단계를 건너뛴다

    # ---- 적재 ---------------------------------------------------------
    @classmethod
    def from_directory(cls, docs_dir: Path | str, **kwargs) -> "DocumentStore":
        store = cls(**kwargs)
        docs_dir = Path(docs_dir)
        if docs_dir.is_dir():
            for path in sorted(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt"))):
                try:
                    store.add_document(path.read_text(encoding="utf-8-sig"), source=str(path.relative_to(docs_dir)), doc_id=path.stem)
                except UnicodeDecodeError:
                    log.warning("UTF-8 이 아닌 파일을 건너뜁니다: %s", path)
        store.ensure_vectors()
        return store

    def add_document(self, content: str, *, source: str, doc_id: str | None = None, title: str | None = None, meta: DocMeta | None = None) -> int:
        doc_id = doc_id or source
        parsed_meta, body_text = parse_front_matter(content)
        meta = meta or parsed_meta
        lines = body_text.strip().splitlines()
        # 첫 '# ' 줄만 문서 제목으로 쓰고 본문에서 뺀다. 이후 H1(장 제목 등)은 '## ' 로 강등해 섹션으로 남기고,
        # 코드 블록 안의 '# 주석' 줄은 그대로 둔다.
        h1_idx: list[int] = []
        in_fence = False
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
            elif not in_fence and ln.startswith("# "):
                h1_idx.append(i)
        if title is None:
            title = meta.title or (lines[h1_idx[0]][2:].strip() if h1_idx else doc_id)
        demote = set(h1_idx[1:])
        body = "\n".join("#" + ln if i in demote else ln for i, ln in enumerate(lines) if not (h1_idx and i == h1_idx[0]))
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
            self._query_embed_failed = False
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
        qv = self._embed_query(query) if self.embedder is not None and len(self._vectors) else None
        if qv is not None and self.embedder is not None:
            weight = self.embedder.weight
            ranked = [(i, s) for i, s in self._vectors.top(qv, candidates * 3) if allowed(i)][:candidates]
            if ranked:
                hi, lo = ranked[0][1], ranked[-1][1]
                vec_rank = {i: r for r, (i, _) in enumerate(ranked, 1)}
                if hi > lo:
                    vec_norm = {i: (s - lo) / (hi - lo) for i, s in ranked}
                else:
                    # 후보가 1개이거나 모두 동점이면 min-max 가 0 이 되어 의미 검색 결과가 버려지므로
                    # 코사인 값(0~1)을 그대로 쓴다. 질의 벡터가 0 이면(신호 없음) 0 이 되어 끌어올리지 않는다.
                    vec_norm = {i: min(max(s, 0.0), 1.0) for i, s in ranked}

        # 정규화 점수의 가중합: 어휘 점수가 기준이고, 벡터 점수는 가까운 후보 간 순위를 바로잡거나
        # 어휘 검색이 놓친 문서를 끌어올리는 보조 역할을 한다.
        fused: dict[int, float] = {}
        for i in set(bm25_norm) | set(vec_norm):
            fused[i] = bm25_norm.get(i, 0.0) + weight * vec_norm.get(i, 0.0)
        if not fused:
            return []
        order = sorted(fused.items(), key=lambda x: (-x[1], bm25_rank.get(x[0], 10**6), vec_rank.get(x[0], 10**6)))
        hits = [SearchHit(chunk=self.chunks[i], score=s * 10, bm25_rank=bm25_rank.get(i), vector_rank=vec_rank.get(i), query=query) for i, s in order]
        return [h for h in hits if h.score > min_score][:top_k]

    def _embed_query(self, query: str) -> list[float] | None:
        """질의 임베딩. 서버 장애 시 None 을 돌려 BM25 만으로 검색하게 하고, 잠시 뒤 다시 시도한다."""
        if self.embedder is None or time.monotonic() < self._embed_retry_at:
            return None
        try:
            qv = self.embedder.embed([query])[0]
        except Exception as e:  # noqa: BLE001 - 임베딩 서버 장애로 비서 전체가 멈추면 안 된다
            self.embedding_error = f"질의 임베딩 실패({self.embedder.name}): {type(e).__name__}: {e}. 어휘 검색(BM25)만 사용합니다."
            self._query_embed_failed = True
            self._embed_retry_at = time.monotonic() + _EMBED_RETRY_SECONDS
            log.warning(self.embedding_error)
            return None
        if self._query_embed_failed:
            self.embedding_error = None
            self._query_embed_failed = False
        return qv

    def format_hits(self, hits: list[SearchHit], max_chars: int = 900) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다."
        parts = []
        for n, h in enumerate(hits, 1):
            text = h.chunk.text
            if len(text) > max_chars:
                start = _best_window(text, h.query, max_chars)
                text = ("… " if start else "") + text[start : start + max_chars].rstrip() + (" …" if start + max_chars < len(text) else "")
            meta_bits = [b for b in (h.chunk.meta.category, h.chunk.meta.effective_date and f"시행 {h.chunk.meta.effective_date}", h.chunk.meta.version and f"v{h.chunk.meta.version}") if b]
            meta_str = f", {' · '.join(meta_bits)}" if meta_bits else ""
            parts.append(f"[{n}] {h.chunk.ref} (출처: {h.chunk.source}{meta_str}, 점수 {h.score:.2f})\n{text}")
        return "\n\n".join(parts)
