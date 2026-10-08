"""임베딩(의미 벡터) 제공자.

- OpenAIEmbedder: Ollama(nomic-embed-text, bge-m3 등)·vLLM·LM Studio 의 `/v1/embeddings` 규격. 망 분리 환경에서
  의미 기반 검색을 쓰려면 이것을 설정한다 (DAI_EMBEDDINGS=openai).
- HashEmbedder: 모델 없이 동작하는 문자 n-gram 해싱 벡터. 동의어는 못 잡지만 오타·부분 일치에 강하고
  설치·다운로드가 전혀 필요 없다 (기본값).
- 디스크 캐시: 청크 본문의 해시를 키로 벡터를 저장해 재시작 때 다시 계산하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

try:  # numpy 가 있으면 내적을 빠르게 계산한다 (필수 의존성은 아님)
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None


class Embedder(Protocol):
    name: str
    weight: float  # 하이브리드 융합 시 벡터 순위의 가중치 (BM25 = 1.0 기준)

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """문자 2·3-gram 해싱 (512차원, L2 정규화). 외부 의존성 없음."""

    name = "hash-ngram-512"
    weight = 0.15  # 의미 벡터가 아니므로 근소한 차이의 순위만 보정 (BM25 = 1.0 기준)

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        s = re.sub(r"\s+", " ", text.lower()).strip()
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                g = s[i : i + n]
                if g.strip() == "":
                    continue
                h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=4).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if n == 2 else 1.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder:
    """OpenAI 호환 `/v1/embeddings` (Ollama, vLLM, LM Studio 등)."""

    weight = 0.8

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0, batch_size: int = 32):
        import httpx2 as httpx

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model = model
        self.name = f"openai:{model}"
        self.batch_size = batch_size
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(timeout, connect=10.0))

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = self.client.post("/embeddings", json={"model": self.model, "input": batch})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in data]
            if len(vecs) != len(batch):
                raise RuntimeError(f"임베딩 서버가 {len(batch)}개 요청에 {len(vecs)}개를 돌려주었습니다.")
            out.extend(_normalize(v) for v in vecs)
        return out


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class EmbeddingCache:
    """{텍스트 해시: 벡터} 를 JSON 파일에 보관한다. 임베더 이름별로 파일이 나뉜다."""

    def __init__(self, cache_dir: Path | str | None, embedder_name: str):
        self.path = Path(cache_dir) / f"embeddings-{re.sub(r'[^A-Za-z0-9._-]+', '_', embedder_name)}.json" if cache_dir else None
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("임베딩 캐시를 읽지 못해 새로 만듭니다: %s", self.path)
                self._data = {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str], embedder: Embedder) -> list[list[float]]:
        keys = [self.key(t) for t in texts]
        missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in self._data]
        if missing:
            vecs = embedder.embed([t for _, t in missing])
            for (i, _), v in zip(missing, vecs):
                self._data[keys[i]] = v
            self._dirty = True
        return [self._data[k] for k in keys]

    def save(self) -> None:
        if not self.path or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False


class VectorIndex:
    """정규화된 벡터 목록에 대한 코사인 유사도 검색."""

    def __init__(self) -> None:
        self._rows: list[list[float]] = []
        self._matrix = None

    def add(self, vectors: list[list[float]]) -> None:
        self._rows.extend(vectors)
        self._matrix = None

    def __len__(self) -> int:
        return len(self._rows)

    def top(self, query: list[float], k: int) -> list[tuple[int, float]]:
        if not self._rows:
            return []
        if _np is not None:
            if self._matrix is None:
                self._matrix = _np.asarray(self._rows, dtype=_np.float32)
            scores = self._matrix @ _np.asarray(query, dtype=_np.float32)
            idx = _np.argsort(-scores)[:k]
            return [(int(i), float(scores[i])) for i in idx]
        scored = [(i, sum(a * b for a, b in zip(row, query))) for i, row in enumerate(self._rows)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
