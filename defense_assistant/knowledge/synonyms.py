"""동의어·유의어 사전.

`data/synonyms.json` 형식:
    {
      "경계근무": ["보초", "경계 근무", "불침번"],
      "MEDEVAC": ["의무후송", "환자후송", "메데박"]
    }
키와 값은 서로 동의어 그룹으로 묶이며, 질문에 그룹의 어느 단어가 나와도 나머지 단어가 검색어에 함께 보태진다.
모델 없이도 "보초"로 "경계근무" 문서를 찾게 해 주는, 운영자가 직접 손볼 수 있는 가장 단순한 장치다.
"""

from __future__ import annotations

import json
from pathlib import Path


class SynonymMap:
    def __init__(self, groups: dict[str, list[str]] | None = None):
        self._groups: list[tuple[str, ...]] = []
        self._index: dict[str, int] = {}
        for head, words in (groups or {}).items():
            self.add_group([head, *words])

    @classmethod
    def load(cls, path: Path | str | None) -> "SynonymMap":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("synonyms.json 은 {단어: [동의어, ...]} 형식이어야 합니다.")
        return cls({k: list(v) if isinstance(v, list) else [str(v)] for k, v in data.items()})

    def add_group(self, words: list[str]) -> None:
        norm = tuple(dict.fromkeys(w.strip() for w in words if w and w.strip()))
        if len(norm) < 2:
            return
        # 기존 그룹과 겹치면 합친다
        existing = {self._index[w.lower()] for w in norm if w.lower() in self._index}
        merged = list(norm)
        for gi in sorted(existing, reverse=True):
            merged = list(self._groups[gi]) + [w for w in merged if w not in self._groups[gi]]
            self._groups[gi] = ()
        gi = len(self._groups)
        self._groups.append(tuple(merged))
        for w in merged:
            self._index[w.lower()] = gi

    def __len__(self) -> int:
        return sum(1 for g in self._groups if g)

    def expand(self, query: str) -> str:
        """질문에 등장한 단어의 동의어를 질문 끝에 덧붙여 돌려준다 (검색용)."""
        if not self._groups:
            return query
        q = query.lower()
        extra: list[str] = []
        seen_groups: set[int] = set()
        for word, gi in self._index.items():
            if gi in seen_groups or word not in q:
                continue
            seen_groups.add(gi)
            extra.extend(w for w in self._groups[gi] if w.lower() not in q)
        return f"{query} {' '.join(extra)}" if extra else query
