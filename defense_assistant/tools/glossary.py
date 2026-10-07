"""군사 용어·약어 사전 조회."""

from __future__ import annotations

import json
from pathlib import Path


def load_glossary(path: Path | str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def lookup_term(glossary: dict[str, dict[str, str]], term: str, limit: int = 5) -> str:
    """약어·용어를 조회한다. 정확히 일치하면 그 항목을, 아니면 부분 일치 후보를 돌려준다."""
    q = term.strip()
    if not q:
        raise ValueError("조회할 용어가 비어 있습니다.")
    def _norm(s: str) -> str:
        return s.upper().replace(" ", "").replace("-", "")

    key = _norm(q)
    exact = {_norm(k): k for k in glossary}
    if key in exact:
        k = exact[key]
        e = glossary[k]
        return f"{k} — {e['full']} ({e['ko']})\n{e['desc']}"
    ql = q.lower()
    candidates = [
        k for k, e in glossary.items()
        if ql in k.lower() or ql in e["full"].lower() or ql in e["ko"].lower() or ql in e["desc"].lower()
    ]
    if not candidates:
        return f"'{q}'에 해당하는 용어를 사전에서 찾지 못했습니다. 지식 베이스 검색(search_defense_docs)을 시도하거나 일반 지식으로 답하되 출처가 사전이 아님을 밝히십시오."
    lines = [f"'{q}' 관련 용어 {min(len(candidates), limit)}건:"]
    for k in candidates[:limit]:
        e = glossary[k]
        lines.append(f"- {k}: {e['full']} ({e['ko']}) — {e['desc']}")
    return "\n".join(lines)
