"""군사 용어·약어 사전 조회."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FIELDS = ("full", "ko", "desc")


def load_glossary(path: Path | str | None) -> dict[str, dict[str, str]]:
    """용어 사전 JSON 을 읽는다.

    운영자가 직접 편집하는 파일이므로 BOM·빈 파일을 허용하고, 깨진 파일이나 잘못된 항목은
    경고 로그를 남긴 뒤 건너뛰어 서버·CLI 기동이 중단되지 않게 한다.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        log.warning("용어 사전을 읽을 수 없어 빈 사전으로 시작합니다: %s (%s)", p, e)
        return {}
    if not text.strip():
        return {}
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("용어 사전 JSON 형식 오류로 빈 사전으로 시작합니다: %s %d행 %d열 (%s)", p, e.lineno, e.colno, e.msg)
        return {}
    if not isinstance(data, dict):
        log.warning("용어 사전 최상위는 객체({약어: {...}})여야 합니다: %s (받은 형식: %s)", p, type(data).__name__)
        return {}
    glossary: dict[str, dict[str, str]] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            log.warning("용어 사전 항목 %r 은(는) 객체가 아니어서 건너뜁니다: %s", key, p)
            continue
        missing = [f for f in _FIELDS if not isinstance(entry.get(f), str) or not entry.get(f)]
        if len(missing) == len(_FIELDS):
            log.warning("용어 사전 항목 %r 에 full/ko/desc 가 없어 건너뜁니다: %s", key, p)
            continue
        if missing:
            log.warning("용어 사전 항목 %r 에 %s 값이 없습니다: %s", key, ", ".join(missing), p)
        glossary[str(key)] = {f: entry[f] if isinstance(entry.get(f), str) else "" for f in _FIELDS}
    return glossary


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
        return f"{k} — {e.get('full', '')} ({e.get('ko', '')})\n{e.get('desc', '')}"
    ql = q.lower()
    candidates = [
        k for k, e in glossary.items()
        if isinstance(e, dict) and (ql in k.lower() or any(ql in str(e.get(f, "")).lower() for f in _FIELDS))
    ]
    if not candidates:
        return f"'{q}'에 해당하는 용어를 사전에서 찾지 못했습니다. 지식 베이스 검색(search_defense_docs)을 시도하거나 일반 지식으로 답하되 출처가 사전이 아님을 밝히십시오."
    lines = [f"'{q}' 관련 용어 {min(len(candidates), limit)}건:"]
    for k in candidates[:limit]:
        e = glossary[k]
        lines.append(f"- {k}: {e.get('full', '')} ({e.get('ko', '')}) — {e.get('desc', '')}")
    return "\n".join(lines)
