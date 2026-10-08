"""Markdown 문서 머리말(front matter) 파싱.

문서 맨 위에 다음 형식으로 메타데이터를 둘 수 있다. 모두 선택 사항이다.

    ---
    title: 경계근무 규정
    category: 규정
    effective_date: 2025-03-01
    version: 3.1
    source_file: 경계근무규정.hwp
    tags: 경계, 근무
    ---
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_FRONT_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)


@dataclass(frozen=True)
class DocMeta:
    title: str = ""
    category: str = ""
    effective_date: str = ""
    version: str = ""
    source_file: str = ""
    tags: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d = {"title": self.title, "category": self.category, "effective_date": self.effective_date, "version": self.version, "source_file": self.source_file, "tags": list(self.tags)}
        d.update(self.extra)
        return {k: v for k, v in d.items() if v}


def parse_front_matter(content: str) -> tuple[DocMeta, str]:
    """(메타데이터, 머리말을 뗀 본문) 을 돌려준다. 머리말이 없으면 빈 메타데이터."""
    content = content.lstrip("﻿")  # UTF-8 BOM 은 매칭과 본문 오프셋 모두에서 제거한다
    m = _FRONT_RE.match(content)
    if not m:
        return DocMeta(), content
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip().strip("'\"")
    known = {"title", "category", "effective_date", "version", "source_file", "tags"}
    tags = tuple(t.strip() for t in re.split(r"[,，]", fields.get("tags", "")) if t.strip())
    meta = DocMeta(
        title=fields.get("title", ""),
        category=fields.get("category", ""),
        effective_date=fields.get("effective_date", ""),
        version=fields.get("version", ""),
        source_file=fields.get("source_file", ""),
        tags=tags,
        extra={k: v for k, v in fields.items() if k not in known},
    )
    return meta, content[m.end():].lstrip("\n")


def render_front_matter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v in (None, "", [], ()):
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"
