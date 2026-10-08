"""추출한 문단을 지식 베이스용 Markdown 으로 구조화한다.

규정류 문서(제N조가 있는 문서)는 조문 단위로, 그 밖의 문서는 번호 매긴 제목(1. / 1.1) 단위로
`## 섹션` 을 만든다. 장·절 제목은 조문 섹션 이름 앞에 붙여 출처 표기가 "제1장 총칙 › 제3조(정의)" 처럼 나오게 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..knowledge.metadata import render_front_matter
from .extract import ExtractError, extract_paragraphs

_ARTICLE = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*(?:\(([^)]*)\)|\[([^\]]*)\])?\s*(.*)$")
_CHAPTER = re.compile(r"^제\s*\d+\s*[장절관편]\s*\S.*$")
_APPENDIX = re.compile(r"^(부칙|별표\s*\d*|별지\s*(?:제\s*\d+\s*호)?\s*(?:서식)?)\b.*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+\.?|\d+[.)])\s+(\S.*)$")  # "1. 개요", "2.1 절차", "3) 항목"
_MAX_HEADING_LEN = 60


@dataclass
class ConvertedDoc:
    title: str
    source_file: str
    format: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)  # (섹션명, 문단들)
    paragraphs: int = 0
    mode: str = "plain"  # article | numbered | plain

    @property
    def section_count(self) -> int:
        return sum(1 for name, _ in self.sections if name)


@dataclass
class IngestResult:
    source: Path
    output: Path | None
    doc: ConvertedDoc | None
    error: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def convert_file(path: Path | str, *, title: str | None = None) -> ConvertedDoc:
    path = Path(path)
    paragraphs = extract_paragraphs(path)
    if not paragraphs:
        raise ExtractError("문서에서 글자를 찾지 못했습니다.")
    doc_title = title or _guess_title(paragraphs, path)
    body = paragraphs[1:] if (not title and paragraphs and paragraphs[0] == doc_title) else paragraphs
    mode = _detect_mode(body)
    sections = _structure(body, mode)
    return ConvertedDoc(title=doc_title, source_file=path.name, format=path.suffix.lower().lstrip("."), sections=sections, paragraphs=len(paragraphs), mode=mode)


def _guess_title(paragraphs: list[str], path: Path) -> str:
    first = paragraphs[0]
    if first.startswith("# "):
        return first[2:].strip()
    if 2 <= len(first) <= _MAX_HEADING_LEN and not first.endswith(("다.", "다", ".")) and not first.startswith("|"):
        return first
    return path.stem


def _detect_mode(paragraphs: list[str]) -> str:
    articles = sum(1 for p in paragraphs if _ARTICLE.match(p))
    if articles >= 2:
        return "article"
    numbered = sum(1 for p in paragraphs if _is_numbered_heading(p))
    if numbered >= 2:
        return "numbered"
    return "plain"


def _is_numbered_heading(p: str) -> bool:
    m = _NUMBERED.match(p)
    if not m:
        return False
    text = m.group(2)
    return len(p) <= _MAX_HEADING_LEN and not re.search(r"(다|음|함|됨|임)\.?$", text) and not text.startswith("|")


def _structure(paragraphs: list[str], mode: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_name = ""
    current: list[str] = []
    chapter = ""

    def flush() -> None:
        nonlocal current
        if current or current_name:
            sections.append((current_name, current))
        current = []

    for p in paragraphs:
        stripped = p.lstrip("#").strip() if p.startswith("#") else p
        is_heading = False
        name = ""
        if mode == "article":
            if _CHAPTER.match(stripped) and len(stripped) <= _MAX_HEADING_LEN:
                chapter = stripped
                flush()
                current_name = chapter
                continue
            m = _ARTICLE.match(stripped)
            if m and len(stripped.split(")")[0]) <= _MAX_HEADING_LEN:
                head = f"제{m.group(1)}조"
                label = m.group(2) or m.group(3)
                name = f"{head}({label})" if label else head
                name = f"{chapter} › {name}" if chapter else name
                is_heading = True
                rest = (m.group(4) or "").strip()
                flush()
                current_name = name
                if rest:
                    current.append(rest)
                continue
            if _APPENDIX.match(stripped) and len(stripped) <= _MAX_HEADING_LEN:
                chapter = ""
                flush()
                current_name = stripped
                continue
        elif mode == "numbered":
            if _is_numbered_heading(stripped):
                is_heading = True
                name = stripped
        if p.startswith("## ") or (p.startswith("# ") and sections):
            is_heading = True
            name = stripped
        if is_heading:
            flush()
            current_name = name
        else:
            current.append(p)
    flush()
    # 섹션명 없는 앞부분이 비어 있으면 제거
    return [(n, ps) for n, ps in sections if ps or n]


def to_markdown(doc: ConvertedDoc, *, category: str = "", effective_date: str = "", version: str = "", tags: list[str] | None = None, extra: dict[str, Any] | None = None) -> str:
    meta: dict[str, Any] = {
        "title": doc.title,
        "category": category,
        "effective_date": effective_date,
        "version": version,
        "source_file": doc.source_file,
        "tags": tags or [],
        "converted": date.today().isoformat(),
    }
    meta.update(extra or {})
    lines = [render_front_matter(meta), f"# {doc.title}", ""]
    for name, paragraphs in doc.sections:
        if name:
            lines.append(f"## {name}")
            lines.append("")
        for p in paragraphs:
            lines.append(p)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|\s]+", "_", name).strip("_.")
    return stem[:80] or "document"


def ingest_paths(paths: list[Path | str], out_dir: Path | str, *, category: str = "", effective_date: str = "", version: str = "", tags: list[str] | None = None, title: str | None = None, overwrite: bool = False, dry_run: bool = False) -> list[IngestResult]:
    """파일·폴더 목록을 변환해 out_dir 에 .md 로 저장한다. 폴더는 지원 형식을 재귀적으로 찾는다."""
    from .extract import SUPPORTED_EXTENSIONS

    out_dir = Path(out_dir)
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS))
        else:
            files.append(p)
    results: list[IngestResult] = []
    for f in files:
        if not f.exists():
            results.append(IngestResult(source=f, output=None, doc=None, error="파일이 없습니다."))
            continue
        try:
            doc = convert_file(f, title=title if len(files) == 1 else None)
        except ExtractError as e:
            results.append(IngestResult(source=f, output=None, doc=None, error=str(e)))
            continue
        except Exception as e:  # noqa: BLE001 - 한 파일의 실패가 전체를 멈추면 안 된다
            results.append(IngestResult(source=f, output=None, doc=None, error=f"{type(e).__name__}: {e}"))
            continue
        target = out_dir / f"{_safe_stem(f.stem)}.md"
        if target.exists() and not overwrite:
            results.append(IngestResult(source=f, output=target, doc=doc, skipped=True))
            continue
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(to_markdown(doc, category=category, effective_date=effective_date, version=version, tags=tags), encoding="utf-8")
        results.append(IngestResult(source=f, output=target, doc=doc))
    return results
