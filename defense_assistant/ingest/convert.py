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
from .extract import ExtractError, extract_paragraphs, read_front_matter

# "제5조(목적)", "제5조의2[특례]", "제7조 삭제". '조'(·'의N') 바로 뒤는 공백·괄호·줄끝이어야 한다("제5조에 따라" 제외)
_ARTICLE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?=[\s(\[]|$)\s*(?:\(([^)]*)\)|\[([^\]]*)\])?\s*(.*)$")
# 라벨 없는 조문 뒤에 이런 말이 오면 다른 조문을 가리키는 본문 문장이다("제5조 및 제6조에 따라", "제5조 제2항에")
_ARTICLE_REF_REST = re.compile(r"^(?:및|또는|내지|부터|까지|에\s|에서\s|의\s|제\s*\d+\s*[조항호])")
_CHAPTER = re.compile(r"^제\s*\d+\s*[장절관편]\s+(\S.*)$")
_APPENDIX = re.compile(r"^(부칙|별표\s*\d*|별지\s*(?:제\s*\d+\s*호)?\s*(?:서식)?)\b.*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+\.?|\d+[.)])\s+(\S.*)$")  # "1. 개요", "2.1 절차", "3) 항목"
_DATE_LINE = re.compile(r"^\d{2,4}\.\s*\d{1,2}\.(?:\s*\d{1,2}\.?)?(?:\s|$)")  # "2025. 3. 1." 개정 이력 등
_UNIT_AFTER_DECIMAL = re.compile(r"^(?:km/h|km|cm|mm|m|kg|g|t|ton|톤|L|l|mL|ml|mil|nm|NM|kt|ft|in|%|°|도|시간|분|초)(?=$|[\s/,.)])")
_SENTENCE_END = re.compile(r"(?:다|음|함|됨|임)\.?$|\.$")
_MAX_HEADING_LEN = 60


def _match_article(p: str) -> re.Match | None:
    """조문 머리면 매치를 돌려준다. 그룹: 1 조 번호, 2 가지번호, 3·4 라벨, 5 나머지 본문."""
    m = _ARTICLE.match(p)
    if not m:
        return None
    if not (m.group(3) or m.group(4)) and _ARTICLE_REF_REST.match(m.group(5) or ""):
        return None
    return m


def _is_chapter(p: str) -> bool:
    """'제2장 근무 편성' 같은 장·절 제목. '제2장의 규정을 준용한다.', '제3절 제12조부터 …' 같은 문장은 제외."""
    m = _CHAPTER.match(p)
    if not m or len(p) > _MAX_HEADING_LEN:
        return False
    rest = m.group(1)
    return not re.match(r"제\s*\d+\s*[조항호]", rest) and not _SENTENCE_END.search(rest)


@dataclass
class ConvertedDoc:
    title: str
    source_file: str
    format: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)  # (섹션명, 문단들)
    paragraphs: int = 0
    mode: str = "plain"  # article | numbered | plain
    meta: dict[str, Any] = field(default_factory=dict)  # 입력 .md/.txt 에 이미 있던 머리말(재변환 시 보존)

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
    meta = read_front_matter(path)
    doc_title = title or meta.get("title") or _guess_title(paragraphs, path)
    first = paragraphs[0]
    # 제목으로 쓴 첫 문단('# 제목' 포함)은 본문에서 뺀다. to_markdown 이 '# 제목' 을 다시 붙인다
    drop_first = not title and (first == doc_title or (first.startswith("# ") and first[2:].strip() == doc_title))
    body = paragraphs[1:] if drop_first else paragraphs
    mode = _detect_mode(body)
    sections = _structure(body, mode)
    return ConvertedDoc(title=doc_title, source_file=path.name, format=path.suffix.lower().lstrip("."), sections=sections, paragraphs=len(paragraphs), mode=mode, meta=meta)


def _guess_title(paragraphs: list[str], path: Path) -> str:
    first = paragraphs[0]
    if first.startswith("# "):
        return first[2:].strip()
    if not (2 <= len(first) <= _MAX_HEADING_LEN) or first.endswith(("다.", "다", ".")) or first.startswith("|"):
        return path.stem
    # 구분선('---')·'key: value' 머리말 줄이나 장·조문·부칙·번호 제목은 문서 제목이 아니다
    if re.fullmatch(r"[-=*_~\s]{3,}", first) or re.match(r"^[A-Za-z_][\w-]*:\s", first):
        return path.stem
    if _is_chapter(first) or _match_article(first) or _APPENDIX.match(first) or _NUMBERED.match(first):
        return path.stem
    return first


def _detect_mode(paragraphs: list[str]) -> str:
    articles = sum(1 for p in paragraphs if _match_article(p))
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
    number, text = m.group(1), m.group(2)
    if _DATE_LINE.match(p):  # "2025. 3. 1."
        return False
    if re.fullmatch(r"\d+\.\d+", number) and _UNIT_AFTER_DECIMAL.match(text):  # "2.5 km 이동"
        return False
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
            if _is_chapter(stripped):
                chapter = stripped
                flush()
                current_name = chapter
                continue
            m = _match_article(stripped)
            if m and len(stripped.split(")")[0]) <= _MAX_HEADING_LEN:
                head = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
                label = m.group(3) or m.group(4)
                name = f"{head}({label})" if label else head
                name = f"{chapter} › {name}" if chapter else name
                is_heading = True
                rest = (m.group(5) or "").strip()
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
    # 입력에 이미 있던 머리말 값은 인자로 새 값을 주지 않은 항목에 한해 그대로 둔다
    old = doc.meta or {}
    meta: dict[str, Any] = {
        "title": doc.title,
        "category": category or old.get("category", ""),
        "effective_date": effective_date or old.get("effective_date", ""),
        "version": version or old.get("version", ""),
        "source_file": old.get("source_file") or doc.source_file,
        "tags": tags or old.get("tags") or [],
    }
    meta.update({k: v for k, v in old.items() if k not in meta and k != "converted"})
    meta["converted"] = date.today().isoformat()
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
    seen: set[Path] = set()
    for p in paths:
        p = Path(p)
        found = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS) if p.is_dir() else [p]
        for f in found:
            key = f.resolve()
            if key not in seen:  # 같은 파일을 두 번 넘겨도 한 번만 변환한다
                seen.add(key)
                files.append(f)
    results: list[IngestResult] = []
    assigned: dict[str, Path] = {}  # 이번 실행에서 배정한 출력 파일 이름(대소문자 무시) → 원본
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
        # 이름 충돌은 기존 파일의 건너뛰기·덮어쓰기 판정보다 먼저 가린다(같은 실행 안의 앞 문서를 덮어쓰지 않게)
        target = out_dir / _unique_name(f, assigned)
        assigned[target.name.casefold()] = f
        if target.exists() and not overwrite:
            results.append(IngestResult(source=f, output=target, doc=doc, skipped=True))
            continue
        if not dry_run:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(to_markdown(doc, category=category, effective_date=effective_date, version=version, tags=tags), encoding="utf-8")
            except Exception as e:  # noqa: BLE001 - 한 파일의 저장 실패가 전체를 멈추면 안 된다
                results.append(IngestResult(source=f, output=None, doc=doc, error=f"저장하지 못했습니다 ({target}): {type(e).__name__}: {e}"))
                continue
        results.append(IngestResult(source=f, output=target, doc=doc))
    return results


def _unique_name(f: Path, assigned: dict[str, Path]) -> str:
    """이번 실행에서 아직 쓰지 않은 출력 파일 이름.

    기본은 '원본이름.md'. 이미 다른 원본에 배정된 이름이면, 앞 원본과 폴더가 다를 때는 상위 폴더명을
    (2대대_경계지침.md), 같은 폴더면 확장자를(규정_pdf.md) 붙인다. 그래도 겹치면 _2, _3 … 을 붙인다.
    파일 순서가 같으면 실행할 때마다 같은 이름이 나온다.
    """
    stem = _safe_stem(f.stem)
    first = f"{stem}.md"
    holder = assigned.get(first.casefold())
    if holder is None:
        return first
    ext = f.suffix.lower().lstrip(".")
    parent = _safe_stem(f.parent.name) if f.parent.name else ""
    by_parent = [f"{parent}_{stem}"] if parent else []
    by_ext = [f"{stem}_{ext}"] if ext else []
    candidates = by_parent + by_ext if holder.parent != f.parent else by_ext + by_parent
    if parent and ext:
        candidates.append(f"{parent}_{stem}_{ext}")
    for c in candidates:
        name = f"{_safe_stem(c)}.md"
        if name.casefold() not in assigned:
            return name
    base = _safe_stem(candidates[-1] if candidates else stem)[:76]
    n = 2
    while f"{base}_{n}.md".casefold() in assigned:
        n += 1
    return f"{base}_{n}.md"
