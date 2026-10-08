"""평가 체계: "잘 답하는가" 를 느낌이 아니라 숫자로 본다.

질문셋(JSONL, 한 줄에 한 질문):
    {"id": "q01", "question": "거수자 발견 시 조치 절차는?", "expected_refs": ["거수자 조치"], "expected_keywords": ["정지", "보고"], "forbidden": ["사살"], "category": "경계"}

- expected_refs: 검색 결과(제목 › 섹션)에 포함되어야 할 문자열. 하나라도 맞으면 적중.
- expected_keywords: 답변에 들어 있어야 할 단어(모두). 동의어는 "A|B" 로 적는다.
  영문·숫자 한 글자(예: "I", "5")는 앞뒤에 영문·숫자가 붙지 않은 경우에만 맞은 것으로 본다("Time" 의 i, "2025" 의 5 는 제외).
- forbidden: 답변에 있으면 안 되는 단어.
- blocked: true 면 보안 차단이 일어나야 정답.

두 단계로 평가한다.
1) 검색 평가 (모델 호출 없음, 무료·수 초): hit@k, MRR
2) 답변 평가 (모델 호출): 키워드 충족률, 금지어 위반, 차단 정확도, 토큰 사용량(입력·캐시 읽기·출력)
결과는 JSON 으로 저장해 이전 결과와 비교(--compare)할 수 있다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..knowledge import DocumentStore


@dataclass
class EvalCase:
    id: str
    question: str
    expected_refs: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    blocked: bool = False
    category: str = ""
    note: str = ""


def load_cases(path: Path | str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} {n}행: JSON 형식 오류 ({e})") from e
        if "question" not in raw:
            raise ValueError(f"{path} {n}행: question 이 없습니다.")
        raw.setdefault("id", f"q{len(cases) + 1:03d}")
        if raw["id"] in seen:
            raise ValueError(f"{path} {n}행: id '{raw['id']}' 가 중복됩니다.")
        seen.add(raw["id"])
        known = {f for f in EvalCase.__dataclass_fields__}
        cases.append(EvalCase(**{k: v for k, v in raw.items() if k in known}))
    if not cases:
        raise ValueError(f"{path}: 질문이 없습니다.")
    return cases


# ---- 검색 평가 ----------------------------------------------------------
@dataclass
class RetrievalResult:
    id: str
    question: str
    expected_refs: list[str]
    top_refs: list[str]
    rank: int | None  # 첫 번째로 맞춘 결과의 순위 (1부터), 없으면 None
    hit: bool
    skipped: bool = False


def _ref_matches(ref: str, expected: list[str]) -> bool:
    r = ref.lower()
    return any(e.lower() in r for e in expected)


def evaluate_retrieval(store: DocumentStore, cases: list[EvalCase], *, k: int = 4) -> dict[str, Any]:
    results: list[RetrievalResult] = []
    for c in cases:
        if not c.expected_refs or c.blocked:
            results.append(RetrievalResult(c.id, c.question, c.expected_refs, [], None, False, skipped=True))
            continue
        hits = store.search(c.question, top_k=k)
        refs = [h.chunk.ref for h in hits]
        rank = next((i for i, r in enumerate(refs, 1) if _ref_matches(r, c.expected_refs)), None)
        results.append(RetrievalResult(c.id, c.question, c.expected_refs, refs, rank, rank is not None))
    scored = [r for r in results if not r.skipped]
    n = len(scored)
    return {
        "k": k,
        "cases": n,
        "hit_at_k": round(sum(1 for r in scored if r.hit) / n, 4) if n else None,
        "hit_at_1": round(sum(1 for r in scored if r.rank == 1) / n, 4) if n else None,
        "mrr": round(sum(1.0 / r.rank for r in scored if r.rank) / n, 4) if n else None,
        "misses": [{"id": r.id, "question": r.question, "expected": r.expected_refs, "got": r.top_refs} for r in scored if not r.hit],
        "results": [asdict(r) for r in results],
    }


# ---- 답변 평가 ----------------------------------------------------------
@dataclass
class AnswerResult:
    id: str
    question: str
    answer: str
    blocked: bool
    expected_blocked: bool
    keywords_hit: list[str]
    keywords_missed: list[str]
    forbidden_found: list[str]
    tools_called: list[str]
    passed: bool
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    cache_read_tokens: int | None = None  # 프롬프트 캐시에서 읽은 입력(시스템 프롬프트·도구 정의). input_tokens 에는 포함되지 않는다.


def _alt_present(t: str, alt: str) -> bool:
    if len(alt) == 1 and alt.isascii() and alt.isalnum():
        # 한 글자 영문·숫자는 부분 문자열로 보면 "Time"·"2025" 같은 무관한 단어에도 걸린다.
        return re.search(rf"(?<![a-z0-9]){re.escape(alt)}(?![a-z0-9])", t) is not None
    return alt in t


def _keyword_present(text: str, keyword: str) -> bool:
    t = text.lower()
    return any(_alt_present(t, alt.strip().lower()) for alt in keyword.split("|") if alt.strip())


def evaluate_answers(ask: Callable[[str], Any], cases: list[EvalCase], *, on_progress: Callable[[int, int, AnswerResult], None] | None = None) -> dict[str, Any]:
    """`ask(question)` 는 ChatResult(또는 text/blocked/tools_called/usage 속성이 있는 객체)를 돌려줘야 한다."""
    results: list[AnswerResult] = []
    for i, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            r = ask(c.question)
            answer, blocked, tools = r.text, bool(r.blocked), list(getattr(r, "tools_called", []) or [])
            usage = getattr(r, "usage", {}) or {}
            error = None
        except Exception as e:  # noqa: BLE001 - 한 질문의 실패가 전체 평가를 멈추면 안 된다
            answer, blocked, tools, usage, error = "", False, [], {}, f"{type(e).__name__}: {e}"
        seconds = round(time.perf_counter() - t0, 2)
        hit = [kw for kw in c.expected_keywords if _keyword_present(answer, kw)]
        missed = [kw for kw in c.expected_keywords if kw not in hit]
        forbidden = [w for w in c.forbidden if _keyword_present(answer, w)]
        if c.blocked:
            passed = blocked and error is None
        else:
            passed = error is None and not blocked and not missed and not forbidden
        res = AnswerResult(c.id, c.question, answer, blocked, c.blocked, hit, missed, forbidden, tools, passed, seconds, usage.get("input_tokens"), usage.get("output_tokens"), error, usage.get("cache_read_input_tokens"))
        results.append(res)
        if on_progress:
            on_progress(i, len(cases), res)
    n = len(results)
    kw_total = sum(len(c.expected_keywords) for c in cases if not c.blocked)
    kw_hit = sum(len(r.keywords_hit) for r in results if not r.expected_blocked)
    block_cases = [r for r in results if r.expected_blocked]
    return {
        "cases": n,
        "pass_rate": round(sum(1 for r in results if r.passed) / n, 4) if n else None,
        "keyword_rate": round(kw_hit / kw_total, 4) if kw_total else None,
        "forbidden_violations": sum(1 for r in results if r.forbidden_found),
        "block_accuracy": round(sum(1 for r in block_cases if r.blocked) / len(block_cases), 4) if block_cases else None,
        "errors": sum(1 for r in results if r.error),
        "avg_seconds": round(sum(r.seconds for r in results) / n, 2) if n else None,
        "total_input_tokens": sum(r.input_tokens or 0 for r in results),
        "total_output_tokens": sum(r.output_tokens or 0 for r in results),
        "total_cache_read_tokens": sum(r.cache_read_tokens or 0 for r in results),
        "failures": [{"id": r.id, "question": r.question, "missed": r.keywords_missed, "forbidden": r.forbidden_found, "blocked": r.blocked, "expected_blocked": r.expected_blocked, "error": r.error} for r in results if not r.passed],
        "results": [asdict(r) for r in results],
    }


# ---- 보고서 ---------------------------------------------------------------
@dataclass
class EvalReport:
    created_at: str
    cases_file: str
    model: str
    retrieval: dict[str, Any] | None
    answers: dict[str, Any] | None

    @classmethod
    def build(cls, cases_file: str, model: str, retrieval: dict[str, Any] | None, answers: dict[str, Any] | None) -> "EvalReport":
        return cls(datetime.now(timezone.utc).isoformat(timespec="seconds"), cases_file, model, retrieval, answers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "EvalReport":
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(d, dict) or not all(isinstance(d.get(k), str) for k in ("created_at", "cases_file")):
            raise ValueError(f"{path}: 평가 보고서 형식이 아닙니다 (created_at·cases_file 이 없음).")
        for k in ("retrieval", "answers"):
            if d.get(k) is not None and not isinstance(d[k], dict):
                raise ValueError(f"{path}: 평가 보고서 형식이 아닙니다 ({k} 가 객체가 아님).")
        return cls(d["created_at"], d["cases_file"], d.get("model", ""), d.get("retrieval"), d.get("answers"))

    def summary_lines(self) -> list[str]:
        lines = [f"평가 시각: {self.created_at}  질문셋: {self.cases_file}  모델: {self.model}"]
        if self.retrieval:
            r = self.retrieval
            lines.append(f"[검색] 질문 {r['cases']}개  hit@{r['k']} {_pct(r['hit_at_k'])}  hit@1 {_pct(r['hit_at_1'])}  MRR {r['mrr']}")
            for m in r["misses"][:10]:
                lines.append(f"   ✖ {m['id']} {m['question']}  (기대: {', '.join(m['expected'])} / 결과: {', '.join(x.split(' › ')[-1] for x in m['got'][:3]) or '없음'})")
        if self.answers:
            a = self.answers
            lines.append(f"[답변] 질문 {a['cases']}개  통과율 {_pct(a['pass_rate'])}  키워드 충족 {_pct(a['keyword_rate'])}  금지어 위반 {a['forbidden_violations']}  차단 정확도 {_pct(a['block_accuracy'])}  오류 {a['errors']}  평균 {a['avg_seconds']}초  토큰 입력 {a['total_input_tokens']}" + (f" (+캐시 읽기 {a['total_cache_read_tokens']})" if a.get("total_cache_read_tokens") else "") + f" / 출력 {a['total_output_tokens']}")
            for f in a["failures"][:10]:
                why = f["error"] or (f"누락 {f['missed']}" if f["missed"] else "") + (f" 금지어 {f['forbidden']}" if f["forbidden"] else "") + (" 차단됨" if f["blocked"] and not f.get("expected_blocked") else "") + (" 차단 안 됨(답변 생성됨)" if f.get("expected_blocked") and not f["blocked"] else "")
                lines.append(f"   ✖ {f['id']} {f['question']}  ({why.strip()})")
        return lines


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def compare_reports(before: EvalReport, after: EvalReport) -> list[str]:
    """두 보고서의 핵심 지표를 나란히 보여 주고, 질문별로 좋아진 것·나빠진 것을 찾는다."""
    lines = [f"이전: {before.created_at} ({before.model})  →  현재: {after.created_at} ({after.model})"]

    def row(label: str, b: float | None, a: float | None, pct: bool = True) -> None:
        if b is None and a is None:
            return
        fmt = (lambda v: _pct(v)) if pct else (lambda v: "–" if v is None else str(v))
        if b is None or a is None:
            delta = ""
        elif pct:
            delta = f"  ({(a - b) * 100:+.1f}%p)"
        elif isinstance(a, int) and isinstance(b, int):
            delta = f"  ({a - b:+d})"
        else:  # MRR 처럼 0~1 범위 실수는 소수 1자리로는 변화가 ±0.0 으로 묻힌다
            delta = f"  ({a - b:+.4f})"
        lines.append(f"  {label:14s} {fmt(b):>8s} → {fmt(a):>8s}{delta}")

    if before.retrieval and after.retrieval:
        row("검색 hit@k", before.retrieval["hit_at_k"], after.retrieval["hit_at_k"])
        row("검색 hit@1", before.retrieval["hit_at_1"], after.retrieval["hit_at_1"])
        row("검색 MRR", before.retrieval["mrr"], after.retrieval["mrr"], pct=False)
        b_hit = {r["id"]: r["hit"] for r in before.retrieval["results"] if not r["skipped"]}
        a_hit = {r["id"]: r["hit"] for r in after.retrieval["results"] if not r["skipped"]}
        improved = sorted(i for i in a_hit if a_hit[i] and b_hit.get(i) is False)
        regressed = sorted(i for i in a_hit if not a_hit[i] and b_hit.get(i) is True)
        if improved:
            lines.append(f"  검색 개선: {', '.join(improved)}")
        if regressed:
            lines.append(f"  검색 퇴보: {', '.join(regressed)}")
    if before.answers and after.answers:
        row("답변 통과율", before.answers["pass_rate"], after.answers["pass_rate"])
        row("키워드 충족", before.answers["keyword_rate"], after.answers["keyword_rate"])
        row("금지어 위반", before.answers["forbidden_violations"], after.answers["forbidden_violations"], pct=False)
        b_pass = {r["id"]: r["passed"] for r in before.answers["results"]}
        a_pass = {r["id"]: r["passed"] for r in after.answers["results"]}
        improved = sorted(i for i in a_pass if a_pass[i] and b_pass.get(i) is False)
        regressed = sorted(i for i in a_pass if not a_pass[i] and b_pass.get(i) is True)
        if improved:
            lines.append(f"  답변 개선: {', '.join(improved)}")
        if regressed:
            lines.append(f"  답변 퇴보: {', '.join(regressed)}")
    return lines

