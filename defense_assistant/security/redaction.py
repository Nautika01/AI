"""민감정보(개인식별정보·군사 식별정보) 마스킹.

모델로 전송되기 전에 주민등록번호, 군번, 전화번호, 이메일, 군사 좌표(MGRS·위경도),
내부망 IP 등을 자리표시자로 치환한다. 원문은 로컬에만 남고 외부로 나가지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def changed(self) -> bool:
        return self.total > 0

    def summary(self) -> str:
        if not self.changed:
            return "마스킹 없음"
        return ", ".join(f"{k} {v}건" for k, v in self.counts.items())


# 순서가 중요하다: 더 구체적인 패턴(주민등록번호)을 군번보다 먼저 처리한다.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("주민등록번호", re.compile(r"(?<!\d)\d{6}\s*-\s*[1-8]\d{6}(?!\d)")),
    ("이메일", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("휴대전화", re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")),
    ("전화번호", re.compile(r"(?<!\d)0(?:2|[3-6]\d)[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")),
    # 군번: 장교 YY-NNNNN(N), 부사관/병 YY-NNNNNNNN (주민등록번호와 구분하기 위해 앞자리 2자리로 한정)
    ("군번", re.compile(r"(?<![\d-])\d{2}-\d{5,8}(?![\d-])")),
    # MGRS 좌표: 52S CG 12345 67890 / 52SCG1234567890
    ("좌표", re.compile(r"(?<![A-Z0-9])\d{1,2}[C-HJ-NP-X]\s?[A-HJ-NP-Z]{2}\s?(?:\d{2}\s?\d{2}|\d{3}\s?\d{3}|\d{4}\s?\d{4}|\d{5}\s?\d{5}|\d{4,10})(?![A-Z0-9])")),
    # 위경도: 37.5665, 126.9780 / N37 33.99 E126 58.68
    ("좌표", re.compile(r"(?<![\d.])-?\d{1,2}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}(?![\d.])")),
    ("좌표", re.compile(r"\b[NS]\s?\d{1,2}[°\s]\s?\d{1,2}(?:\.\d+)?['\s]?\s*[EW]\s?\d{1,3}[°\s]\s?\d{1,2}(?:\.\d+)?'?")),
    # 사설/내부망 IP
    ("내부IP", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")),
]


def redact(text: str, enabled: bool = True) -> RedactionResult:
    """민감정보를 `[종류]` 자리표시자로 치환한다."""
    if not enabled or not text:
        return RedactionResult(text=text)
    counts: dict[str, int] = {}
    out = text
    for label, pattern in _RULES:
        out, n = pattern.subn(f"[{label}]", out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(text=out, counts=counts)
