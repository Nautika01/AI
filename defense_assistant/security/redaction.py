"""민감정보(개인식별정보·군사 식별정보) 마스킹.

모델로 전송되기 전에 주민등록번호, 군번, 전화번호, 이메일, 군사 좌표(MGRS·위경도),
내부망 IP 등을 자리표시자로 치환한다. 원문은 로컬에만 남고 외부로 나가지 않는다.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
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


# 매칭용 정규화: 전각 숫자·기호(NFKC), 서식 문자(Cf) 제거, 각종 대시 → '-'.
# 치환은 원문 위치에 되돌려 적용하므로, 마스킹되지 않은 부분은 원문 그대로 남는다.
_DASHES = str.maketrans({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})


def _matching_view(text: str) -> tuple[str, list[int] | None]:
    """정규화된 매칭용 문자열과, 각 문자가 유래한 원문 인덱스 목록을 돌려준다.

    정규화로 바뀌는 글자가 없으면 인덱스 목록 대신 None(원문과 위치 동일)을 돌려준다.
    """
    if text.isascii():
        return text, None
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isascii() or "\uac00" <= ch <= "\ud7a3":  # ASCII·완성형 한글은 NFKC 불변
            chars.append(ch)
            index.append(i)
            continue
        if unicodedata.category(ch) == "Cf":
            continue
        for c in unicodedata.normalize("NFKC", ch).translate(_DASHES):
            if unicodedata.category(c) != "Cf":
                chars.append(c)
                index.append(i)
    return "".join(chars), index


# --- 매치 검증기: 모양만 비슷한 일반 수치·약어의 과잉 마스킹을 줄인다 ---

def _valid_decimal_pair(m: re.Match[str]) -> bool:
    """위경도 범위(|위도|≤90, |경도|≤180) 안이고, 둘 다 |값|<1 인 측정값 쌍(0.001, 0.002)이 아닐 것."""
    lat, lon = float(m.group("lat")), float(m.group("lon"))
    if abs(lat) > 90 or abs(lon) > 180:
        return False
    return not (abs(lat) < 1 and abs(lon) < 1)


_MGRS_COLUMNS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")  # UTM 구역 번호 % 3 에 따른 100km 열 문자 집합
_MGRS_ROWS = "ABCDEFGHJKLMNPQRSTUV"  # 100km 행 문자(A~V, I·O 제외)


def _valid_mgrs(m: re.Match[str]) -> bool:
    """구역 1~60, 구역에 맞는 100km 격자 문자, 짝수 자리 숫자부일 때만 MGRS 로 본다."""
    zone = int(m.group("zone"))
    if not 1 <= zone <= 60:
        return False
    if m.group("col") not in _MGRS_COLUMNS[(zone - 1) % 3] or m.group("row") not in _MGRS_ROWS:
        return False
    return len(re.sub(r"\s", "", m.group("digits"))) % 2 == 0


_VERSION_CONTEXT = re.compile(r"(?:버전|펌웨어|[Vv]er(?:sion)?\.?)\s*[:：]?\s*$")


def _valid_private_ip(m: re.Match[str]) -> bool:
    """옥텟이 0~255 이고, 바로 앞이 '버전'·'ver' 같은 버전 표기 문맥이 아닐 것."""
    if any(int(o) > 255 for o in m.group(0).split(".")):
        return False
    return not _VERSION_CONTEXT.search(m.string[max(0, m.start() - 12) : m.start()])


# 도(°)·분(′)·초(″) 성분. NFKC 후 '″' 는 '′′' 로 바뀌므로 함께 받는다.
_MIN = r"(?:\d{1,2}(?:\.\d+)?\s?['′])"
_SEC = r"(?:\d{1,2}(?:\.\d+)?\s?(?:\"|″|′′|''))"
_LAT_DMS = rf"\d{{1,2}}(?:\.\d+)?°\s?(?:{_MIN}\s?)?(?:{_SEC}\s?)?"
_LON_DMS = rf"\d{{1,3}}(?:\.\d+)?°\s?(?:{_MIN}\s?)?(?:{_SEC}\s?)?"

_Validator = Callable[[re.Match[str]], bool]

# 순서가 중요하다: 더 구체적인 패턴(주민등록번호)을 군번보다 먼저 처리한다.
_RULES: list[tuple[str, re.Pattern[str], _Validator | None]] = [
    # 900101-1234567 / 900101 - 1234567, 그리고 구분자 없는·공백 구분 표기(생년월일 범위로 오탐 축소)
    ("주민등록번호", re.compile(
        r"(?<!\d)(?:\d{6}\s*-\s*[1-8]\d{6}"
        r"|\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s*[1-8]\d{6})(?!\d)"
    ), None),
    ("이메일", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), None),
    # 010-1234-5678 / +82-10-1234-5678 / +82 10 1234 5678 / +821012345678
    ("휴대전화", re.compile(
        r"(?:(?<![\d+])0|(?<![\d+])\+82[-.\s]?(?:\(0\)\s?|0)?)1[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"
    ), None),
    # 02-123-4567 / 02)123-4567 / (02) 123-4567 / 070-1234-5678 / 0505-123-4567 / +82-2-123-4567
    ("전화번호", re.compile(
        r"(?:\(\s*0(?:2|[3-6]\d|70|50\d|80)\s*\)\s*"
        r"|(?:(?<![\d+])0|(?<![\d+])\+82[-.\s]?(?:\(0\)\s?|0)?)(?:2|[3-6]\d|70|50\d|80)(?:\s*\)\s*|[-.\s])?)"
        r"\d{3,4}[-.\s]?\d{4}(?!\d)"
    ), None),
    # 군번: 장교 YY-NNNNN(N), 부사관/병 YY-NNNNNNNN (주민등록번호와 구분하기 위해 앞자리 2자리로 한정)
    # 'FY24-12345' 처럼 영문 접두가 붙은 문서·예산 번호는 제외한다.
    ("군번", re.compile(r"(?<![A-Za-z\d-])\d{2}-\d{5,8}(?![\d-])"), None),
    # MGRS 좌표: 52S CG 12345 67890 / 52SCG1234567890
    ("좌표", re.compile(
        r"(?<![A-Za-z0-9])(?P<zone>\d{1,2})[C-HJ-NP-X]\s?(?P<col>[A-HJ-NP-Z])(?P<row>[A-HJ-NP-Z])\s?"
        r"(?P<digits>\d{5}\s?\d{5}|\d{4}\s?\d{4}|\d{3}\s?\d{3}|\d{2}\s?\d{2}|\d{4,10})(?![A-Za-z0-9])"
    ), _valid_mgrs),
    # 위경도: 37.5665, 126.9780
    ("좌표", re.compile(
        r"(?<![\d.])(?P<lat>-?\d{1,2}\.\d{3,})\s*,\s*(?P<lon>-?\d{1,3}\.\d{3,})(?![\d.])"
    ), _valid_decimal_pair),
    # 공백 구분 십진 쌍: 37.5665 126.9780 — 일반 수치('3.1415 2.7182')와 구분하려고 경도는 100~180 의 세 자리로 한정
    ("좌표", re.compile(
        r"(?<![\d.])(?P<lat>-?\d{1,2}\.\d{3,})\s+(?P<lon>-?1\d{2}\.\d{3,})(?![\d.])"
    ), _valid_decimal_pair),
    # 반구 접미 DMS·도 표기: 37°33'59"N 126°58'41"E / 37.5665°N, 126.9780°E
    ("좌표", re.compile(
        rf"(?<![\d.]){_LAT_DMS}[NS](?:[\s,]*{_LON_DMS}[EW])?(?![A-Za-z])"
    ), None),
    # 반구 접미 십진 표기: 37.5665N 126.9780E
    ("좌표", re.compile(r"(?<![\d.])\d{1,2}\.\d+\s?[NS][\s,]+\d{1,3}\.\d+\s?[EW](?![A-Za-z])"), None),
    # 반구 접두 DMS: N37°33'59" E126°58'41"
    ("좌표", re.compile(
        rf"(?<![A-Za-z])[NS]\s?{_LAT_DMS}[\s,]*[EW]\s?\d{{1,3}}(?:\.\d+)?°\s?(?:{_MIN}\s?)?{_SEC}?"
    ), None),
    # 반구 접두 도·분: N37 33.99 E126 58.68
    ("좌표", re.compile(r"\b[NS]\s?\d{1,2}[°\s]\s?\d{1,2}(?:\.\d+)?['\s]?\s*[EW]\s?\d{1,3}[°\s]\s?\d{1,2}(?:\.\d+)?'?"), None),
    # 사설/내부망 IP (옥텟 0~255, 버전 문자열 제외)
    ("내부IP", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"), _valid_private_ip),
]


def redact(text: str, enabled: bool = True) -> RedactionResult:
    """민감정보를 `[종류]` 자리표시자로 치환한다.

    매칭은 정규화 사본(전각 숫자·서식 문자 처리)에 대해 하고, 치환은 원문 위치에 적용한다.
    """
    if not enabled or not text:
        return RedactionResult(text=text)
    counts: dict[str, int] = {}
    out = text
    view, index = _matching_view(out)
    for label, pattern, validator in _RULES:
        spans: list[tuple[int, int]] = []
        for m in pattern.finditer(view):
            if validator is not None and not validator(m):
                continue
            if index is None:
                start, end = m.start(), m.end()
            else:
                start, end = index[m.start()], index[m.end() - 1] + 1
            if spans and start < spans[-1][1]:
                # 원문 한 글자가 여러 글자로 펼쳐진 경계에서 겹치면 합친다.
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
            counts[label] = counts.get(label, 0) + 1
        if spans:
            parts: list[str] = []
            prev = 0
            for start, end in spans:
                parts.append(out[prev:start])
                parts.append(f"[{label}]")
                prev = end
            parts.append(out[prev:])
            out = "".join(parts)
            view, index = _matching_view(out)
    return RedactionResult(text=out, counts=counts)
