"""한국어·영문 혼용 텍스트용 경량 토크나이저.

외부 형태소 분석기 없이 동작하도록, 한글 음절 연속 구간은 전체 어절과 함께
문자 바이그램(bigram)을 생성한다. 조사·어미 변화에 강건하면서도 설치 부담이 없다.
영문·숫자는 소문자 단어 단위로 처리하고, 하이픈·점·슬래시로 이어진 묶음('K-9', 'METT-TC')은
전체 토큰과 함께 구성 요소와 구분자를 뺀 결합형('k9')도 만들어 표기 차이에 강건하게 한다.
"""

from __future__ import annotations

import re

_HANGUL_RUN = re.compile(r"[가-힣]+")
_LATIN_RUN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_LATIN_SEP = re.compile(r"[.\-/]")
_HAS_ALPHA = re.compile(r"[a-z]")
# 자주 쓰는 영문 약어의 한글 음차 → 영문 토큰 보강.
# 일상 어휘와 겹치는 음차(예: '마치' → MARCH 는 부사 '마치'·동사 '마치다' 활용형과 겹친다)는 넣지 않는다.
_TRANSLITERATIONS = {
    "메데박": "medevac", "메디박": "medevac", "케이스백": "casevac",
    "시트렙": "sitrep", "싯렙": "sitrep", "스팟렙": "spotrep", "스폿렙": "spotrep",
    "프라고": "frago", "옵오더": "opord", "워노드": "warnord",
    "살루트": "salute", "디티지": "dtg",
    "줄루": "zulu", "알오이": "roe", "나토": "nato",
}
_PARTICLES = ("으로는", "에서는", "에게는", "으로", "에서", "에게", "까지", "부터", "이란", "란", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로")


def _strip_particle(word: str) -> str:
    if len(word) <= 2:
        return word
    for p in _PARTICLES:
        if word.endswith(p) and len(word) - len(p) >= 2:
            return word[: -len(p)]
    return word


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    lowered = text.lower()
    for m in _LATIN_RUN.finditer(lowered):
        run = m.group(0)
        tokens.append(run)
        parts = _LATIN_SEP.split(run)
        if len(parts) > 1:
            tokens.extend(parts)
            # 결합형('k-9' → 'k9')은 영문자가 섞인 경우만 만든다 ('3.1' → '31', 날짜 → '20250301' 같은 오탐 방지)
            if _HAS_ALPHA.search(run):
                tokens.append("".join(parts))
    for m in _HANGUL_RUN.finditer(lowered):
        run = m.group(0)
        stem = _strip_particle(run)
        tokens.append(stem)
        for ko, en in _TRANSLITERATIONS.items():
            if ko in stem:
                tokens.append(en)
        if len(stem) >= 3:
            tokens.extend(stem[i : i + 2] for i in range(len(stem) - 1))
    return tokens
