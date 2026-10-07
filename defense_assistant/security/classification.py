"""비밀 등급 표기 탐지 및 처리 허용 여부 판단.

한국군 비밀 등급 체계(I급·II급·III급 비밀, 대외비, 일반)와 영문 표기
(TOP SECRET / SECRET / CONFIDENTIAL / RESTRICTED)를 함께 인식한다.

이 모듈은 *표기(marking)* 를 탐지할 뿐 내용 자체의 비밀성을 판단하지 않는다.
비밀 자료는 인가된 망·장비 밖으로 반출되어서는 안 되므로, 허용 등급을 넘는
표기가 감지되면 모델 호출 전에 차단하는 것이 목적이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class Classification(IntEnum):
    UNCLASSIFIED = 0  # 일반(평문)
    RESTRICTED = 1  # 대외비
    CONFIDENTIAL = 2  # III급 비밀
    SECRET = 3  # II급 비밀
    TOP_SECRET = 4  # I급 비밀

    @property
    def korean(self) -> str:
        return _KOREAN_NAMES[self]

    @classmethod
    def parse(cls, value: str | int | "Classification") -> "Classification":
        if isinstance(value, Classification):
            return value
        if isinstance(value, int):
            return cls(value)
        key = value.strip().upper().replace(" ", "_").replace("-", "_")
        aliases = {
            "UNCLASSIFIED": cls.UNCLASSIFIED,
            "U": cls.UNCLASSIFIED,
            "일반": cls.UNCLASSIFIED,
            "평문": cls.UNCLASSIFIED,
            "RESTRICTED": cls.RESTRICTED,
            "R": cls.RESTRICTED,
            "FOUO": cls.RESTRICTED,
            "대외비": cls.RESTRICTED,
            "CONFIDENTIAL": cls.CONFIDENTIAL,
            "C": cls.CONFIDENTIAL,
            "III급": cls.CONFIDENTIAL,
            "3급": cls.CONFIDENTIAL,
            "SECRET": cls.SECRET,
            "S": cls.SECRET,
            "II급": cls.SECRET,
            "2급": cls.SECRET,
            "TOP_SECRET": cls.TOP_SECRET,
            "TS": cls.TOP_SECRET,
            "I급": cls.TOP_SECRET,
            "1급": cls.TOP_SECRET,
        }
        if key not in aliases:
            raise ValueError(f"알 수 없는 비밀 등급: {value!r}")
        return aliases[key]


_KOREAN_NAMES = {
    Classification.UNCLASSIFIED: "일반",
    Classification.RESTRICTED: "대외비",
    Classification.CONFIDENTIAL: "III급 비밀",
    Classification.SECRET: "II급 비밀",
    Classification.TOP_SECRET: "I급 비밀",
}

# 각 패턴은 명시적 표기만 잡도록 보수적으로 작성한다. 영문 표기는 관례대로 대문자만 인식한다.
# ("비밀번호", "비밀스러운" 같은 일상어에 반응하면 안 된다.)
_PATTERNS: list[tuple[Classification, re.Pattern[str]]] = [
    # I급 비밀 / 1급비밀 / 1급 기밀 / 극비 / TOP SECRET / TS//
    (Classification.TOP_SECRET, re.compile(r"(?:(?<![IVX\d])I(?![IVX])|(?<!\d)1(?!\d))\s*급\s*(?:비밀|기밀)")),
    (Classification.TOP_SECRET, re.compile(r"극비")),
    (Classification.TOP_SECRET, re.compile(r"\bTOP\s*SECRET\b")),
    (Classification.TOP_SECRET, re.compile(r"(?:^|[\s(\[/])TS(?:\s*//|\s*/\s*[A-Z]|\s*[\])])")),
    # II급 비밀 / 2급비밀 / SECRET
    (Classification.SECRET, re.compile(r"(?:(?<![IVX\d])II(?![IVX])|(?<!\d)2(?!\d))\s*급\s*(?:비밀|기밀)")),
    (Classification.SECRET, re.compile(r"(?<!TOP )(?<!TOP)\bSECRET\b(?!\s*번호)")),
    # III급 비밀 / 3급비밀 / CONFIDENTIAL / 군사기밀 / 비밀문서
    (Classification.CONFIDENTIAL, re.compile(r"(?:(?<![IVX\d])III(?![IVX])|(?<!\d)3(?!\d))\s*급\s*(?:비밀|기밀)")),
    (Classification.CONFIDENTIAL, re.compile(r"\bCONFIDENTIAL\b")),
    (Classification.CONFIDENTIAL, re.compile(r"군사\s*(?:비밀|기밀)")),
    (Classification.CONFIDENTIAL, re.compile(r"비밀\s*(?:문서|자료|취급|등급\s*[:：]\s*(?:비밀|기밀))")),
    # 대외비 / RESTRICTED / FOUO
    (Classification.RESTRICTED, re.compile(r"대외비")),
    (Classification.RESTRICTED, re.compile(r"\bRESTRICTED\b")),
    (Classification.RESTRICTED, re.compile(r"\bFOUO\b")),
    (Classification.RESTRICTED, re.compile(r"\bFOR\s+OFFICIAL\s+USE\s+ONLY\b")),
]


@dataclass
class ClassificationResult:
    level: Classification
    markings: list[str] = field(default_factory=list)

    @property
    def is_marked(self) -> bool:
        return self.level > Classification.UNCLASSIFIED

    def allowed_under(self, max_level: Classification) -> bool:
        return self.level <= max_level


def classify_text(text: str) -> ClassificationResult:
    """텍스트에서 가장 높은 비밀 등급 표기를 찾는다. 표기가 없으면 UNCLASSIFIED."""
    highest = Classification.UNCLASSIFIED
    markings: list[str] = []
    for level, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            markings.append(m.group(0).strip())
            if level > highest:
                highest = level
    # 중복 제거(순서 유지)
    seen: set[str] = set()
    unique = [m for m in markings if not (m in seen or seen.add(m))]
    return ClassificationResult(level=highest, markings=unique)


def block_message(result: ClassificationResult, max_level: Classification) -> str:
    """차단 시 사용자에게 보여 줄 안내문."""
    found = ", ".join(f"'{m}'" for m in result.markings[:5])
    return (
        f"⚠️ 입력에서 {result.level.korean} 등급 표기({found})가 감지되었습니다. "
        f"이 비서는 {max_level.korean} 등급까지만 처리할 수 있으며, 해당 내용은 모델로 전송하지 않았습니다. "
        "비밀 자료는 인가된 보안망과 장비에서만 취급하시고, 표기를 제거한 일반 질문으로 다시 문의해 주십시오."
    )
