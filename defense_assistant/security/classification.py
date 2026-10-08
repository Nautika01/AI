"""비밀 등급 표기 탐지 및 처리 허용 여부 판단.

한국군 비밀 등급 체계(I급·II급·III급 비밀, 대외비, 일반)와 영문 표기
(TOP SECRET / SECRET / CONFIDENTIAL / RESTRICTED)를 함께 인식한다.

이 모듈은 *표기(marking)* 를 탐지할 뿐 내용 자체의 비밀성을 판단하지 않는다.
비밀 자료는 인가된 망·장비 밖으로 반출되어서는 안 되므로, 허용 등급을 넘는
표기가 감지되면 모델 호출 전에 차단하는 것이 목적이다.
"""

from __future__ import annotations

import re
import unicodedata
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
        # 'Ⅱ급'(U+2161)·전각 'ＳＥＣＲＥＴ' 같은 호환 문자도 같은 별칭으로 받는다.
        key = normalize_for_matching(value).strip().upper().replace(" ", "_").replace("-", "_")
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

# NFKC 가 바꾸지 않는 각종 대시·마이너스 기호(‐ – — − 등)를 '-' 로 맞춘다.
_DASHES = str.maketrans({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})


def normalize_for_matching(text: str) -> str:
    """표기 탐지용 정규화 사본을 만든다(원문은 바꾸지 않는다).

    - NFKC: 유니코드 로마숫자 'Ⅱ'(U+2161) → 'II', 전각 'ＳＥＣＲＥＴ' → 'SECRET', '：' → ':'
    - 서식 문자(Cf: U+200B 제로폭 공백, U+00AD 소프트하이픈, U+FEFF 등) 제거
    - 각종 대시(–, — 등)를 '-' 로 통일
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return text.translate(_DASHES)


# 등급 숫자 토큰: I/II/III(로마숫자, NFKC 후 ASCII) 또는 1/2/3. 'IV', '12' 등은 제외한다.
_GRADE = {
    Classification.TOP_SECRET: r"(?:(?<![IVX\d])I(?![IVX])|(?<!\d)1(?!\d))",
    Classification.SECRET: r"(?:(?<![IVX\d])II(?![IVX])|(?<!\d)2(?!\d))",
    Classification.CONFIDENTIAL: r"(?:(?<![IVX\d])III(?![IVX])|(?<!\d)3(?!\d))",
}


def _grade_patterns(level: Classification) -> list[tuple[Classification, re.Pattern[str]]]:
    g = _GRADE[level]
    return [
        # II급 비밀 / 2급비밀 / II급 기밀
        (level, re.compile(rf"{g}\s*급\s*(?:비밀|기밀)")),
        # 표지·결재란 형식: '비밀등급: II급' (콜론 필수 — '비밀등급 체계' 같은 질문은 제외)
        (level, re.compile(rf"비밀\s*등급\s*:\s*{g}\s*급")),
        # 괄호 형식: '비밀(II급)'
        (level, re.compile(rf"비밀\s*\(\s*{g}\s*급\s*\)")),
    ]


# TOP 과 SECRET 사이 구분자: 공백·밑줄·하이픈(정규화 후 대시는 모두 '-')
_TOP_SEP = r"[\s_-]"

# 각 패턴은 명시적 표기만 잡도록 보수적으로 작성한다. 영문 표기는 관례대로 대문자만 인식한다.
# ("비밀번호", "비밀스러운", "비밀취급인가", "극비리에" 같은 일상어·업무 용어에 반응하면 안 된다.)
# 패턴은 normalize_for_matching() 을 거친 텍스트에 적용한다.
_PATTERNS: list[tuple[Classification, re.Pattern[str]]] = [
    # I급 비밀 / 1급비밀 / 1급 기밀 / 비밀등급: I급 / 극비 / TOP SECRET / TS//
    *_grade_patterns(Classification.TOP_SECRET),
    (Classification.TOP_SECRET, re.compile(r"극비(?!리)")),
    (Classification.TOP_SECRET, re.compile(rf"(?<![A-Za-z\d])TOP{_TOP_SEP}*SECRET(?![A-Za-z\d])")),
    (Classification.TOP_SECRET, re.compile(r"(?:^|[\s(\[/])TS(?:\s*//|\s*/\s*[A-Z]|\s*[\])])")),
    # II급 비밀 / 2급비밀 / 비밀등급: II급 / SECRET
    *_grade_patterns(Classification.SECRET),
    (Classification.SECRET, re.compile(rf"(?<!TOP)(?<!TOP{_TOP_SEP})\bSECRET\b(?!\s*번호)")),
    # III급 비밀 / 3급비밀 / 비밀등급: III급 / CONFIDENTIAL / 군사기밀 / 비밀문서
    *_grade_patterns(Classification.CONFIDENTIAL),
    (Classification.CONFIDENTIAL, re.compile(r"\bCONFIDENTIAL\b")),
    (Classification.CONFIDENTIAL, re.compile(r"군사\s*(?:비밀|기밀)")),
    # '비밀취급인가'(인사 용어)는 표기가 아니므로 제외한다.
    (Classification.CONFIDENTIAL, re.compile(r"비밀\s*(?:문서|자료|취급(?!\s*인가)|등급\s*:\s*(?:비밀|기밀))")),
    # 대외비 / RESTRICTED / FOUO
    (Classification.RESTRICTED, re.compile(r"대외비")),
    (Classification.RESTRICTED, re.compile(r"\bRESTRICTED\b")),
    (Classification.RESTRICTED, re.compile(r"\bFOUO\b")),
    (Classification.RESTRICTED, re.compile(r"\bFOR\s+OFFICIAL\s+USE\s+ONLY\b")),
]


# 등급 용어 자체를 묻는 교육용 질문('II급 비밀이 뭐야', 'I급 비밀과 II급 비밀 차이')은 표기가 아니다.
# 입력 전체가 [등급 용어 + 정의·차이를 묻는 말] 로만 이루어진 경우에만 예외로 둔다(fullmatch).
# 다른 내용이 한 글자라도 붙으면('II급 비밀 문서 요약해 줘', 'II급 비밀: 작전 계획…') 그대로 차단한다.
# '알려줘'는 '비밀(내용)을 알려줘' 로도 읽히므로 술어로 받지 않는다.
_ANY_GRADE = r"(?:III|II|I|[123])"
_GRADE_TERM = rf"{_ANY_GRADE}\s*급\s*(?:비밀|기밀)"
_CONCEPT_QUESTION = re.compile(
    rf"""
    {_GRADE_TERM}(?:\s*(?:과|와|및|,|하고|이랑|랑)\s*{_GRADE_TERM})*
    \s*(?:이란|란|이|은|는|의|과|와)?
    (?:
        \s*(?:정의|뜻|의미|개념|기준|차이점|차이|구분)
        \s*(?:이|은|는|가|을|를)?
        (?:\s*(?:뭐야|뭐예요|뭐에요|뭔가요|뭐지|뭐냐|뭡니까|무엇인가요|무엇입니까|무엇이야|무엇|설명해\s*(?:줘|주세요|주십시오)))?
      | \s*(?:뭐야|뭐예요|뭐에요|뭔가요|뭐지|뭐냐|뭡니까|무엇인가요|무엇입니까|무엇이야|무엇|설명해\s*(?:줘|주세요|주십시오))
      | (?<=란)\s*(?=[?])
    )
    \s*[?.!]*
    """,
    re.VERBOSE,
)


def is_concept_question(text: str) -> bool:
    """입력 전체가 비밀 등급 용어의 뜻·차이를 묻는 질문뿐인지(표기가 아님) 판단한다."""
    return _CONCEPT_QUESTION.fullmatch(normalize_for_matching(text).strip()) is not None


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
    """텍스트에서 가장 높은 비밀 등급 표기를 찾는다. 표기가 없으면 UNCLASSIFIED.

    매칭은 정규화 사본(normalize_for_matching)에 대해 수행하므로 markings 에는
    정규화된 표기('Ⅱ급비밀' → 'II급비밀')가 담긴다.
    """
    if is_concept_question(text):
        return ClassificationResult(level=Classification.UNCLASSIFIED)
    text = normalize_for_matching(text)
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
