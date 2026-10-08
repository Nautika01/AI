"""표준 군 보고서 양식 생성 (SITREP / SPOTREP(SALUTE) / 9-Line MEDEVAC / WARNORD / OPORD)."""

from __future__ import annotations

import json
from typing import Any

TEMPLATES: dict[str, dict[str, Any]] = {
    "SITREP": {
        "title": "상황보고 (SITREP)",
        "fields": [
            ("unit", "보고 부대"),
            ("dtg", "일시군(DTG)"),
            ("enemy", "적 상황"),
            ("friendly", "아군 상황"),
            ("personnel", "인원 현황"),
            ("equipment", "장비 현황"),
            ("supply", "보급 현황 (탄약/유류/식량/식수)"),
            ("assessment", "지휘관 평가 및 건의"),
        ],
    },
    "SPOTREP": {
        "title": "첩보보고 (SPOTREP, SALUTE 형식)",
        "fields": [
            ("size", "S – 규모 (Size)"),
            ("activity", "A – 활동 (Activity)"),
            ("location", "L – 위치 (Location)"),
            ("unit", "U – 부대/복장 (Unit/Uniform)"),
            ("time", "T – 시각 (Time, DTG)"),
            ("equipment", "E – 장비 (Equipment)"),
            ("observer", "관측자 / 보고 수단"),
        ],
    },
    "MEDEVAC": {
        "title": "환자 후송 요청 (9-Line MEDEVAC)",
        "fields": [
            ("line1", "1. 착륙 지점 위치(좌표)"),
            ("line2", "2. 무전 주파수·호출부호"),
            ("line3", "3. 환자 수 및 긴급도 (A 긴급 / B 긴급-외과 / C 우선 / D 일상 / E 편의)"),
            ("line4", "4. 특수 장비 (A 없음 / B 호이스트(인양기) / C 구조장비 / D 인공호흡기)"),
            ("line5", "5. 환자 유형 (L 들것 / A 보행)"),
            ("line6", "6. 착륙 지점 안전 (N 적 없음 / P 적 가능 / E 적 활동 중 / X 호위 필요)"),
            ("line7", "7. 착륙 지점 표시 (A 패널 / B 발광 신호 / C 연막 / D 없음 / E 기타)"),
            ("line8", "8. 환자 국적·신분 (A 아군 군인 / B 아군 민간인 / C 비아군 군인 / D 비아군 민간인 / E 포로)"),
            ("line9", "9. 지형 특징 / NBC 오염 여부"),
        ],
    },
    "WARNORD": {
        "title": "준비명령 (WARNORD)",
        "fields": [
            ("situation", "1. 상황 (적/아군 개요)"),
            ("mission", "2. 임무"),
            ("execution", "3. 실시 개요 (예상 과업, 준비 지시, 예행연습·검사 시간)"),
            ("service_support", "4. 전투근무지원 (보급 요청, 장비 준비)"),
            ("command_signal", "5. 지휘 및 통신 (명령 하달 시간·장소, 통신 수단)"),
        ],
    },
    "OPORD": {
        "title": "작전명령 (OPORD, 5단락)",
        "fields": [
            ("situation", "1. 상황 (적 부대, 아군 부대, 배속 및 파견, 민간 고려사항)"),
            ("mission", "2. 임무 (누가·무엇을·언제·어디서·왜)"),
            ("execution", "3. 실시 (지휘관 의도, 작전 개념, 예하 부대 과업, 협조 지시)"),
            ("service_support", "4. 전투근무지원 (보급, 정비, 의무, 수송)"),
            ("command_signal", "5. 지휘 및 통신 (지휘소 위치, 지휘 승계, 통신 운용)"),
        ],
    },
}

_ALIASES = {
    "상황보고": "SITREP", "상황 보고": "SITREP", "sitrep": "SITREP", "시트렙": "SITREP", "싯렙": "SITREP",
    "첩보보고": "SPOTREP", "첩보 보고": "SPOTREP", "spotrep": "SPOTREP", "salute": "SPOTREP", "스팟렙": "SPOTREP", "스폿렙": "SPOTREP",
    "메데박": "MEDEVAC", "medevac": "MEDEVAC", "9line": "MEDEVAC", "9-line": "MEDEVAC", "후송요청": "MEDEVAC", "환자후송": "MEDEVAC",
    "준비명령": "WARNORD", "warnord": "WARNORD", "warno": "WARNORD",
    "작전명령": "OPORD", "opord": "OPORD", "작명": "OPORD",
}


def resolve_kind(kind: str) -> str:
    key = kind.strip()
    if key.upper() in TEMPLATES:
        return key.upper()
    norm = key.lower().replace(" ", "")
    for alias, target in _ALIASES.items():
        if alias.replace(" ", "") == norm:
            return target
    raise ValueError(f"지원하지 않는 보고서 종류: {kind!r}. 가능: {', '.join(TEMPLATES)}")


def _field_text(val: Any) -> str | None:
    """필드 값을 양식에 넣을 문자열로 정규화한다 (숫자·목록 허용)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return "예" if val else "아니오"
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val)
    if isinstance(val, dict):
        return ", ".join(f"{k}: {v}" for k, v in val.items())
    return str(val)


def render_report(kind: str, fields: dict[str, Any] | str | None = None) -> str:
    """양식을 렌더링한다. `fields`가 주어지면 해당 항목을 채우고, 없는 항목은 빈칸으로 둔다."""
    k = resolve_kind(kind)
    if isinstance(fields, str):
        try:
            fields = json.loads(fields) if fields.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"fields 는 JSON 객체 형식이어야 합니다 (예: {{\"unit\": \"1대대\"}}): {e.msg}") from e
    fields = fields or {}
    if not isinstance(fields, dict):
        raise ValueError(f"fields 는 항목명→값 객체여야 합니다 (받은 형식: {type(fields).__name__})")
    fields = {str(k): _field_text(v) for k, v in fields.items()}
    tpl = TEMPLATES[k]
    lines = [f"■ {tpl['title']}", ""]
    missing: list[str] = []
    for key, label in tpl["fields"]:
        val = fields.get(key)
        if val in (None, ""):
            missing.append(label)
            val = "____________"
        lines.append(f"{label}: {val}")
    unknown = sorted(set(fields) - {k for k, _ in tpl["fields"]})
    if missing:
        lines += ["", "※ 미기재 항목: " + ", ".join(missing)]
    if unknown:
        lines += ["※ 양식에 없는 항목(무시됨): " + ", ".join(unknown)]
    lines += ["", "※ 작성 시 유의: 비밀·대외비 내용 및 실제 좌표·주파수는 인가된 체계에서만 기재합니다."]
    return "\n".join(lines)
