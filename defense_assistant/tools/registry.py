"""Claude 도구 레지스트리.

각 도구는 순수 함수(테스트 가능)를 `beta_tool`로 감싼 것이다. 지식 베이스·용어 사전처럼
상태가 필요한 도구는 클로저로 묶는다. 모든 도구는 로컬에서 실행되며 외부 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any

from anthropic import beta_tool
from anthropic.lib.tools import BetaFunctionTool

from ..knowledge import DocumentStore
from .datetime_tools import convert_dtg, current_dtg
from .glossary import lookup_term
from .phonetic import phonetic_spell
from .reports import render_report
from .units import convert_units


def build_tools(store: DocumentStore, glossary: dict[str, dict[str, str]]) -> list[BetaFunctionTool[Any]]:
    def search_defense_docs(query: str, top_k: int = 4, category: str | None = None) -> str:
        """부대 지식 베이스(규정·지침·교범 요약 문서)를 검색한다. 절차·규정·양식·용어에 대한 질문이면 답하기 전에 먼저 호출하라.

        Args:
            query: 검색어. 한국어 또는 영문 약어. 핵심 명사 위주로 짧게 작성.
            top_k: 반환할 문서 청크 수 (1~8).
            category: 문서 분류로 범위를 좁힐 때만 지정 (예: 규정, 교범, 지침). 보통은 비워 둔다.
        """
        top_k = max(1, min(int(top_k), 8))
        hits = store.search(query, top_k=top_k, category=category or None)
        return store.format_hits(hits)

    def lookup_military_term(term: str) -> str:
        """군사 약어·용어 사전을 조회한다 (예: OPORD, METT-TC, 작전통제).

        Args:
            term: 조회할 약어 또는 용어.
        """
        return lookup_term(glossary, term)

    def convert_military_time(value: str, to_zone: str = "Z") -> str:
        """일시군(DTG) 또는 ISO 8601 시각을 다른 군 시간대로 변환한다. 한국 표준시는 I(India), 협정세계시는 Z(Zulu).

        Args:
            value: 변환할 시각. 예: "071430IOCT26" 또는 "2026-10-07T14:30+09:00".
            to_zone: 목표 시간대 문자 (Z, I 등) 또는 "KST"/"UTC".
        """
        return convert_dtg(value, to_zone)

    def get_current_dtg(zone: str = "I") -> str:
        """현재 시각을 군 일시군(DTG) 형식으로 돌려준다.

        Args:
            zone: 시간대 문자 (기본 I = 한국 표준시, Z = 협정세계시).
        """
        return current_dtg(zone)

    def spell_phonetic(text: str) -> str:
        """영문자·숫자를 NATO 음성 문자(Alfa, Bravo, …)로 풀어 쓴다. 호출부호·좌표·차량번호 송신 시 사용.

        Args:
            text: 변환할 문자열.
        """
        return phonetic_spell(text)

    def generate_report_template(kind: str, fields: dict[str, Any] | str | None = None) -> str:
        """표준 보고서 양식을 만든다. 종류: SITREP, SPOTREP(SALUTE), MEDEVAC(9-line), WARNORD, OPORD. 사용자가 준 정보는 fields로 채운다.

        Args:
            kind: 보고서 종류 (SITREP / SPOTREP / MEDEVAC / WARNORD / OPORD, 한글 명칭도 가능).
            fields: 채워 넣을 항목(객체 또는 JSON 문자열, 값은 문자열·숫자 모두 가능). 키는 양식 필드명(SITREP: unit, dtg, enemy, friendly, personnel, equipment, supply, assessment; SPOTREP: size, activity, location, unit, time, equipment, observer; MEDEVAC: line1~line9; WARNORD/OPORD: situation, mission, execution, service_support, command_signal).
        """
        return render_report(kind, fields)

    def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
        """군사 단위를 변환한다. 길이(m/km/mi/nm/yd/ft), 속도(km/h/mph/kn/m/s), 무게(kg/lb), 부피(l/gal), 각도(deg/mil), 압력(hPa/psi), 온도(C/F).

        Args:
            value: 변환할 값.
            from_unit: 원래 단위.
            to_unit: 목표 단위.
        """
        return convert_units(value, from_unit, to_unit)

    return [
        beta_tool(search_defense_docs),
        beta_tool(lookup_military_term),
        beta_tool(convert_military_time),
        beta_tool(get_current_dtg),
        beta_tool(spell_phonetic),
        beta_tool(generate_report_template),
        beta_tool(convert_unit),
    ]
