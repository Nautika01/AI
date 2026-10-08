import pytest

from defense_assistant.tools import build_tools
from defense_assistant.tools.datetime_tools import convert_dtg, current_dtg, format_dtg, parse_dtg, zone_letter, zone_offset
from defense_assistant.tools.glossary import lookup_term
from defense_assistant.tools.phonetic import phonetic_spell
from defense_assistant.tools.reports import render_report, resolve_kind
from defense_assistant.tools.units import convert_units
from datetime import datetime, timedelta, timezone


def test_zone_letters():
    assert zone_offset("Z") == timedelta(0)
    assert zone_offset("I") == timedelta(hours=9)
    assert zone_offset("K") == timedelta(hours=10)
    assert zone_offset("N") == timedelta(hours=-1)
    assert zone_offset("Y") == timedelta(hours=-12)
    assert zone_letter(9) == "I" and zone_letter(0) == "Z" and zone_letter(-5) == "R"
    with pytest.raises(ValueError):
        zone_offset("J")


def test_parse_and_convert_dtg():
    dt = parse_dtg("071430IOCT26")
    assert dt == datetime(2026, 10, 7, 14, 30, tzinfo=timezone(timedelta(hours=9)))
    assert format_dtg(dt, "Z") == "070530ZOCT26"
    out = convert_dtg("071430IOCT26", "Z")
    assert "070530ZOCT26" in out
    out = convert_dtg("2026-10-07T14:30+09:00", "UTC")
    assert "070530ZOCT26" in out
    # 날짜 경계를 넘는 변환
    assert "010100IJAN27" in convert_dtg("311600ZDEC26", "KST")
    with pytest.raises(ValueError):
        convert_dtg("hello", "Z")
    with pytest.raises(ValueError):
        convert_dtg("2026-10-07T14:30", "Z")


def test_current_dtg_format():
    fixed = datetime(2026, 10, 7, 5, 30, tzinfo=timezone.utc)
    assert current_dtg("I", now=fixed).startswith("071430IOCT26")
    assert current_dtg("Z", now=fixed).startswith("070530ZOCT26")


def test_phonetic():
    assert phonetic_spell("AB-9") == "AB-9 → Alfa Bravo Dash Niner"
    assert "Zulu" in phonetic_spell("z")
    with pytest.raises(ValueError):
        phonetic_spell("  ")


def test_units():
    assert "5.625 deg" in convert_units(100, "mil", "deg")
    assert "37.04 km/h" in convert_units(20, "kn", "km/h")
    assert "1.852 km" in convert_units(1, "nm", "km")
    assert "98.60 F" in convert_units(37, "C", "F")
    assert "2.205 lb" in convert_units(1, "kg", "lb")
    assert "90 도" in convert_units(1600, "밀", "도")
    with pytest.raises(ValueError):
        convert_units(1, "kg", "km")


def test_report_templates():
    assert resolve_kind("메데박") == "MEDEVAC"
    assert resolve_kind("상황보고") == "SITREP"
    assert resolve_kind("스팟렙") == "SPOTREP"
    with pytest.raises(ValueError):
        resolve_kind("없는양식")
    out = render_report("SPOTREP", {"size": "차량 3대", "time": "071430IOCT26", "extra": "x"})
    assert "차량 3대" in out and "미기재 항목" in out and "extra" in out
    out = render_report("MEDEVAC", '{"line1": "[좌표]"}')
    assert out.count("____________") == 8


def test_glossary_lookup(glossary):
    assert "Operation Order" in lookup_term(glossary, "opord")
    assert "METT-TC" in lookup_term(glossary, "mett tc")
    out = lookup_term(glossary, "후송")
    assert "MEDEVAC" in out and "CASEVAC" in out
    assert "찾지 못했습니다" in lookup_term(glossary, "zzzz")
    with pytest.raises(ValueError):
        lookup_term(glossary, " ")


def test_build_tools_schema_and_call(store, glossary):
    tools = build_tools(store, glossary)
    names = [t.name for t in tools]
    assert names == [
        "search_defense_docs", "lookup_military_term", "convert_military_time", "get_current_dtg",
        "spell_phonetic", "generate_report_template", "convert_unit",
    ]
    for t in tools:
        d = t.to_dict()
        assert d["description"] and d["input_schema"]["type"] == "object"
    by_name = {t.name: t for t in tools}
    assert "거수자 조치 절차" in by_name["search_defense_docs"].call({"query": "거수자 조치", "top_k": 1})
    assert "070530ZOCT26" in by_name["convert_military_time"].call({"value": "071430IOCT26", "to_zone": "Z"})
    with pytest.raises(ValueError):
        by_name["convert_unit"].call({"value": "many", "from_unit": "m", "to_unit": "km"})


def test_medevac_codes_follow_standard_9line():
    # 9-Line 표준(ATP 4-02.2): 4번 B=호이스트, 7번 A 패널 / B 발광(pyrotechnic) / C 연막(smoke) / D 없음 / E 기타.
    # 이전 양식은 7번 B·C 가 뒤바뀌어 있었고, 모델이 이 양식을 그대로 답변에 옮겼다(평가 q07).
    out = render_report("MEDEVAC", {})
    assert "A 패널 / B 발광 신호 / C 연막 / D 없음 / E 기타" in out
    assert "B 호이스트" in out and "들것걸이" not in out
