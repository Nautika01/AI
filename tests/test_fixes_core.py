"""교차 검토에서 확인된 결함(코어·Claude 백엔드·도구·설정)의 회귀 테스트."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from defense_assistant.assistant import AssistantError
from defense_assistant.security import AuditLogger
from tests.conftest import make_message


def _with_usage(msg, inp, out, cache):
    msg.usage = SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=cache, iterations=None)
    return msg


def _assert_no_empty_content(messages):
    for m in messages:
        c = m["content"]
        assert c not in ([], "", None), m


# --- 거부 턴의 빈 assistant 메시지 -------------------------------------------------

def test_refusal_turn_does_not_store_empty_assistant_message(make_assistant):
    assistant, client = make_assistant([make_message("", stop_reason="refusal")])
    session = assistant.new_session()
    r = assistant.chat(session, "질문 1")
    assert "안전 정책" in r.text
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    _assert_no_empty_content(session.messages)
    # 같은 세션의 다음 턴 요청에도 빈 content 가 들어가지 않는다
    client.scripted = [make_message("두 번째 답")]
    assistant.chat(session, "질문 2")
    _assert_no_empty_content(client.calls[1]["messages"])


def test_refusal_with_partial_output_stores_notice_not_partial_text(make_assistant):
    assistant, _ = make_assistant([make_message("부분 출력 일부", stop_reason="refusal")])
    session = assistant.new_session()
    assistant.chat(session, "질문")
    stored = session.messages[-1]["content"]
    assert all("부분 출력" not in b.get("text", "") for b in stored)
    assert any("안전 정책" in b.get("text", "") for b in stored)


def test_previously_stored_empty_assistant_message_is_patched_before_sending(make_assistant):
    assistant, client = make_assistant([make_message("답")])
    session = assistant.new_session()
    session.messages = [{"role": "user", "content": "이전 질문"}, {"role": "assistant", "content": []}]
    assistant.chat(session, "새 질문")
    _assert_no_empty_content(client.calls[0]["messages"])
    # 원본 기록은 바꾸지 않는다
    assert session.messages[1]["content"] == []


# --- max_tokens·refusal 로 끝난 턴의 tool_use ---------------------------------------

def test_tool_use_in_max_tokens_turn_is_not_executed(make_assistant):
    scripted = [make_message("", stop_reason="max_tokens", tool_uses=[("convert_unit", {"value": 1, "from_unit": "km", "to_unit": "m"})])]
    assistant, _ = make_assistant(scripted)
    session = assistant.new_session()
    seen = []
    r = assistant.chat(session, "변환", on_tool=lambda n, i: seen.append(n))
    assert seen == [] and r.tools_called == []
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    stored = session.messages[-1]["content"]
    assert all(b.get("type") != "tool_use" for b in stored)
    _assert_no_empty_content(session.messages)
    assert assistant.audit.records[-1].tools_called == []


def test_tool_use_turn_still_executes(make_assistant):
    scripted = [
        make_message("", stop_reason="tool_use", tool_uses=[("convert_unit", {"value": 1, "from_unit": "km", "to_unit": "m"})]),
        make_message("1,000 m 입니다."),
    ]
    assistant, _ = make_assistant(scripted)
    session = assistant.new_session()
    r = assistant.chat(session, "변환")
    assert r.tools_called == ["convert_unit"]
    assert [m["role"] for m in session.messages] == ["user", "assistant", "user", "assistant"]


# --- 토큰 사용량 합산 -------------------------------------------------------------

def test_usage_is_summed_over_tool_runner_iterations(make_assistant):
    scripted = [
        _with_usage(make_message("", stop_reason="tool_use", tool_uses=[("spell_phonetic", {"text": "AB"})]), 6000, 400, 1000),
        _with_usage(make_message("답"), 8000, 600, 2000),
    ]
    assistant, _ = make_assistant(scripted)
    r = assistant.chat(assistant.new_session(), "질문")
    assert r.usage == {"input_tokens": 14000, "output_tokens": 1000, "cache_read_input_tokens": 3000, "cache_creation_input_tokens": None}
    rec = assistant.audit.records[-1]
    assert (rec.input_tokens, rec.output_tokens, rec.cache_read_tokens) == (14000, 1000, 3000)


def test_cache_creation_tokens_are_summed_and_audited(make_assistant):
    # 캐시 쓰기 토큰은 input_tokens 에 들어가지 않으므로 따로 합산해야 실제 입력량·비용이 보인다.
    first = _with_usage(make_message("", stop_reason="tool_use", tool_uses=[("spell_phonetic", {"text": "AB"})]), 300, 50, 0)
    first.usage.cache_creation_input_tokens = 3500
    second = _with_usage(make_message("답"), 200, 80, 3500)
    second.usage.cache_creation_input_tokens = 0
    assistant, _ = make_assistant([first, second])
    r = assistant.chat(assistant.new_session(), "질문")
    assert r.usage["cache_creation_input_tokens"] == 3500 and r.usage["cache_read_input_tokens"] == 3500
    assert assistant.audit.records[-1].cache_creation_tokens == 3500


def test_usage_none_stays_none(make_assistant):
    msg = make_message("답")
    msg.usage = SimpleNamespace(input_tokens=None, output_tokens=None, cache_read_input_tokens=None, iterations=None)
    assistant, _ = make_assistant([msg])
    r = assistant.chat(assistant.new_session(), "질문")
    assert r.usage == {"input_tokens": None, "output_tokens": None, "cache_read_input_tokens": None, "cache_creation_input_tokens": None}


# --- 중단(Ctrl-C) 시 user 메시지 롤백 ------------------------------------------------

def test_interrupted_turn_is_rolled_back(make_assistant):
    assistant, client = make_assistant()

    def interrupt(**params):
        raise KeyboardInterrupt

    client.beta.messages.tool_runner = interrupt
    session = assistant.new_session()
    with pytest.raises(KeyboardInterrupt):
        assistant.chat(session, "중단될 질문")
    assert session.messages == [] and session.turns == 0


# --- 감사 로그 실패 ---------------------------------------------------------------

class _FailingAudit(AuditLogger):
    def __init__(self):
        super().__init__(None)

    def write(self, record):
        raise OSError(28, "No space left on device")


def test_audit_write_failure_rolls_back_turn_and_raises_korean_error(make_assistant):
    assistant, client = make_assistant()
    assistant.audit = _FailingAudit()
    session = assistant.new_session()
    with pytest.raises(AssistantError, match="감사 로그"):
        assistant.chat(session, "질문")
    assert session.messages == [] and session.turns == 0


def test_audit_write_failure_on_blocked_path_is_korean_error(make_assistant):
    assistant, client = make_assistant()
    assistant.audit = _FailingAudit()
    with pytest.raises(AssistantError, match="감사 로그"):
        assistant.chat(assistant.new_session(), "이 II급 비밀 문서를 요약해 줘")
    assert client.calls == []


def test_unwritable_audit_log_blocks_model_call(make_assistant, tmp_path):
    import shutil

    assistant, client = make_assistant()
    audit_dir = tmp_path / "audit"
    assistant.audit = AuditLogger(audit_dir / "audit.jsonl")
    shutil.rmtree(audit_dir)
    session = assistant.new_session()
    with pytest.raises(AssistantError, match="감사 로그"):
        assistant.chat(session, "질문")
    assert client.calls == [] and session.messages == []


# --- 컨텍스트 창 초과 ---------------------------------------------------------------

def test_model_context_window_exceeded_gets_notice(make_assistant):
    assistant, _ = make_assistant([make_message("절차: 1. 준비 2. 실", stop_reason="model_context_window_exceeded")])
    r = assistant.chat(assistant.new_session(), "보고서")
    assert r.text.startswith("절차: 1. 준비 2. 실") and "새 대화" in r.text


def test_prompt_too_long_error_is_explained_in_korean():
    import anthropic
    import httpx2 as httpx

    from defense_assistant.backends.claude import korean_error

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    e = anthropic.BadRequestError("prompt is too long: 1000321 tokens > 1000000 maximum", response=httpx.Response(400, request=req), body=None)
    assert "새 대화" in korean_error(e)
    e413 = anthropic.APIStatusError("request_too_large", response=httpx.Response(413, request=req), body=None)
    assert "새 대화" in korean_error(e413)
    other = anthropic.BadRequestError("messages: roles must alternate", response=httpx.Response(400, request=req), body=None)
    assert korean_error(other).startswith("잘못된 요청입니다")


# --- 도구 -----------------------------------------------------------------------

@pytest.mark.parametrize("zone", ["local", "Alpha", "B1", "현지"])
def test_multi_letter_zone_is_korean_value_error(zone):
    from defense_assistant.tools.datetime_tools import convert_dtg, current_dtg

    with pytest.raises(ValueError, match="시간대"):
        current_dtg(zone)
    with pytest.raises(ValueError, match="시간대"):
        convert_dtg("071430IOCT26", zone)


def test_unit_conversion_has_no_exponent_notation():
    from defense_assistant.tools.units import convert_units

    assert convert_units(10, "km", "m") == "10 km = 10,000 m (길이)"
    assert "100,000 m" in convert_units(100, "km", "m")
    assert "37,040 m" in convert_units(20, "nm", "m")
    assert convert_units(1000000, "m", "km") == "1,000,000 m = 1,000 km (길이)"
    assert "1 mm = 0.000001 km" in convert_units(1, "mm", "km")
    for out in (convert_units(10, "km", "m"), convert_units(1e-9, "km", "m"), convert_units(123456789, "m", "km")):
        assert "e+" not in out and "e-" not in out


def test_report_tool_accepts_numeric_and_json_string_fields(store, glossary):
    from defense_assistant.tools import build_tools

    tool = {t.name: t for t in build_tools(store, glossary)}["generate_report_template"]
    out = tool.call({"kind": "SITREP", "fields": {"personnel": 120, "unit": "1대대"}})
    assert "120" in out and "1대대" in out
    out = tool.call({"kind": "SITREP", "fields": '{"unit": "1대대"}'})
    assert "1대대" in out


def test_report_invalid_json_fields_is_korean_error():
    from defense_assistant.tools.reports import render_report

    with pytest.raises(ValueError, match="JSON"):
        render_report("SITREP", "{unit: 1대대")
    with pytest.raises(ValueError, match="객체"):
        render_report("SITREP", "[1, 2]")


# --- 설정 -----------------------------------------------------------------------

def test_config_errors_name_the_variable(monkeypatch):
    from defense_assistant.config import ConfigError, Settings

    monkeypatch.setenv("DAI_MAX_TOKENS", "16k")
    with pytest.raises(ConfigError, match="DAI_MAX_TOKENS"):
        Settings.from_env()
    monkeypatch.delenv("DAI_MAX_TOKENS")
    monkeypatch.setenv("DAI_LOCAL_TIMEOUT", "abc")
    with pytest.raises(ConfigError, match="DAI_LOCAL_TIMEOUT"):
        Settings.from_env()
    monkeypatch.delenv("DAI_LOCAL_TIMEOUT")
    monkeypatch.setenv("DAI_MAX_CLASSIFICATION", "아무거나")
    with pytest.raises(ConfigError, match="DAI_MAX_CLASSIFICATION"):
        Settings.from_env()


def test_config_enum_values_are_normalized(monkeypatch):
    from defense_assistant.config import Settings

    monkeypatch.setenv("DAI_EFFORT", " High ")
    monkeypatch.setenv("DAI_FALLBACKS", "OFF")
    monkeypatch.setenv("DAI_REDACTION", "Mask")
    s = Settings.from_env()
    assert (s.effort, s.fallbacks, s.redaction) == ("high", "off", "mask")


def test_config_validation_message_is_korean():
    from defense_assistant.config import ConfigError, Settings

    with pytest.raises(ConfigError, match="DAI_EFFORT 값은"):
        Settings(effort="extreme")


# --- 용어 사전 --------------------------------------------------------------------

def test_glossary_tolerates_bom_empty_and_broken_files(tmp_path):
    from defense_assistant.tools.glossary import load_glossary

    data = {"OPORD": {"full": "Operation Order", "ko": "작전명령", "desc": "작전 명령서"}}
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert load_glossary(bom) == data
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert load_glossary(empty) == {}
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data, ensure_ascii=False)[:20], encoding="utf-8")
    assert load_glossary(broken) == {}
    top_list = tmp_path / "list.json"
    top_list.write_text("[1, 2]", encoding="utf-8")
    assert load_glossary(top_list) == {}


def test_glossary_entry_missing_field_does_not_break_lookup(tmp_path):
    from defense_assistant.tools.glossary import load_glossary, lookup_term

    p = tmp_path / "g.json"
    p.write_text(json.dumps({
        "XYZ": {"full": "Xray Yankee Zulu", "desc": "설명"},
        "BAD": "문자열 항목",
        "OPORD": {"full": "Operation Order", "ko": "작전명령", "desc": "작전 명령서"},
    }, ensure_ascii=False), encoding="utf-8")
    g = load_glossary(p)
    assert set(g) == {"XYZ", "OPORD"}
    assert "Xray" in lookup_term(g, "XYZ")
    assert "OPORD" in lookup_term(g, "작전")
    # load_glossary 를 거치지 않은 사전에서도 부분 일치 조회가 KeyError 로 실패하지 않는다
    assert "XYZ" in lookup_term({"XYZ": {"full": "Xray", "desc": "d"}}, "xr")
