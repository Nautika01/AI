import pytest

from defense_assistant.assistant import AssistantError
from defense_assistant.security import Classification
from tests.conftest import make_message


def test_request_params_shape(make_assistant):
    assistant, client = make_assistant()
    session = assistant.new_session("u1")
    assistant.chat(session, "안녕하세요")
    p = client.calls[0]
    assert p["model"] == "claude-opus-5-5"
    assert p["thinking"] == {"type": "adaptive"}
    assert p["output_config"] == {"effort": "high"}
    assert p["stream"] is True
    assert p["betas"] == ["server-side-fallback-2026-07-01"] and p["fallbacks"] == "default"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "국방 특화" in p["system"][0]["text"]
    assert len(p["tools"]) == 7


def test_fallback_off(make_assistant):
    assistant, client = make_assistant(fallbacks="off", effort="low")
    assistant.chat(assistant.new_session(), "안녕")
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]
    assert client.calls[0]["output_config"] == {"effort": "low"}


def test_streaming_and_history(make_assistant):
    assistant, client = make_assistant([make_message("첫 번째 응답")])
    session = assistant.new_session()
    chunks = []
    r = assistant.chat(session, "질문입니다", on_text=chunks.append)
    assert r.text == "첫 번째 응답" == "".join(chunks)
    assert r.stop_reason == "end_turn" and r.served_by == "claude-opus-5-5" and not r.fallback_used
    assert r.usage == {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50, "cache_creation_input_tokens": None}
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert session.turns == 1
    # 두 번째 턴은 전체 기록을 보낸다
    assistant.chat(session, "두 번째")
    assert len(client.calls[1]["messages"]) == 3


def test_classification_gate_blocks_before_model_call(make_assistant):
    assistant, client = make_assistant()
    session = assistant.new_session("u1")
    r = assistant.chat(session, "이 II급 비밀 문서를 요약해 줘")
    assert r.blocked and r.classification.level == Classification.SECRET
    assert "II급 비밀" in r.text
    assert client.calls == [] and session.messages == []
    rec = assistant.audit.records[-1]
    assert rec.event == "blocked" and rec.classification == "SECRET"


def test_restricted_is_allowed_by_default(make_assistant):
    assistant, client = make_assistant()
    r = assistant.chat(assistant.new_session(), "대외비 양식 설명")
    assert not r.blocked and len(client.calls) == 1


def test_higher_clearance_setting(make_assistant):
    assistant, client = make_assistant(max_classification=Classification.SECRET)
    r = assistant.chat(assistant.new_session(), "II급 비밀 양식")
    assert not r.blocked
    r = assistant.chat(assistant.new_session(), "I급 비밀 양식")
    assert r.blocked


def test_redaction_applied_before_sending(make_assistant):
    assistant, client = make_assistant()
    session = assistant.new_session()
    r = assistant.chat(session, "군번 19-70012345 전화 010-1234-5678 확인 요청")
    sent = client.calls[0]["messages"][0]["content"]
    assert "19-70012345" not in sent and "[군번]" in sent and "[휴대전화]" in sent
    assert r.redaction.counts == {"휴대전화": 1, "군번": 1}
    rec = assistant.audit.records[-1]
    assert rec.redactions == {"휴대전화": 1, "군번": 1}
    assert rec.input_sha256 != "" and rec.output_chars == len(r.text)


def test_tool_calls_are_executed_and_recorded(make_assistant):
    scripted = [
        make_message("", stop_reason="tool_use", tool_uses=[("convert_military_time", {"value": "071430IOCT26", "to_zone": "Z"})]),
        make_message("Z 시간은 070530ZOCT26 입니다."),
    ]
    assistant, client = make_assistant(scripted)
    session = assistant.new_session()
    seen = []
    r = assistant.chat(session, "071430IOCT26 를 줄루로", on_tool=lambda n, i: seen.append((n, i)))
    assert r.tools_called == ["convert_military_time"]
    assert seen == [("convert_military_time", {"value": "071430IOCT26", "to_zone": "Z"})]
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result = session.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and "070530ZOCT26" in tool_result["content"]
    assert assistant.audit.records[-1].tools_called == ["convert_military_time"]


def test_refusal_and_max_tokens_and_fallback(make_assistant):
    assistant, _ = make_assistant([make_message("", stop_reason="refusal")])
    r = assistant.chat(assistant.new_session(), "질문")
    assert "안전 정책" in r.text and "cyber" in r.text

    assistant, _ = make_assistant([make_message("긴 답변", stop_reason="max_tokens")])
    r = assistant.chat(assistant.new_session(), "질문")
    assert r.text.startswith("긴 답변") and "잘렸습니다" in r.text

    assistant, _ = make_assistant([make_message("폴백 응답", model="claude-opus-4-8", fallback=True)])
    r = assistant.chat(assistant.new_session(), "질문")
    assert r.fallback_used and r.served_by == "claude-opus-4-8"


def test_api_error_is_wrapped_and_turn_rolled_back(make_assistant):
    import anthropic
    import httpx2 as httpx

    assistant, client = make_assistant()
    resp = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))

    def boom(**params):
        raise anthropic.AuthenticationError("bad key", response=resp, body=None)

    client.beta.messages.tool_runner = boom
    session = assistant.new_session()
    with pytest.raises(AssistantError, match="인증"):
        assistant.chat(session, "안녕")
    assert session.messages == []
    assert assistant.audit.records[-1].event == "error"


def test_empty_input_rejected(make_assistant):
    assistant, _ = make_assistant()
    with pytest.raises(ValueError):
        assistant.chat(assistant.new_session(), "   ")


def test_result_to_dict(make_assistant):
    assistant, _ = make_assistant()
    d = assistant.chat(assistant.new_session(), "안녕").to_dict()
    assert set(d) >= {"text", "blocked", "classification", "redactions", "tools_called", "stop_reason", "usage"}
