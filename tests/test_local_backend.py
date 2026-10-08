import json

import pytest

fastapi = pytest.importorskip("fastapi")

from defense_assistant.assistant import AssistantError, DefenseAssistant  # noqa: E402
from defense_assistant.backends import BackendError  # noqa: E402
from defense_assistant.backends.local import LocalBackend  # noqa: E402
from defense_assistant.config import Settings  # noqa: E402
from defense_assistant.prompts import build_local_system_prompt  # noqa: E402
from defense_assistant.security import AuditLogger  # noqa: E402
from defense_assistant.tools import build_tools  # noqa: E402
from tests.conftest import ROOT  # noqa: E402
from tests.fake_openai_server import create_fake_server, run_in_thread  # noqa: E402


@pytest.fixture
def make_local(store, glossary):
    """가짜 OpenAI 호환 서버를 실제 포트에 띄우고, 그 서버를 보는 LocalBackend 와 비서를 만든다."""
    stops = []

    def _make(*, reject_tools=False, store_override=None, **overrides):
        app = create_fake_server(reject_tools=reject_tools)
        base_url, stop = run_in_thread(app)
        stops.append(stop)
        kwargs = dict(backend="local", local_model="fake-model", local_base_url=base_url, local_timeout=20, docs_dir=ROOT / "data/docs", glossary_path=ROOT / "data/glossary.json")
        kwargs.update(overrides)
        settings = Settings(**kwargs)
        s = store_override if store_override is not None else store
        tools = build_tools(s, glossary)
        backend = LocalBackend(settings, tools, s, build_local_system_prompt(s.titles))
        assistant = DefenseAssistant(settings, store=s, glossary=glossary, audit=AuditLogger(None), backend=backend)
        return assistant, backend, app

    yield _make
    for stop in stops:
        stop()


def test_check_lists_models(make_local):
    _, backend, _ = make_local()
    info = backend.check()
    assert info["model_available"] and "fake-model" in info["models"]


def test_rag_injection_and_streaming(make_local):
    assistant, backend, app = make_local()
    session = assistant.new_session("kim")
    chunks = []
    r = assistant.chat(session, "거수자 조치 절차 알려줘", on_text=chunks.append)
    assert r.text.endswith("[1]") and "".join(chunks) == r.text
    assert r.tools_called == ["search_defense_docs"]
    assert r.served_by == "fake-model" and r.stop_reason == "end_turn"
    assert r.usage == {"input_tokens": 42, "output_tokens": 7, "cache_read_input_tokens": None}
    sent = app.state.requests[-1]
    assert sent["messages"][0]["role"] == "system" and "국방" in sent["messages"][0]["content"]
    assert "[참고 문서]" in sent["messages"][-1]["content"] and "거수자 조치 절차" in sent["messages"][-1]["content"]
    assert sent["stream"] is True and "tools" in sent
    assert all(t["function"]["name"] != "search_defense_docs" for t in sent["tools"])
    # 세션 기록에는 원래 질문(참고 문서 없이)과 답변만 남는다
    assert session.messages[0]["content"] == "거수자 조치 절차 알려줘"
    assert session.messages[1] == {"role": "assistant", "content": [{"type": "text", "text": r.text}]}
    json.dumps(session.messages)
    assert assistant.model_label == "local:fake-model"
    assert assistant.audit.records[-1].model == "local:fake-model"


def test_history_is_converted_for_second_turn(make_local):
    assistant, _, app = make_local()
    session = assistant.new_session("kim")
    assistant.chat(session, "첫 질문")
    assistant.chat(session, "둘째 질문")
    roles = [m["role"] for m in app.state.requests[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "[참고 문서]" not in app.state.requests[-1]["messages"][1]["content"]  # 이전 턴은 원문만


def test_tool_call_round_trip(make_local):
    assistant, _, app = make_local()
    seen = []
    r = assistant.chat(assistant.new_session("kim"), "시간: 071430IOCT26", on_tool=lambda n, i: seen.append((n, i)))
    assert r.tools_called == ["search_defense_docs", "convert_military_time"]
    assert seen[-1] == ("convert_military_time", {"value": "071430IOCT26", "to_zone": "Z"})
    assert "070530ZOCT26" in r.text
    tool_msg = [m for m in app.state.requests[-1]["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_1" and "070530ZOCT26" in tool_msg["content"]


def test_tools_disabled_when_server_rejects_them(make_local):
    assistant, backend, app = make_local(reject_tools=True)
    r = assistant.chat(assistant.new_session("kim"), "시간: 071430IOCT26")
    assert "답입니다" in r.text  # 도구 없이 재시도한 일반 응답
    assert backend._tools_supported is False
    assert [("tools" in q) for q in app.state.requests] == [True, False]
    assistant.chat(assistant.new_session("kim"), "다음 질문")
    assert "tools" not in app.state.requests[-1]  # 이후로는 처음부터 도구 없이


def test_tools_off_setting(make_local):
    assistant, _, app = make_local(local_tools="off")
    assistant.chat(assistant.new_session("kim"), "질문")
    assert "tools" not in app.state.requests[-1]


def test_length_finish_and_no_hits(make_local):
    from defense_assistant.knowledge import DocumentStore

    empty = DocumentStore()
    assistant, _, app = make_local(store_override=empty)
    r = assistant.chat(assistant.new_session("kim"), "길게 말해줘")
    assert r.stop_reason == "max_tokens" and "잘렸습니다" in r.text
    assert r.tools_called == []
    assert "관련 문서가 없습니다" in app.state.requests[-1]["messages"][-1]["content"]


def test_security_gate_still_applies_in_local_mode(make_local):
    assistant, _, app = make_local()
    r = assistant.chat(assistant.new_session("kim"), "II급 비밀 요약 군번 19-70012345")
    assert r.blocked and app.state.requests == []
    r = assistant.chat(assistant.new_session("kim"), "군번 19-70012345 확인")
    assert "[군번]" in app.state.requests[-1]["messages"][-1]["content"] and "19-70012345" not in json.dumps(app.state.requests)


def test_unknown_model_error(make_local):
    assistant, _, _ = make_local(local_model="missing-model")
    with pytest.raises(AssistantError, match="찾을 수 없습니다"):
        assistant.chat(assistant.new_session("kim"), "질문")
    assert assistant.audit.records[-1].event == "error"


def test_connection_error_message(store, glossary):
    settings = Settings(backend="local", local_base_url="http://127.0.0.1:9", local_timeout=2, docs_dir=ROOT / "data/docs", glossary_path=ROOT / "data/glossary.json")
    backend = LocalBackend(settings, build_tools(store, glossary), store, "sys")
    with pytest.raises(BackendError, match="연결할 수 없습니다"):
        backend.check()
    with pytest.raises(BackendError, match="연결할 수 없습니다"):
        backend.run([{"role": "user", "content": "x"}], user_text="x", on_text=None, on_tool=None)


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(backend="openai")
    with pytest.raises(ValueError):
        Settings(local_tools="maybe")
    s = Settings(backend="local")
    assert s.local_base_url.endswith("/v1")


def test_assistant_builds_local_backend_from_settings(store, glossary):
    settings = Settings(backend="local", docs_dir=ROOT / "data/docs", glossary_path=ROOT / "data/glossary.json")
    a = DefenseAssistant(settings, store=store, glossary=glossary, audit=AuditLogger(None))
    assert a.backend.name == "local" and a.model_label == "local:qwen2.5:7b"
    assert "[참고 문서]" in a.system_prompt
