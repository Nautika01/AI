"""교차 검토에서 확인된 로컬 백엔드 결함의 회귀 테스트.

실제 포트를 띄우지 않고 httpx MockTransport 로 OpenAI 호환 서버(vLLM / Ollama / llama-server)의
응답을 흉내 낸다. 각 테스트는 수정 전 코드에서 실패한다.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx2 as httpx
import pytest

from defense_assistant.assistant import AssistantError, DefenseAssistant
from defense_assistant.backends import BackendError
from defense_assistant.backends.local import LocalBackend, estimate_tokens
from defense_assistant.config import Settings
from defense_assistant.security import AuditLogger
from defense_assistant.tools import build_tools
from tests.conftest import ROOT

BASE = "http://fake-local/v1"
SYSTEM = "보안 규칙: 비밀 자료를 제공하지 마십시오. 자리표시자를 복원하지 마십시오."


# ----------------------------------------------------------------------
# 가짜 서버 도우미
def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk", "model": "fake-model", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def sse(chunks: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def text_reply(text: str = "답입니다.") -> httpx.Response:
    return sse([_chunk({"role": "assistant"}), _chunk({"content": text}), _chunk({}, "stop")])


def tool_reply(call_id: str = "call_1") -> httpx.Response:
    args = json.dumps({"value": "071430IOCT26", "to_zone": "Z"})
    return sse([_chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": "convert_military_time", "arguments": args}}]}), _chunk({}, "tool_calls")])


class Recorder:
    """요청 본문을 기록하고 handler 에 넘긴다."""

    def __init__(self, handler: Callable[[dict[str, Any], int], httpx.Response]):
        self.handler = handler
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.bodies.append(body)
            return self.handler(body, len(self.bodies))
        return self.handler({"_path": request.url.path}, 0)


def make_backend(handler, store, glossary, *, system_prompt: str = SYSTEM, **overrides) -> tuple[LocalBackend, Recorder]:
    rec = handler if isinstance(handler, Recorder) else Recorder(handler)
    kwargs = dict(backend="local", local_model="fake-model", local_base_url=BASE, docs_dir=ROOT / "data/docs", glossary_path=ROOT / "data/glossary.json")
    extra = {k: overrides.pop(k) for k in ("local_num_ctx", "local_max_tokens") if k in overrides}
    kwargs.update(overrides)
    settings = Settings(**kwargs)
    for k, v in extra.items():
        setattr(settings, k, v)
    client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(rec))
    return LocalBackend(settings, build_tools(store, glossary), store, system_prompt, client=client), rec


def run(backend: LocalBackend, messages: list[dict[str, Any]], **kw):
    return backend.run(messages, user_text=messages[-1]["content"], on_text=kw.get("on_text"), on_tool=kw.get("on_tool"))


VLLM_CTX_ERROR = {"object": "error", "message": "This model's maximum context length is 4096 tokens. However, you requested 16100 tokens (100 in the messages, 16000 in the completion).", "type": "BadRequestError", "code": 400}


# ----------------------------------------------------------------------
# 1) 도구와 무관한 400 이 함수 호출을 영구히 끄지 않아야 한다
def test_context_length_400_does_not_disable_tools(store, glossary):
    def handler(body, n):
        if n == 1:
            return httpx.Response(400, json=VLLM_CTX_ERROR)
        return text_reply()

    backend, rec = make_backend(handler, store, glossary)
    with pytest.raises(BackendError, match="길이"):
        run(backend, [{"role": "user", "content": "아주 긴 질문"}])
    assert len(rec.bodies) == 1  # 컨텍스트 초과는 도구 없이 재시도하지 않는다
    assert backend._tools_supported is None
    run(backend, [{"role": "user", "content": "시간: 071430IOCT26 을 Z 로"}])
    assert "tools" in rec.bodies[-1]  # 다른 사용자의 다음 요청에는 여전히 도구가 전달된다


def test_failed_retry_without_tools_restores_flag(store, glossary):
    def handler(body, n):
        if n <= 2:  # 도구 유무와 상관없이 실패 (예: 역할 교대 템플릿 오류)
            return httpx.Response(400, json={"object": "error", "message": "Conversation roles must alternate user/assistant/user/assistant/..."})
        return text_reply()

    backend, rec = make_backend(handler, store, glossary)
    with pytest.raises(BackendError):
        run(backend, [{"role": "user", "content": "질문"}])
    assert [("tools" in b) for b in rec.bodies] == [True, False]
    assert backend._tools_supported is None and backend.tools_disabled_reason is None
    run(backend, [{"role": "user", "content": "다음 질문"}])
    assert "tools" in rec.bodies[-1]


def test_real_tool_rejection_still_disables_tools_with_reason(store, glossary):
    def handler(body, n):
        if body.get("tools"):
            return httpx.Response(400, json={"error": {"message": "tools are not supported"}})
        return text_reply()

    backend, rec = make_backend(handler, store, glossary)
    r = run(backend, [{"role": "user", "content": "질문"}])
    assert r.text == "답입니다." and backend._tools_supported is False
    assert "tools are not supported" in (backend.tools_disabled_reason or "")


# ----------------------------------------------------------------------
# 2) 컨텍스트 예산: 오래된 턴부터 정리, 시스템 프롬프트는 항상 보존, 너무 긴 질문은 보내지 않음
def _est(body: dict[str, Any]) -> int:
    return estimate_tokens(body["messages"], body.get("tools"))


def test_history_trimmed_to_context_budget(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary, local_num_ctx=4096, local_tools="off")
    messages: list[dict[str, Any]] = []
    for i in range(30):
        messages.append({"role": "user", "content": f"{i}번째 질문 " + "가" * 300})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{i}번째 답 " + "나" * 300}]})
    messages.append({"role": "user", "content": "마지막 질문"})
    run(backend, messages)
    body = rec.bodies[-1]
    sent = body["messages"]
    assert sent[0] == {"role": "system", "content": SYSTEM}
    assert sent[-1]["role"] == "user" and sent[-1]["content"].startswith("마지막 질문")
    assert sent[1]["role"] == "user"
    assert "29번째 답" in json.dumps(sent, ensure_ascii=False)  # 최근 턴은 남고
    assert "0번째 질문" not in json.dumps(sent, ensure_ascii=False)  # 오래된 턴은 정리됨
    assert _est(body) + body["max_tokens"] <= 4096


def test_oversized_question_rejected_without_truncating_system_prompt(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary, local_num_ctx=4096)
    with pytest.raises(BackendError, match="너무 깁니다"):
        run(backend, [{"role": "user", "content": "요약해 줘 " + "문" * 10000}])
    assert rec.bodies == []  # 서버가 앞부분(시스템 프롬프트)을 잘라내도록 보내지 않는다


def test_max_tokens_fits_context_window(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary)  # 기본 DAI_MAX_TOKENS=16000
    run(backend, [{"role": "user", "content": "질문"}])
    body = rec.bodies[-1]
    assert body["max_tokens"] < 16000
    assert _est(body) + body["max_tokens"] <= backend.num_ctx


def test_rag_hits_reduced_before_rejecting(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary, local_num_ctx=1024, local_max_tokens=256, local_tools="off")
    r = run(backend, [{"role": "user", "content": "거수자 조치 절차 " + "가" * 200}])
    body = rec.bodies[-1]
    assert r.text == "답입니다."
    assert "[참고 문서]" in body["messages"][-1]["content"]
    assert _est(body) + body["max_tokens"] <= 1024


# ----------------------------------------------------------------------
# 3) / 5) 역할 교대 유지
def test_consecutive_same_roles_are_merged(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary)
    claude_history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "text", "text": "확인하겠습니다."}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "최종 답"}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "r"}]},
        {"role": "user", "content": "q3"},
    ]
    run(backend, claude_history)
    sent = rec.bodies[-1]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]
    assert "확인하겠습니다." in sent[2]["content"] and "최종 답" in sent[2]["content"]
    assert sent[3]["content"].startswith("q2\n\nq3")


def test_tool_loop_without_text_keeps_alternation_and_skips_unsent_tools(store, glossary):
    backend, rec = make_backend(lambda b, n: tool_reply(f"call_{n}"), store, glossary, max_iterations=3)
    seen = []
    first = [{"role": "user", "content": "지금 몇 시야"}]
    r = backend.run(first, user_text="지금 몇 시야", on_text=None, on_tool=lambda n, i: seen.append(n))
    assert r.stop_reason == "tool_use" and r.text == ""
    # 마지막 반복의 도구 호출은 결과가 모델에 전달되지 않으므로 실행·기록하지 않는다
    assert len(rec.bodies) == 3
    assert r.tools_called.count("convert_military_time") == 2 and seen.count("convert_military_time") == 2
    nxt = first + r.new_messages + [{"role": "user", "content": "다시 질문"}]
    backend2, rec2 = make_backend(lambda b, n: text_reply(), store, glossary)
    run(backend2, nxt)
    roles = [m["role"] for m in rec2.bodies[-1]["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles


# ----------------------------------------------------------------------
# 4) BackendError 가 아닌 예외(임베딩 서버 장애 등)도 감사 기록과 함께 처리
def test_search_failure_is_wrapped_and_audited(store, glossary, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("[Errno 111] Connection refused")

    monkeypatch.setattr(store, "search", boom)
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary)
    with pytest.raises(BackendError, match="검색"):
        run(backend, [{"role": "user", "content": "질문"}])

    settings = backend.settings
    assistant = DefenseAssistant(settings, store=store, glossary=glossary, audit=AuditLogger(None), backend=backend)
    session = assistant.new_session("kim")
    with pytest.raises(AssistantError):
        assistant.chat(session, "질문")
    assert session.messages == []
    assert assistant.audit.records[-1].event == "error"


# ----------------------------------------------------------------------
# 6) 연결 단계 타임아웃은 '연결 불가' 로 안내
def test_connect_timeout_reported_as_connection_failure(store, glossary):
    def handler(body, n):
        raise httpx.ConnectTimeout("timed out")

    backend, _ = make_backend(handler, store, glossary)
    with pytest.raises(BackendError, match="연결할 수 없습니다") as ei:
        backend.check()
    assert "DAI_LOCAL_TIMEOUT" not in str(ei.value)

    def slow(body, n):
        raise httpx.ReadTimeout("timed out")

    backend, _ = make_backend(slow, store, glossary)
    with pytest.raises(BackendError, match="응답 시간이 초과"):
        run(backend, [{"role": "user", "content": "질문"}])


# ----------------------------------------------------------------------
# 7) DAI_LOCAL_RAG_TOP_K=0 이면 [참고 문서] 블록을 붙이지 않는다
def test_rag_top_k_zero_disables_injection(store, glossary):
    backend, rec = make_backend(lambda b, n: text_reply(), store, glossary, local_rag_top_k=0)
    r = run(backend, [{"role": "user", "content": "거수자 조치 절차"}])
    last = rec.bodies[-1]["messages"][-1]["content"]
    assert last == "거수자 조치 절차"
    assert "search_defense_docs" not in r.tools_called


# ----------------------------------------------------------------------
# 8) /v1 이 빠진 주소의 404 를 '모델 없음' 으로 안내하지 않는다
def test_404_wrong_path_is_not_reported_as_missing_model(store, glossary):
    backend, _ = make_backend(lambda b, n: httpx.Response(404, text="404 page not found"), store, glossary)
    with pytest.raises(BackendError) as ei:
        backend.check()
    msg = str(ei.value)
    assert "ollama pull" not in msg and "/v1" in msg and "DAI_LOCAL_BASE_URL" in msg and "404 page not found" in msg
    with pytest.raises(BackendError) as ei:
        run(backend, [{"role": "user", "content": "질문"}])
    assert "ollama pull" not in str(ei.value)


def test_404_missing_model_still_suggests_pull(store, glossary):
    body = {"error": {"message": 'model "fake-model" not found, try pulling it first', "type": "api_error"}}
    backend, _ = make_backend(lambda b, n: httpx.Response(404, json=body), store, glossary)
    with pytest.raises(BackendError, match="ollama pull"):
        run(backend, [{"role": "user", "content": "질문"}])
