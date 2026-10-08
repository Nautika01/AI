"""로컬 모델 백엔드 (OpenAI 호환 Chat Completions API).

Ollama, vLLM, llama.cpp(llama-server), LM Studio 등은 모두 `/v1/chat/completions` 규격을 제공하므로
`DAI_LOCAL_BASE_URL` 만 바꾸면 그대로 동작한다. 망 분리 환경을 위해 외부 네트워크를 쓰지 않는다.

작은 모델은 스스로 검색 도구를 호출하는 판단이 서툴기 때문에, 이 백엔드는 **매 턴 지식 베이스를
먼저 검색해 참고 문서를 질문에 붙여 준다**(RAG 주입). 나머지 도구(시간 변환 등)는 모델 서버가
함수 호출을 지원할 때만 사용하며, 지원하지 않으면 자동으로 끈다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2 as httpx

from ..config import Settings
from ..knowledge import DocumentStore
from .base import BackendError, OnText, OnTool, TurnResult

log = logging.getLogger(__name__)


class LocalBackend:
    name = "local"

    def __init__(self, settings: Settings, tools: list[Any], store: DocumentStore, system_prompt: str, client: httpx.Client | None = None):
        self.settings = settings
        self.tools = {t.name: t for t in tools}
        self.store = store
        self.system_prompt = system_prompt
        self.base_url = settings.local_base_url.rstrip("/")
        self._tools_supported: bool | None = None if settings.local_tools == "auto" else False
        headers = {"Content-Type": "application/json"}
        if settings.local_api_key:
            headers["Authorization"] = f"Bearer {settings.local_api_key}"
        self.client = client or httpx.Client(base_url=self.base_url, headers=headers, timeout=httpx.Timeout(settings.local_timeout, connect=10.0))

    @property
    def label(self) -> str:
        return f"local:{self.settings.local_model}"

    # ------------------------------------------------------------------
    def check(self) -> dict[str, Any]:
        """서버 연결과 모델 존재 여부를 확인한다 (CLI `local-check` 용)."""
        try:
            r = self.client.get("/models")
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            raise BackendError(self._korean_error(e)) from e
        return {"base_url": self.base_url, "models": models, "model_available": self.settings.local_model in models}

    # ------------------------------------------------------------------
    def run(self, messages: list[dict[str, Any]], *, user_text: str, on_text: OnText | None, on_tool: OnTool | None) -> TurnResult:
        history = self._to_openai_messages(messages)
        hits = self.store.search(user_text, top_k=self.settings.local_rag_top_k) if len(self.store) else []
        tools_called: list[str] = []
        if hits:
            tools_called.append("search_defense_docs")
            if on_tool is not None:
                on_tool("search_defense_docs", {"query": user_text[:80], "auto": True})
        history[-1] = {"role": "user", "content": self._augment(user_text, hits)}

        request_messages = [{"role": "system", "content": self.system_prompt}] + history
        text_parts: list[str] = []
        model_name: str | None = None
        usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None, "cache_read_input_tokens": None}
        stop_reason = "tool_use"

        for _ in range(self.settings.max_iterations):
            chunk_text, tool_calls, finish, model_name, chunk_usage = self._stream_completion(request_messages, on_text)
            text_parts.append(chunk_text)
            if chunk_usage:
                usage = chunk_usage
            if not tool_calls:
                stop_reason = "max_tokens" if finish == "length" else "end_turn"
                break
            request_messages.append({"role": "assistant", "content": chunk_text or None, "tool_calls": tool_calls})
            for call in tool_calls:
                name = call["function"]["name"]
                tools_called.append(name)
                raw_args = call["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    result = f"오류: 도구 인자가 올바른 JSON이 아닙니다: {raw_args[:200]}"
                else:
                    if on_tool is not None:
                        on_tool(name, args)
                    result = self._call_tool(name, args)
                request_messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        text = "".join(text_parts)
        new_messages = [{"role": "assistant", "content": [{"type": "text", "text": text}]}] if text else []
        return TurnResult(text=text, new_messages=new_messages, tools_called=tools_called, stop_reason=stop_reason, model=model_name or self.settings.local_model, usage=usage)

    # ------------------------------------------------------------------
    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"오류: 알 수 없는 도구 {name}"
        try:
            out = tool.call(args)
            return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 - 도구 오류는 모델에게 돌려준다
            return f"오류: {e}"

    def _augment(self, user_text: str, hits: list[Any]) -> str:
        if not hits:
            return f"{user_text}\n\n[참고 문서]\n(지식 베이스에 관련 문서가 없습니다. 일반 지식으로 답하되 그 점을 밝히십시오.)"
        return f"{user_text}\n\n[참고 문서]\n{self.store.format_hits(hits, max_chars=700)}"

    def _tool_defs(self) -> list[dict[str, Any]] | None:
        if self._tools_supported is False:
            return None
        defs = []
        for name, t in self.tools.items():
            if name == "search_defense_docs":
                continue  # 매 턴 자동 주입하므로 제외
            d = t.to_dict()
            defs.append({"type": "function", "function": {"name": d["name"], "description": d.get("description", ""), "parameters": d["input_schema"]}})
        return defs or None

    def _stream_completion(self, messages: list[dict[str, Any]], on_text: OnText | None) -> tuple[str, list[dict[str, Any]], str | None, str | None, dict[str, int | None] | None]:
        body: dict[str, Any] = {
            "model": self.settings.local_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.local_temperature,
        }
        tool_defs = self._tool_defs()
        if tool_defs:
            body["tools"] = tool_defs
        try:
            return self._do_stream(body, on_text)
        except httpx.HTTPStatusError as e:
            # 함수 호출을 지원하지 않는 서버: 도구 없이 한 번 더 시도하고 이후로는 끈다.
            if tool_defs and self._tools_supported is None and e.response.status_code in (400, 422):
                log.warning("로컬 모델 서버가 도구 호출을 거부해 도구 없이 재시도합니다: %s", e.response.text[:200])
                self._tools_supported = False
                body.pop("tools", None)
                try:
                    return self._do_stream(body, on_text)
                except Exception as e2:  # noqa: BLE001
                    raise BackendError(self._korean_error(e2)) from e2
            raise BackendError(self._korean_error(e)) from e
        except Exception as e:  # noqa: BLE001
            raise BackendError(self._korean_error(e)) from e

    def _do_stream(self, body: dict[str, Any], on_text: OnText | None) -> tuple[str, list[dict[str, Any]], str | None, str | None, dict[str, int | None] | None]:
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        model: str | None = None
        usage: dict[str, int | None] | None = None
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"), "cache_read_input_tokens": None}
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                        if on_text is not None:
                            on_text(content)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = calls.setdefault(idx, {"id": tc.get("id") or f"call_{idx}", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        tool_calls = [calls[i] for i in sorted(calls)]
        tool_calls = [c for c in tool_calls if c["function"]["name"]]
        self._tools_supported = True if (tool_calls or "tools" in body) and self._tools_supported is None else self._tools_supported
        return "".join(text_parts), tool_calls, finish, model, usage

    @staticmethod
    def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "user" and isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(content, str):
                    text = content
                else:
                    text = "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")
                if text:
                    out.append({"role": "assistant", "content": text})
            # user 역할의 tool_result 목록, system 메시지 등은 로컬 모델에 보내지 않는다.
        if not out or out[-1]["role"] != "user":
            raise BackendError("마지막 메시지가 사용자 질문이 아닙니다.")
        return out

    def _korean_error(self, e: BaseException) -> str:
        if isinstance(e, httpx.ConnectError):
            return f"로컬 모델 서버에 연결할 수 없습니다: {self.base_url}. 모델 서버(Ollama 등)가 실행 중인지, DAI_LOCAL_BASE_URL 이 맞는지 확인하십시오."
        if isinstance(e, httpx.TimeoutException):
            return f"로컬 모델 응답 시간이 초과되었습니다({self.settings.local_timeout:g}초). 더 작은 모델을 쓰거나 DAI_LOCAL_TIMEOUT 을 늘리십시오."
        if isinstance(e, httpx.HTTPStatusError):
            code = e.response.status_code
            if code == 404:
                return f"모델 '{self.settings.local_model}' 을(를) 서버에서 찾을 수 없습니다. `ollama pull {self.settings.local_model}` 로 받았는지, DAI_LOCAL_MODEL 이 맞는지 확인하십시오."
            if code in (401, 403):
                return "로컬 모델 서버 인증에 실패했습니다. DAI_LOCAL_API_KEY 를 확인하십시오."
            return f"로컬 모델 서버 오류 ({code}): {e.response.text[:200]}"
        if isinstance(e, httpx.HTTPError):
            return f"로컬 모델 서버 통신 오류: {e}"
        return f"로컬 모델 호출 중 오류가 발생했습니다 ({type(e).__name__}): {e}"
