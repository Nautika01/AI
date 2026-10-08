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
import os
import re
from typing import Any

import httpx2 as httpx

from ..config import Settings
from ..knowledge import DocumentStore
from .base import BackendError, OnText, OnTool, TurnResult

log = logging.getLogger(__name__)

# 서버의 컨텍스트 길이를 알 수 없을 때의 기본값 (Ollama 의 기본 num_ctx 와 같음).
# vLLM(--max-model-len)·llama-server(-c) 를 더 크게 띄웠다면 DAI_LOCAL_NUM_CTX 로 맞춰 준다.
DEFAULT_LOCAL_NUM_CTX = 4096
_MIN_NUM_CTX = 512
_MIN_COMPLETION_TOKENS = 64

# 서버가 '입력이 컨텍스트 길이를 넘었다' 고 거절할 때의 문구 (vLLM / llama-server / Ollama / LM Studio)
_CONTEXT_ERROR = re.compile(r"context[ _-]?(length|window|size)|maximum context|max_model_len|n_ctx|too many tokens|prompt is too long|exceed_context", re.IGNORECASE)
# 404 본문이 '모델 없음' 을 뜻하는지 (Ollama: model "x" not found, vLLM: The model `x` does not exist.)
_MODEL_ERROR = re.compile(r"\bmodel\b", re.IGNORECASE)


def _text_tokens(text: str | None) -> int:
    """토큰 수를 보수적으로(많게) 어림한다. 한글 등 비ASCII 문자 1자 = 1토큰, ASCII 3자 = 1토큰."""
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii + (len(text) - non_ascii + 2) // 3


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """요청 메시지(와 도구 정의)의 토큰 수 추정치. 메시지마다 역할 표시 등 4토큰을 더한다."""
    total = 0
    for m in messages:
        content = m.get("content")
        if content and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        total += 4 + _text_tokens(content)
        if m.get("tool_calls"):
            total += _text_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    if tools:
        total += _text_tokens(json.dumps(tools, ensure_ascii=False))
    return total


def _int_setting(settings: Settings, attr: str, env: str, default: int | None) -> int | None:
    """Settings 필드가 있으면 그 값을, 없으면 환경 변수를, 둘 다 없으면 기본값을 쓴다."""
    value = getattr(settings, attr, None)
    if value is None:
        raw = os.environ.get(env, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError as e:
                raise ValueError(f"{env} must be an integer, got {raw!r}") from e
    return default if value is None else int(value)


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:  # noqa: BLE001 - 본문을 아직 읽지 않은 스트림 응답 등
        return ""


class LocalBackend:
    name = "local"

    def __init__(self, settings: Settings, tools: list[Any], store: DocumentStore, system_prompt: str, client: httpx.Client | None = None):
        self.settings = settings
        self.tools = {t.name: t for t in tools}
        self.store = store
        self.system_prompt = system_prompt
        self.base_url = settings.local_base_url.rstrip("/")
        self._tools_supported: bool | None = None if settings.local_tools == "auto" else False
        # 도구를 끈 이유 (local-check·로그 표시용). None 이면 꺼지지 않았거나 아직 판정 전.
        self.tools_disabled_reason: str | None = None if settings.local_tools == "auto" else "DAI_LOCAL_TOOLS=off"
        # --- 컨텍스트 예산: 입력(시스템+이력+질문+도구 정의) + 출력(max_tokens) <= num_ctx ---
        self.num_ctx = _int_setting(settings, "local_num_ctx", "DAI_LOCAL_NUM_CTX", DEFAULT_LOCAL_NUM_CTX) or DEFAULT_LOCAL_NUM_CTX
        if self.num_ctx < _MIN_NUM_CTX:
            raise ValueError(f"DAI_LOCAL_NUM_CTX must be >= {_MIN_NUM_CTX}")
        explicit_max = _int_setting(settings, "local_max_tokens", "DAI_LOCAL_MAX_TOKENS", None)
        if explicit_max is not None and explicit_max < 1:
            raise ValueError("DAI_LOCAL_MAX_TOKENS must be >= 1")
        self._max_tokens_cap = min(settings.max_tokens, explicit_max) if explicit_max else settings.max_tokens
        reserve = min(self._max_tokens_cap, explicit_max or self.num_ctx // 4, self.num_ctx // 2)
        self.input_budget = self.num_ctx - reserve
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
        try:
            return self._run(messages, user_text=user_text, on_text=on_text, on_tool=on_tool)
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001 - 실패는 모두 BackendError 로 감싸 호출 측이 감사 기록·턴 정리를 하게 한다
            raise BackendError(self._korean_error(e)) from e

    def _run(self, messages: list[dict[str, Any]], *, user_text: str, on_text: OnText | None, on_tool: OnTool | None) -> TurnResult:
        last = messages[-1] if messages else {}
        if last.get("role") != "user" or not isinstance(last.get("content"), str):
            raise BackendError("마지막 메시지가 사용자 질문이 아닙니다.")
        prior = self._to_openai_messages(messages[:-1], require_last_user=False)

        rag_enabled = self.settings.local_rag_top_k > 0  # 0 이면 검색·[참고 문서] 주입을 모두 끈다
        hits: list[Any] = []
        if rag_enabled and len(self.store):
            try:
                hits = self.store.search(user_text, top_k=self.settings.local_rag_top_k)
            except Exception as e:  # noqa: BLE001 - 임베딩 서버 장애 등
                raise BackendError(
                    f"지식 베이스 검색 중 오류가 발생했습니다 ({type(e).__name__}): {e}. "
                    "임베딩 서버를 쓰는 경우(DAI_EMBEDDINGS=openai) 서버가 실행 중인지 확인하십시오."
                ) from e
        tools_called: list[str] = []
        if hits:
            tools_called.append("search_defense_docs")
            if on_tool is not None:
                on_tool("search_defense_docs", {"query": user_text[:80], "auto": True})

        request_messages = self._fit_context(prior, user_text, hits if rag_enabled else None)
        text_parts: list[str] = []
        model_name: str | None = None
        usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None, "cache_read_input_tokens": None}
        stop_reason = "tool_use"

        for i in range(self.settings.max_iterations):
            chunk_text, tool_calls, finish, model_name, chunk_usage = self._stream_completion(request_messages, on_text)
            text_parts.append(chunk_text)
            if chunk_usage:
                usage = chunk_usage
            if not tool_calls:
                stop_reason = "max_tokens" if finish == "length" else "end_turn"
                break
            if i == self.settings.max_iterations - 1:
                break  # 마지막 반복의 도구 결과는 모델에 전달되지 않으므로 실행·기록하지 않는다
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

    def _fit_context(self, prior: list[dict[str, Any]], user_text: str, hits: list[Any] | None) -> list[dict[str, Any]]:
        """시스템 프롬프트 + (들어가는 만큼의) 이전 대화 + 현재 질문을 컨텍스트 예산 안에서 구성한다.

        시스템 프롬프트(보안 규칙)는 항상 맨 앞에 온전히 보낸다. 넘치면 참고 문서 수 → 오래된 대화
        순으로 줄이고, 시스템 프롬프트와 질문만으로도 넘치면 서버에 보내지 않고 거절한다
        (서버가 입력 앞부분을 조용히 잘라 보안 규칙이 사라지는 것을 막기 위해).
        `hits` 가 None 이면 RAG 주입을 끈 상태(DAI_LOCAL_RAG_TOP_K=0)로 질문만 보낸다.
        """
        system = {"role": "system", "content": self.system_prompt}
        tool_defs = self._tool_defs()
        budget = self.input_budget
        prior = list(prior)
        pending: str | None = None
        if prior and prior[-1]["role"] == "user":  # 답을 받지 못한 이전 질문은 현재 질문 앞에 합친다
            pending = prior.pop()["content"]

        counts = list(range(len(hits), 0, -1)) if hits else [0]
        current: dict[str, Any] | None = None
        used = 0
        for n in counts:
            content = user_text if hits is None else self._augment(user_text, hits[:n])
            candidate = {"role": "user", "content": content}
            used = estimate_tokens([system, candidate], tool_defs)
            if used <= budget:
                current = candidate
                if hits and n < len(hits):
                    log.info("컨텍스트 예산(%d토큰)에 맞추려고 참고 문서를 %d개에서 %d개로 줄였습니다.", budget, len(hits), n)
                break
        if current is None:
            raise BackendError(
                f"질문이 너무 깁니다(약 {used}토큰, 허용 약 {budget}토큰). 질문을 줄이거나 나눠서 보내십시오. "
                f"모델 서버의 컨텍스트 길이가 더 크다면 DAI_LOCAL_NUM_CTX(현재 {self.num_ctx})를 그에 맞게 늘리십시오."
            )
        if pending:
            merged = {"role": "user", "content": f"{pending}\n\n{current['content']}"}
            merged_used = estimate_tokens([system, merged], tool_defs)
            if merged_used <= budget:
                current, used = merged, merged_used

        kept: list[dict[str, Any]] = []
        for m in reversed(prior):
            cost = estimate_tokens([m])
            if used + cost > budget:
                break
            kept.append(m)
            used += cost
        kept.reverse()
        while kept and kept[0]["role"] != "user":
            kept.pop(0)  # 시스템 다음은 user 로 시작해야 역할 교대 템플릿이 받아들인다
        if len(kept) < len(prior):
            log.info("컨텍스트 예산(%d토큰)에 맞추려고 오래된 대화 %d건을 요청에서 제외했습니다.", budget, len(prior) - len(kept))
        return [system] + kept + [current]

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
            "temperature": self.settings.local_temperature,
        }
        tool_defs = self._tool_defs()
        if tool_defs:
            body["tools"] = tool_defs
        # vLLM·llama-server 는 '입력 + max_tokens > 컨텍스트 길이' 면 400 으로 거절하므로 남은 만큼만 요청한다.
        remaining = self.num_ctx - estimate_tokens(messages, tool_defs)
        body["max_tokens"] = max(min(self._max_tokens_cap, remaining), _MIN_COMPLETION_TOKENS)
        try:
            return self._do_stream(body, on_text)
        except httpx.HTTPStatusError as e:
            # 함수 호출을 지원하지 않는 서버일 수 있다: 도구 없이 한 번 더 시도해 보고, 그 재시도가 성공했을 때만
            # (즉 실패 원인이 tools 였음이 확인됐을 때만) 이후로 도구를 끈다. 컨텍스트 초과는 도구와 무관하므로 제외.
            err_text = _response_text(e.response)
            if tool_defs and self._tools_supported is None and e.response.status_code in (400, 422) and not _CONTEXT_ERROR.search(err_text):
                log.warning("로컬 모델 서버가 도구 포함 요청을 거부해 도구 없이 재시도합니다: %s", err_text[:200])
                body.pop("tools", None)
                try:
                    result = self._do_stream(body, on_text)
                except Exception as e2:  # noqa: BLE001
                    log.warning("도구 없이 재시도해도 실패해 도구 미지원으로 판정하지 않습니다: %s", e2)
                    raise BackendError(self._korean_error(e2)) from e2
                self._tools_supported = False
                self.tools_disabled_reason = f"서버가 도구 포함 요청을 거부함 ({e.response.status_code}): {err_text[:200]}"
                log.warning("로컬 모델 서버가 함수 호출을 지원하지 않아 이후 도구 없이 요청합니다: %s", self.tools_disabled_reason)
                return result
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
    def _to_openai_messages(messages: list[dict[str, Any]], *, require_last_user: bool = True) -> list[dict[str, Any]]:
        """Claude 형식 기록을 OpenAI 형식으로 바꾼다.

        역할 교대를 강제하는 chat template(Mistral·Gemma 등)을 위해 같은 역할이 연속되면 하나로 합친다
        (도구 호출 전 텍스트 + 최종 답, 답을 받지 못한 질문 + 다음 질문 등).
        """
        out: list[dict[str, Any]] = []

        def add(role: str, text: str) -> None:
            if out and out[-1]["role"] == role:
                out[-1] = {"role": role, "content": f"{out[-1]['content']}\n\n{text}"}
            else:
                out.append({"role": role, "content": text})

        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "user" and isinstance(content, str):
                add("user", content)
            elif role == "assistant":
                if isinstance(content, str):
                    text = content
                else:
                    text = "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")
                if text:
                    add("assistant", text)
            # user 역할의 tool_result 목록, system 메시지 등은 로컬 모델에 보내지 않는다.
        if require_last_user and (not out or out[-1]["role"] != "user"):
            raise BackendError("마지막 메시지가 사용자 질문이 아닙니다.")
        return out

    def _korean_error(self, e: BaseException) -> str:
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
            how = "연결 시간 초과" if isinstance(e, httpx.ConnectTimeout) else "연결 거부"
            return f"로컬 모델 서버에 연결할 수 없습니다({how}): {self.base_url}. 모델 서버(Ollama 등)가 실행 중인지, DAI_LOCAL_BASE_URL 주소·포트와 방화벽을 확인하십시오."
        if isinstance(e, httpx.TimeoutException):
            return f"로컬 모델 응답 시간이 초과되었습니다({self.settings.local_timeout:g}초). 더 작은 모델을 쓰거나 DAI_LOCAL_TIMEOUT 을 늘리십시오."
        if isinstance(e, httpx.HTTPStatusError):
            code = e.response.status_code
            text = _response_text(e.response)
            if code == 404:
                if _MODEL_ERROR.search(text) or self.settings.local_model in text:
                    return f"모델 '{self.settings.local_model}' 을(를) 서버에서 찾을 수 없습니다. `ollama pull {self.settings.local_model}` 로 받았는지, DAI_LOCAL_MODEL 이 맞는지 확인하십시오."
                return (
                    f"로컬 모델 서버에서 요청 경로를 찾을 수 없습니다(404): {self.base_url}. "
                    f"DAI_LOCAL_BASE_URL 이 `/v1` 까지 포함하는지 확인하십시오(예: http://127.0.0.1:11434/v1). 서버 응답: {text[:200]}"
                )
            if code in (401, 403):
                return "로컬 모델 서버 인증에 실패했습니다. DAI_LOCAL_API_KEY 를 확인하십시오."
            if code in (400, 413, 422) and _CONTEXT_ERROR.search(text):
                return (
                    "대화가 로컬 모델이 처리할 수 있는 길이(컨텍스트)를 넘었습니다. 새 대화를 시작하거나 질문을 줄이십시오. "
                    f"서버의 컨텍스트 길이에 맞게 DAI_LOCAL_NUM_CTX(현재 {self.num_ctx})를 설정하십시오. 서버 응답: {text[:200]}"
                )
            return f"로컬 모델 서버 오류 ({code}): {text[:200]}"
        if isinstance(e, httpx.HTTPError):
            return f"로컬 모델 서버 통신 오류: {e}"
        return f"로컬 모델 호출 중 오류가 발생했습니다 ({type(e).__name__}): {e}"
