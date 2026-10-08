"""Claude API 백엔드 (툴 러너 스트리밍, adaptive thinking, 프롬프트 캐싱, 서버측 폴백)."""

from __future__ import annotations

from typing import Any

import anthropic

from ..config import Settings
from .base import BackendError, OnText, OnTool, TurnResult, to_jsonable

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeBackend:
    name = "claude"

    def __init__(self, settings: Settings, tools: list[Any], system_prompt: str, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.tools = tools
        self.system_prompt = system_prompt
        self._client = client

    @property
    def label(self) -> str:
        return self.settings.model

    # 클라이언트는 실제 호출 시점에 생성한다 (API 키 없이도 검색·도구·차단 기능은 동작).
    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def request_params(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = dict(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=self.tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            max_iterations=self.settings.max_iterations,
            stream=True,
        )
        if self.settings.fallbacks == "default":
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def run(self, messages: list[dict[str, Any]], *, user_text: str, on_text: OnText | None, on_tool: OnTool | None) -> TurnResult:
        try:
            return self._run(messages, on_text=on_text, on_tool=on_tool)
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001 - 모델 호출 실패는 모두 한국어 메시지로 감싼다
            raise BackendError(korean_error(e)) from e

    def _run(self, messages: list[dict[str, Any]], *, on_text: OnText | None, on_tool: OnTool | None) -> TurnResult:
        runner = self.client.beta.messages.tool_runner(**self.request_params(list(messages)))
        new_messages: list[dict[str, Any]] = []
        tools_called: list[str] = []
        final = None
        for stream in runner:
            for event in stream:
                if event.type == "text" and on_text is not None:
                    on_text(event.text)
            final = stream.get_final_message()
            for block in final.content:
                if block.type == "tool_use":
                    tools_called.append(block.name)
                    if on_tool is not None:
                        on_tool(block.name, dict(block.input) if isinstance(block.input, dict) else {"input": block.input})
            # 러너는 내부 기록을 노출하지 않으므로 세션 기록을 직접 복제한다.
            # 저장·전송이 모두 가능하도록 SDK 객체는 평범한 dict 로 바꿔 둔다.
            new_messages.append({"role": "assistant", "content": to_jsonable(final.content)})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                new_messages.append(to_jsonable(tool_response))
        if final is None:
            raise BackendError("모델이 응답을 돌려주지 않았습니다.")

        usage = final.usage
        fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in (getattr(usage, "iterations", None) or []))
        refusal_detail = None
        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            if details is not None:
                cat = getattr(details, "category", None)
                expl = getattr(details, "explanation", None)
                refusal_detail = f"{cat or '미분류'}{' — ' + expl if expl else ''}"
        return TurnResult(
            text="".join(b.text for b in final.content if b.type == "text"),
            new_messages=new_messages,
            tools_called=tools_called,
            stop_reason=final.stop_reason,
            model=final.model,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
            fallback_used=fallback_used,
            refusal_detail=refusal_detail,
        )


def korean_error(e: BaseException) -> str:
    if "Could not resolve authentication method" in str(e):
        return "API 인증 정보가 없습니다. ANTHROPIC_API_KEY 환경 변수를 설정하거나 `ant auth login`으로 로그인하십시오."
    if isinstance(e, anthropic.AuthenticationError):
        return "API 인증에 실패했습니다. ANTHROPIC_API_KEY 또는 `ant auth login` 프로필을 확인하십시오."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "API 키에 필요한 권한이 없습니다."
    if isinstance(e, anthropic.NotFoundError):
        return f"모델 또는 엔드포인트를 찾을 수 없습니다 (DAI_MODEL 확인): {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.RateLimitError):
        return "요청 한도를 초과했습니다. 잠시 후 다시 시도하십시오."
    if isinstance(e, anthropic.BadRequestError):
        return f"잘못된 요청입니다: {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "API 서버 오류입니다. 잠시 후 다시 시도하십시오."
    if isinstance(e, anthropic.APIConnectionError):
        return "API 서버에 연결할 수 없습니다. 네트워크 또는 프록시 설정을 확인하십시오."
    if isinstance(e, anthropic.APIError):
        return f"API 오류: {getattr(e, 'message', e)}"
    return f"모델 호출 중 오류가 발생했습니다 ({type(e).__name__}): {e}"
