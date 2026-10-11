"""Claude API 백엔드 (툴 러너 스트리밍, adaptive thinking, 프롬프트 캐싱, 서버측 폴백)."""

from __future__ import annotations

from typing import Any

import anthropic

from ..config import Settings
from .base import USAGE_KEYS, BackendError, OnText, OnTool, TurnResult, to_jsonable

FALLBACK_BETA = "server-side-fallback-2026-07-01"

REFUSAL_PLACEHOLDER = "(안전 정책에 따라 답변하지 않음)"
EMPTY_PLACEHOLDER = "(응답 없음)"


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
        runner = self.client.beta.messages.tool_runner(**self.request_params(sanitize_history(messages)))
        new_messages: list[dict[str, Any]] = []
        tools_called: list[str] = []
        totals: dict[str, int | None] = {k: None for k in USAGE_KEYS}
        fallback_used = False
        final = None
        for stream in runner:
            for event in stream:
                if event.type == "text" and on_text is not None:
                    on_text(event.text)
            final = stream.get_final_message()
            # 툴 러너는 반복마다 새 API 호출을 하므로 사용량은 반복별로 합산한다.
            usage = getattr(final, "usage", None)
            for key in USAGE_KEYS:
                val = getattr(usage, key, None)
                if isinstance(val, int):
                    totals[key] = (totals[key] or 0) + val
            if any(getattr(it, "type", None) == "fallback_message" for it in (getattr(usage, "iterations", None) or [])):
                fallback_used = True
            # 러너는 내부 기록을 노출하지 않으므로 세션 기록을 직접 복제한다.
            # 저장·전송이 모두 가능하도록 SDK 객체는 평범한 dict 로 바꿔 둔다.
            content = to_jsonable(final.content)
            if final.stop_reason == "tool_use":
                # 도구는 stop_reason 이 tool_use 일 때만 실행한다(SDK 러너와 같은 규칙).
                for block in final.content:
                    if block.type == "tool_use":
                        tools_called.append(block.name)
                        if on_tool is not None:
                            on_tool(block.name, dict(block.input) if isinstance(block.input, dict) else {"input": block.input})
                new_messages.append({"role": "assistant", "content": content})
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    new_messages.append(to_jsonable(tool_response))
                continue
            if final.stop_reason in ("pause_turn", "compaction"):
                # 서버가 이어서 처리할 미완료 턴은 그대로 둔다.
                new_messages.append({"role": "assistant", "content": content})
                continue
            if final.stop_reason == "refusal":
                # 거부된 부분 출력은 버리고 안내 문장 하나로 대체한다(빈 content 는 다음 요청을 400 으로 만든다).
                content = [{"type": "text", "text": REFUSAL_PLACEHOLDER}]
            else:
                # max_tokens 등으로 끝난 턴의 tool_use 는 입력이 잘렸을 수 있어 실행하지 않는다.
                # tool_result 없는 tool_use 는 다음 요청을 400 으로 만들므로 기록에서도 뺀다.
                content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_use")]
                if not any(isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in content):
                    content.append({"type": "text", "text": EMPTY_PLACEHOLDER})
            new_messages.append({"role": "assistant", "content": content})
        if final is None:
            raise BackendError("모델이 응답을 돌려주지 않았습니다.")

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
            usage=totals,
            fallback_used=fallback_used,
            refusal_detail=refusal_detail,
        )


def _is_empty_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return False


def sanitize_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """전송용 기록 사본. 예전에 저장된 빈 assistant 메시지(거부 턴 등)는 안내 문장으로 채운다.

    API 는 마지막을 제외한 모든 메시지의 content 가 비어 있지 않아야 하므로, 이미 DB 에
    저장된 세션도 계속 쓸 수 있도록 전송 직전에 보정한다. 원본 기록은 바꾸지 않는다.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant" and _is_empty_content(m.get("content")):
            m = {**m, "content": [{"type": "text", "text": EMPTY_PLACEHOLDER}]}
        out.append(m)
    return out


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
    if _is_context_overflow(e):
        return "대화가 너무 길어져 모델의 처리 한도를 넘었습니다. 새 대화를 시작하십시오."
    if isinstance(e, anthropic.BadRequestError):
        return f"잘못된 요청입니다: {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "API 서버 오류입니다. 잠시 후 다시 시도하십시오."
    if isinstance(e, anthropic.APIConnectionError):
        return "API 서버에 연결할 수 없습니다. 네트워크 또는 프록시 설정을 확인하십시오."
    if isinstance(e, anthropic.APIError):
        return f"API 오류: {getattr(e, 'message', e)}"
    return f"모델 호출 중 오류가 발생했습니다 ({type(e).__name__}): {e}"


def _is_context_overflow(e: BaseException) -> bool:
    """누적 대화가 컨텍스트 창을 넘은 400(prompt is too long) 또는 413(request_too_large) 인지 판별한다."""
    if isinstance(e, anthropic.APIStatusError) and e.status_code == 413:
        return True
    if isinstance(e, anthropic.BadRequestError):
        msg = str(getattr(e, "message", "") or e).lower()
        return "prompt is too long" in msg or "request_too_large" in msg or "context window" in msg
    return False
