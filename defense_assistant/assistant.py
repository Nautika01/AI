"""국방 특화 AI 비서 핵심 로직.

처리 흐름 (한 턴):
1. 비밀 등급 표기 탐지 → 허용 등급 초과 시 모델 호출 없이 차단
2. 민감정보 마스킹 → 자리표시자로 치환된 텍스트만 모델로 전송
3. Claude 툴 러너(스트리밍)로 응답 생성, 로컬 도구 호출
4. 거부(refusal)·토큰 초과 등 종료 사유 처리
5. 감사 로그 기록 (원문은 해시만 저장)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import anthropic

from .config import Settings
from .knowledge import DocumentStore
from .prompts import build_system_prompt
from .security import AuditLogger, AuditRecord, Classification, ClassificationResult, RedactionResult, classify_text, redact
from .security.audit import sha256_text
from .security.classification import block_message
from .tools import build_tools
from .tools.glossary import load_glossary

log = logging.getLogger(__name__)

OnText = Callable[[str], None]
OnTool = Callable[[str, dict[str, Any]], None]

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AssistantError(RuntimeError):
    """모델 호출 실패를 한국어 메시지로 감싼 예외."""


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str = "anonymous"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turns: int = 0
    max_classification: Classification | None = None
    """사용자 비밀취급인가 등급. 설정의 전역 허용 등급과 비교해 더 낮은 쪽이 적용된다."""

    def reset(self) -> None:
        self.messages.clear()
        self.turns = 0


@dataclass
class ChatResult:
    text: str
    blocked: bool
    classification: ClassificationResult
    redaction: RedactionResult
    tools_called: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    served_by: str | None = None
    fallback_used: bool = False
    usage: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "blocked": self.blocked,
            "classification": self.classification.level.name,
            "markings": self.classification.markings,
            "redactions": self.redaction.counts,
            "tools_called": self.tools_called,
            "stop_reason": self.stop_reason,
            "served_by": self.served_by,
            "fallback_used": self.fallback_used,
            "usage": self.usage,
        }


class DefenseAssistant:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: anthropic.Anthropic | None = None,
        store: DocumentStore | None = None,
        glossary: dict[str, dict[str, str]] | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._client = client
        self.store = store if store is not None else DocumentStore.from_directory(self.settings.docs_dir)
        self.glossary = glossary if glossary is not None else load_glossary(self.settings.glossary_path)
        self.audit = audit if audit is not None else AuditLogger(self.settings.audit_log)
        self.tools = build_tools(self.store, self.glossary)
        self.system_prompt = build_system_prompt(self.store.titles)
        log.info("지식 베이스 %d개 청크, 용어 %d개, 도구 %d개 적재", len(self.store), len(self.glossary), len(self.tools))

    # 클라이언트는 실제 호출 시점에 생성한다 (API 키 없이도 검색·도구·차단 기능은 동작).
    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def new_session(self, user_id: str = "anonymous", *, max_classification: Classification | None = None) -> Session:
        return Session(user_id=user_id, max_classification=max_classification)

    def effective_max_classification(self, session: Session) -> Classification:
        limit = self.settings.max_classification
        if session.max_classification is not None and session.max_classification < limit:
            limit = session.max_classification
        return limit

    # ------------------------------------------------------------------
    def chat(self, session: Session, user_text: str, *, on_text: OnText | None = None, on_tool: OnTool | None = None) -> ChatResult:
        """한 턴을 처리한다. `on_text`는 스트리밍 텍스트 조각, `on_tool`은 도구 호출(이름, 입력)을 받는다."""
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("입력이 비어 있습니다.")

        classification = classify_text(user_text)
        base = dict(
            session_id=session.session_id,
            user_id=session.user_id,
            input_sha256=sha256_text(user_text),
            input_chars=len(user_text),
            classification=classification.level.name,
        )

        # 1) 비밀 등급 게이트 — 모델 호출 전에 차단 (전역 설정과 사용자 인가 등급 중 낮은 쪽)
        max_level = self.effective_max_classification(session)
        if not classification.allowed_under(max_level):
            text = block_message(classification, max_level)
            self.audit.write(AuditRecord(event="blocked", detail="classification_marking", **base))
            return ChatResult(text=text, blocked=True, classification=classification, redaction=RedactionResult(text=user_text))

        # 2) 민감정보 마스킹
        redaction = redact(user_text, enabled=self.settings.redaction == "mask")

        # 3) 모델 호출
        session.messages.append({"role": "user", "content": redaction.text})
        try:
            final, new_messages, tools_called = self._run_model(session.messages, on_text=on_text, on_tool=on_tool)
        except AssistantError:
            session.messages.pop()
            raise
        except Exception as e:  # noqa: BLE001 - 모델 호출 실패는 모두 한국어 메시지로 감싼다
            session.messages.pop()  # 실패한 턴은 기록에 남기지 않는다
            self.audit.write(AuditRecord(event="error", redactions=redaction.counts, detail=f"{type(e).__name__}: {getattr(e, 'message', e)}", **base))
            raise AssistantError(_korean_error(e)) from e
        session.messages.extend(new_messages)
        session.turns += 1

        # 4) 종료 사유 처리
        text = "".join(b.text for b in final.content if b.type == "text")
        if final.stop_reason == "refusal":
            text = _refusal_text(final)
        elif final.stop_reason == "max_tokens":
            text += "\n\n(응답이 최대 길이에 도달해 잘렸습니다. 이어서 답하도록 요청해 주십시오.)"
        elif final.stop_reason == "tool_use":
            text += "\n\n(도구 호출 횟수 한도에 도달해 중단했습니다. 질문을 나눠서 다시 요청해 주십시오.)"

        usage = final.usage
        fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in (getattr(usage, "iterations", None) or []))
        result = ChatResult(
            text=text,
            blocked=False,
            classification=classification,
            redaction=redaction,
            tools_called=tools_called,
            stop_reason=final.stop_reason,
            served_by=final.model,
            fallback_used=fallback_used,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
        )

        # 5) 감사 로그
        self.audit.write(AuditRecord(
            event="chat",
            redactions=redaction.counts,
            tools_called=tools_called,
            model=self.settings.model,
            served_by=final.model,
            stop_reason=final.stop_reason,
            input_tokens=result.usage["input_tokens"],
            output_tokens=result.usage["output_tokens"],
            cache_read_tokens=result.usage["cache_read_input_tokens"],
            output_sha256=sha256_text(text),
            output_chars=len(text),
            **base,
        ))
        return result

    # ------------------------------------------------------------------
    def _request_params(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _run_model(self, messages: list[dict[str, Any]], *, on_text: OnText | None, on_tool: OnTool | None):
        """툴 러너를 돌리고 (최종 메시지, 이번 턴에 추가된 메시지들, 호출된 도구 이름) 을 돌려준다."""
        runner = self.client.beta.messages.tool_runner(**self._request_params(list(messages)))
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
            new_messages.append({"role": "assistant", "content": _to_jsonable(final.content)})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                new_messages.append(_to_jsonable(tool_response))
        if final is None:
            raise AssistantError("모델이 응답을 돌려주지 않았습니다.")
        return final, new_messages, tools_called


def _to_jsonable(obj: Any) -> Any:
    """SDK pydantic 모델을 포함한 메시지 구조를 JSON 직렬화 가능한 dict/list 로 변환한다."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _refusal_text(final: Any) -> str:
    details = getattr(final, "stop_details", None)
    reason = ""
    if details is not None:
        cat = getattr(details, "category", None)
        expl = getattr(details, "explanation", None)
        reason = f" (사유: {cat or '미분류'}{' — ' + expl if expl else ''})"
    return "요청하신 내용은 안전 정책에 따라 답변할 수 없습니다." + reason + " 질문을 바꾸어 다시 문의해 주십시오."


def _korean_error(e: BaseException) -> str:
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
