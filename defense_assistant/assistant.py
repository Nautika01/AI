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
from typing import Any

import anthropic

from .backends import BackendError, ModelBackend
from .backends.base import OnText, OnTool
from .config import Settings, build_store
from .knowledge import DocumentStore
from .prompts import build_local_system_prompt, build_system_prompt
from .security import AuditLogger, AuditRecord, Classification, ClassificationResult, RedactionResult, classify_text, redact
from .security.audit import sha256_text
from .security.classification import block_message
from .tools import build_tools
from .tools.glossary import load_glossary

log = logging.getLogger(__name__)


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
        backend: ModelBackend | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._client = client
        self.store = store if store is not None else build_store(self.settings)
        self.glossary = glossary if glossary is not None else load_glossary(self.settings.glossary_path)
        self.audit = audit if audit is not None else AuditLogger(self.settings.audit_log)
        self.tools = build_tools(self.store, self.glossary)
        self.backend: ModelBackend = backend or self._make_backend()
        self.system_prompt = self.backend.system_prompt  # type: ignore[attr-defined]
        log.info("백엔드 %s, 지식 베이스 %d개 청크, 용어 %d개, 도구 %d개 적재", self.backend.label, len(self.store), len(self.glossary), len(self.tools))

    def _make_backend(self) -> ModelBackend:
        if self.settings.backend == "local":
            from .backends.local import LocalBackend

            return LocalBackend(self.settings, self.tools, self.store, build_local_system_prompt(self.store.titles))
        from .backends.claude import ClaudeBackend

        return ClaudeBackend(self.settings, self.tools, build_system_prompt(self.store.titles), client=self._client)

    @property
    def model_label(self) -> str:
        """표시용 모델 이름 (claude-opus-5-5, local:qwen2.5:7b 등)."""
        return self.backend.label

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
            turn = self.backend.run(session.messages, user_text=redaction.text, on_text=on_text, on_tool=on_tool)
        except BackendError as e:
            session.messages.pop()  # 실패한 턴은 기록에 남기지 않는다
            cause = e.__cause__
            self.audit.write(AuditRecord(event="error", redactions=redaction.counts, detail=f"{type(cause).__name__ if cause else 'BackendError'}: {e}", **base))
            raise AssistantError(str(e)) from e
        session.messages.extend(turn.new_messages)
        session.turns += 1

        # 4) 종료 사유 처리
        text = turn.text
        if turn.stop_reason == "refusal":
            text = "요청하신 내용은 안전 정책에 따라 답변할 수 없습니다." + (f" (사유: {turn.refusal_detail})" if turn.refusal_detail else "") + " 질문을 바꾸어 다시 문의해 주십시오."
        elif turn.stop_reason == "max_tokens":
            text += "\n\n(응답이 최대 길이에 도달해 잘렸습니다. 이어서 답하도록 요청해 주십시오.)"
        elif turn.stop_reason == "tool_use":
            text += "\n\n(도구 호출 횟수 한도에 도달해 중단했습니다. 질문을 나눠서 다시 요청해 주십시오.)"

        result = ChatResult(
            text=text,
            blocked=False,
            classification=classification,
            redaction=redaction,
            tools_called=turn.tools_called,
            stop_reason=turn.stop_reason,
            served_by=turn.model,
            fallback_used=turn.fallback_used,
            usage=dict(turn.usage),
        )

        # 5) 감사 로그
        self.audit.write(AuditRecord(
            event="chat",
            redactions=redaction.counts,
            tools_called=turn.tools_called,
            model=self.model_label,
            served_by=turn.model,
            stop_reason=turn.stop_reason,
            input_tokens=turn.usage.get("input_tokens"),
            output_tokens=turn.usage.get("output_tokens"),
            cache_read_tokens=turn.usage.get("cache_read_input_tokens"),
            output_sha256=sha256_text(text),
            output_chars=len(text),
            **base,
        ))
        return result
