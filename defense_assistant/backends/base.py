"""모델 백엔드 공통 인터페이스.

비서 본체는 메시지 기록(Anthropic 메시지 형식의 dict 목록)을 넘기고 `TurnResult` 를 받는다.
어떤 백엔드든 기록에 추가하는 메시지는 JSON 직렬화 가능한 dict 여야 한다(세션 저장 때문).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# TurnResult.usage 의 키. 백엔드가 보고하지 않는 값은 None.
# cache_read 는 캐시에서 읽은 입력(약 0.1배 단가), cache_creation 은 캐시에 새로 쓴 입력(약 1.25배 단가). 둘 다 input_tokens 에 포함되지 않는다.
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

OnText = Callable[[str], None]
OnTool = Callable[[str, dict[str, Any]], None]


class BackendError(RuntimeError):
    """모델 호출 실패. 메시지는 사용자에게 그대로 보여 줄 수 있는 한국어 문장이다."""


@dataclass
class TurnResult:
    text: str
    new_messages: list[dict[str, Any]]
    tools_called: list[str] = field(default_factory=list)
    stop_reason: str | None = None  # end_turn | max_tokens | tool_use | refusal
    model: str | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    fallback_used: bool = False
    refusal_detail: str | None = None


class ModelBackend(Protocol):
    name: str
    label: str

    def run(self, messages: list[dict[str, Any]], *, user_text: str, on_text: OnText | None, on_tool: OnTool | None) -> TurnResult: ...


def to_jsonable(obj: Any) -> Any:
    """SDK pydantic 모델을 포함한 메시지 구조를 JSON 직렬화 가능한 dict/list 로 변환한다."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj
