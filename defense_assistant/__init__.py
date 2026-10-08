"""국방 특화 생성형 AI 비서 (Defense-domain generative AI assistant)."""

from .assistant import AssistantError, ChatResult, DefenseAssistant, Session
from .config import Settings

__all__ = ["AssistantError", "ChatResult", "DefenseAssistant", "Session", "Settings"]
__version__ = "0.3.0"
