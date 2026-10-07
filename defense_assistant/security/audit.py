"""감사 로그: 모든 상호작용을 JSON Lines 형식으로 추적한다.

원문 질의는 저장하지 않고 SHA-256 해시만 남긴다(보안 검토·추적 가능성 확보,
민감정보의 2차 저장 방지). 로그 파일은 운영 환경에서 별도 보안 저장소로
전송하는 것을 권장한다.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    session_id: str
    user_id: str
    event: str  # "chat" | "blocked" | "error"
    input_sha256: str
    input_chars: int
    classification: str
    redactions: dict[str, int] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    model: str | None = None
    served_by: str | None = None
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    output_sha256: str | None = None
    output_chars: int | None = None
    detail: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """스레드 안전한 JSONL 감사 로거. `path=None`이면 메모리에만 보관한다."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.records: list[AuditRecord] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock:
            self.records.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path or not self.path.exists():
            return [r.to_dict() for r in self.records]
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
