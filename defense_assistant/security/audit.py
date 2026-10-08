"""감사 로그: 모든 상호작용을 JSON Lines 형식으로 추적한다.

원문 질의는 저장하지 않고 SHA-256 해시만 남긴다(보안 검토·추적 가능성 확보,
민감정보의 2차 저장 방지). 로그 파일은 운영 환경에서 별도 보안 저장소로
전송하는 것을 권장한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    session_id: str
    user_id: str
    event: str  # "chat" | "blocked" | "error" | "login_failed" | "login_locked"
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
    cache_creation_tokens: int | None = None
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
        data = (json.dumps(record.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if self.path:
                with self.path.open("a+b") as f:
                    # 이전 기록이 개행 없이 잘려 있으면(쓰기 중 비정상 종료) 줄을 먼저 닫아 새 기록이 붙지 않게 한다
                    f.seek(0, os.SEEK_END)
                    if f.tell() > 0:
                        f.seek(-1, os.SEEK_END)
                        if f.read(1) != b"\n":
                            data = b"\n" + data
                    f.write(data)  # 한 번에 써서 다른 기록과 섞이지 않게 한다
            # 파일 쓰기가 성공한 뒤에만 메모리에도 남겨, 실패 시 두 기록이 어긋나지 않게 한다
            self.records.append(record)

    def read_all(self) -> list[dict[str, Any]]:
        """기록 전체를 읽는다. 손상된 줄은 건너뛰지 않고 `event="corrupt"` 항목으로 표시한다."""
        if not self.path or not self.path.exists():
            return [r.to_dict() for r in self.records]
        out: list[dict[str, Any]] = []
        # utf-8-sig: 편집기가 붙인 BOM 허용, errors="replace": 멀티바이트 중간에서 잘린 줄 허용
        with self.path.open(encoding="utf-8-sig", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = None
                if not isinstance(item, dict):
                    log.warning("감사 로그 %s 의 %d번째 줄이 손상되어 해석하지 못했습니다.", self.path, line_no)
                    item = {"event": "corrupt", "line_no": line_no, "raw": line.rstrip("\n")[:200]}
                out.append(item)
        return out
