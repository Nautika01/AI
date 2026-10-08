"""SQLite 저장소: 사용자, 토큰, 대화 세션.

단일 파일 DB로 외부 서비스 없이 동작한다. 다중 인스턴스 운영 시에는 같은 인터페이스로
PostgreSQL 등으로 교체할 수 있도록 SQL을 이 모듈에만 둔다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import auth
from .assistant import Session
from .security.classification import Classification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    clearance     INTEGER NOT NULL DEFAULT 1,
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash  TEXT PRIMARY KEY,
    username    TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(username);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    username    TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '',
    messages    TEXT NOT NULL DEFAULT '[]',
    turns       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username, updated_at DESC);
"""


@dataclass(frozen=True)
class UserRecord:
    username: str
    role: str
    clearance: Classification
    disabled: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"username": self.username, "role": self.role, "clearance": self.clearance.name, "disabled": self.disabled, "created_at": self.created_at}


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    username: str
    title: str
    turns: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if str(self.path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- 사용자 ---------------------------------------------------------
    def create_user(self, username: str, password: str, *, role: str = "user", clearance: Classification = Classification.RESTRICTED) -> UserRecord:
        username = username.strip()
        if not username or len(username) > 64 or not username.replace("_", "").replace(".", "").replace("-", "").isalnum():
            raise ValueError("아이디는 영문·숫자·._- 로 1~64자여야 합니다.")
        if role not in auth.ROLES:
            raise ValueError(f"역할은 {auth.ROLES} 중 하나여야 합니다.")
        pw_hash = auth.hash_password(password)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users(username, password_hash, role, clearance, created_at) VALUES (?,?,?,?,?)",
                    (username, pw_hash, role, int(clearance), _now()),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"이미 존재하는 아이디입니다: {username}") from e
        return self.get_user(username)  # type: ignore[return-value]

    def get_user(self, username: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return self._user(row) if row else None

    def list_users(self) -> list[UserRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY username").fetchall()
        return [self._user(r) for r in rows]

    def count_users(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def delete_user(self, username: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
        return cur.rowcount > 0

    def set_password(self, username: str, password: str) -> bool:
        pw_hash = auth.hash_password(password)
        with self._lock:
            cur = self._conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (pw_hash, username))
            self._conn.execute("DELETE FROM tokens WHERE username = ?", (username,))  # 비밀번호 변경 시 기존 로그인 무효화
        return cur.rowcount > 0

    def update_user(self, username: str, *, role: str | None = None, clearance: Classification | None = None, disabled: bool | None = None) -> bool:
        sets, params = [], []
        if role is not None:
            if role not in auth.ROLES:
                raise ValueError(f"역할은 {auth.ROLES} 중 하나여야 합니다.")
            sets.append("role = ?"); params.append(role)
        if clearance is not None:
            sets.append("clearance = ?"); params.append(int(clearance))
        if disabled is not None:
            sets.append("disabled = ?"); params.append(int(disabled))
        if not sets:
            return self.get_user(username) is not None
        params.append(username)
        with self._lock:
            cur = self._conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE username = ?", params)
            if disabled:
                self._conn.execute("DELETE FROM tokens WHERE username = ?", (username,))
        return cur.rowcount > 0

    def verify_credentials(self, username: str, password: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            auth.verify_password(password, "scrypt$16384$8$1$00$00")  # 타이밍 차이 완화
            return None
        if row["disabled"] or not auth.verify_password(password, row["password_hash"]):
            return None
        return self._user(row)

    @staticmethod
    def _user(row: sqlite3.Row) -> UserRecord:
        return UserRecord(username=row["username"], role=row["role"], clearance=Classification(row["clearance"]), disabled=bool(row["disabled"]), created_at=row["created_at"])

    # ---- 토큰 ----------------------------------------------------------
    def issue_token(self, username: str, ttl_hours: float) -> tuple[str, datetime]:
        token, token_hash = auth.new_token()
        exp = auth.expiry(ttl_hours)
        with self._lock:
            self._conn.execute("INSERT INTO tokens(token_hash, username, created_at, expires_at) VALUES (?,?,?,?)", (token_hash, username, _now(), exp.isoformat()))
        return token, exp

    def resolve_token(self, token: str) -> auth.Principal | None:
        token_hash = auth.hash_token(token)
        with self._lock:
            row = self._conn.execute(
                "SELECT t.expires_at, u.* FROM tokens t JOIN users u ON u.username = t.username WHERE t.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= auth.utcnow() or row["disabled"]:
                self._conn.execute("DELETE FROM tokens WHERE token_hash = ?", (token_hash,))
                return None
        return auth.Principal(username=row["username"], role=row["role"], clearance=Classification(row["clearance"]))

    def revoke_token(self, token: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tokens WHERE token_hash = ?", (auth.hash_token(token),))
        return cur.rowcount > 0

    def purge_expired_tokens(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tokens WHERE expires_at <= ?", (auth.utcnow().isoformat(),))
        return cur.rowcount

    # ---- 세션 ----------------------------------------------------------
    def save_session(self, session: Session, *, title: str | None = None, create: bool = True) -> bool:
        """세션을 저장하고 저장 여부를 돌려준다.

        `create=False` 이면 이미 있는 행만 갱신한다. 턴 처리 중에 삭제된 대화가 턴 종료 시
        다시 생기지 않게 하려면 이 방식을 쓴다 (행이 없으면 False).
        """
        payload = json.dumps(session.messages, ensure_ascii=False)
        if title is None:
            title = _derive_title(session.messages)
        with self._lock:
            if not create:
                cur = self._conn.execute(
                    "UPDATE sessions SET title = ?, messages = ?, turns = ?, updated_at = ? WHERE session_id = ? AND username = ?",
                    (title, payload, session.turns, _now(), session.session_id, session.user_id),
                )
                return cur.rowcount > 0
            self._conn.execute(
                """INSERT INTO sessions(session_id, username, title, messages, turns, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET title = excluded.title, messages = excluded.messages,
                       turns = excluded.turns, updated_at = excluded.updated_at""",
                (session.session_id, session.user_id, title, payload, session.turns, session.created_at.isoformat(), _now()),
            )
        return True

    def load_session(self, session_id: str, username: str | None = None) -> Session | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None or (username is not None and row["username"] != username):
            return None
        user = self.get_user(row["username"])
        return Session(
            session_id=row["session_id"],
            user_id=row["username"],
            messages=json.loads(row["messages"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            turns=row["turns"],
            max_classification=user.clearance if user else None,
        )

    def list_sessions(self, username: str, limit: int = 50) -> list[SessionSummary]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, username, title, turns, created_at, updated_at FROM sessions WHERE username = ? ORDER BY updated_at DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        return [SessionSummary(**dict(r)) for r in rows]

    def delete_session(self, session_id: str, username: str | None = None) -> bool:
        with self._lock:
            if username is None:
                cur = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            else:
                cur = self._conn.execute("DELETE FROM sessions WHERE session_id = ? AND username = ?", (session_id, username))
        return cur.rowcount > 0


def _derive_title(messages: list[dict[str, Any]], max_len: int = 40) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            text = " ".join(m["content"].split())
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return ""


NO_RESPONSE_TEXT = "(표시할 응답이 없습니다. 안전 정책에 따라 답변이 거부되었을 수 있습니다.)"


def transcript(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """저장된 메시지에서 사람이 읽을 사용자/비서 텍스트만 추린다 (도구 호출·사고 블록 제외).

    사용자 질문 하나에 대한 비서 응답(도구 호출 전후의 여러 assistant 메시지)은 실시간 화면처럼
    말풍선 하나로 합친다. 텍스트 없이 끝난 턴(출력 전 거부 등)은 안내 문구로 표시한다.
    """
    out: list[dict[str, str]] = []
    pending: list[str] | None = None  # 진행 중인 턴의 비서 텍스트 조각

    def flush() -> None:
        if pending is not None:
            text = "".join(pending)
            out.append({"role": "assistant", "text": text if text.strip() else NO_RESPONSE_TEXT})

    for m in messages:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, str):
            flush()
            out.append({"role": "user", "text": content})
            pending = []
        elif m.get("role") == "assistant" and isinstance(content, list):
            text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            if pending is None:
                pending = []
            pending.append(text)
    flush()
    return out
