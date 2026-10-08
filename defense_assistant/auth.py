"""아이디·비밀번호 인증.

- 비밀번호는 표준 라이브러리 scrypt로 해시한다 (외부 의존성 없음).
- 로그인 성공 시 무작위 베어러 토큰을 발급하고, DB에는 토큰의 SHA-256 해시만 저장한다.
- 로그인 실패가 반복되면 일정 시간 계정을 잠근다 (무차별 대입 방지).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .security.classification import Classification

ROLES = ("user", "admin")
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"비밀번호는 {MIN_PASSWORD_LENGTH}자 이상이어야 합니다.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, digest_hex = stored.split("$")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p), dklen=len(digest_hex) // 2)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def new_token() -> tuple[str, str]:
    """(클라이언트에 줄 토큰, DB에 저장할 해시) 를 돌려준다."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def expiry(ttl_hours: float) -> datetime:
    return utcnow() + timedelta(hours=ttl_hours)


@dataclass(frozen=True)
class Principal:
    """인증된 사용자."""

    username: str
    role: str
    clearance: Classification

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class LoginThrottle:
    """메모리 기반 로그인 실패 제한. 단일 인스턴스 기준이며 재시작 시 초기화된다."""

    def __init__(self, max_attempts: int = 5, lockout_seconds: float = 600.0):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> list[float]:
        recent = [t for t in self._failures.get(key, []) if now - t < self.lockout_seconds]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def is_locked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, time.monotonic())) >= self.max_attempts

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            self._failures.setdefault(key, []).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
