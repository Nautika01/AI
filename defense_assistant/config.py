"""환경 변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .security.classification import Classification

load_dotenv()

_VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}


class ConfigError(ValueError):
    """설정 값 오류. 메시지는 문제의 환경 변수 이름을 담은 한국어 문장이다."""


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _env_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} 값은 정수여야 합니다: {raw!r}") from None


def _env_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} 값은 숫자여야 합니다: {raw!r}") from None


@dataclass
class Settings:
    backend: str = "claude"  # "claude" | "local"
    model: str = "claude-opus-5-5"
    effort: str = "high"
    max_tokens: int = 16000
    max_iterations: int = 12
    fallbacks: str = "default"  # "default" | "off"
    max_classification: Classification = Classification.RESTRICTED
    redaction: str = "mask"  # "mask" | "off"
    docs_dir: Path = field(default_factory=lambda: Path("data/docs"))
    glossary_path: Path = field(default_factory=lambda: Path("data/glossary.json"))
    audit_log: Path = field(default_factory=lambda: Path("audit/audit.jsonl"))
    # --- 검색 ---
    synonyms_path: Path = field(default_factory=lambda: Path("data/synonyms.json"))
    index_cache_dir: Path = field(default_factory=lambda: Path("storage/index"))
    embeddings: str = "hash"  # "hash"(모델 불필요) | "openai"(Ollama 등 /v1/embeddings) | "off"(BM25만)
    embeddings_base_url: str = "http://127.0.0.1:11434/v1"
    embeddings_model: str = "nomic-embed-text"
    embeddings_api_key: str | None = None
    # --- 서버 운영 ---
    db_path: Path = field(default_factory=lambda: Path("storage/defense.db"))
    token_ttl_hours: float = 12.0
    login_max_attempts: int = 5
    login_lockout_minutes: int = 10
    cors_origins: tuple[str, ...] = ()
    admin_password: str | None = None  # 최초 기동 시 관리자 계정 자동 생성용 (사용자 0명일 때만)
    trusted_proxies: str = ""  # X-Forwarded-For 를 믿을 프록시 IP/CIDR (쉼표 구분, "*" = 전부)
    # --- 로컬 모델 (DAI_BACKEND=local) ---
    local_base_url: str = "http://127.0.0.1:11434/v1"  # Ollama 기본값. vLLM/llama-server 는 http://host:8000/v1 등
    local_model: str = "qwen2.5:7b"
    local_api_key: str | None = None
    local_tools: str = "auto"  # "auto" | "off"
    local_rag_top_k: int = 4
    local_timeout: float = 300.0
    local_temperature: float = 0.2
    local_num_ctx: int | None = None  # 모델 서버 컨텍스트 길이 (None → 4096)
    local_max_tokens: int | None = None  # 로컬 응답 최대 토큰 (None → 컨텍스트 예산으로 자동)

    def __post_init__(self) -> None:
        if self.effort not in _VALID_EFFORT:
            raise ConfigError(f"DAI_EFFORT 값은 {', '.join(sorted(_VALID_EFFORT))} 중 하나여야 합니다: {self.effort!r}")
        if self.fallbacks not in {"default", "off"}:
            raise ConfigError(f"DAI_FALLBACKS 값은 'default' 또는 'off' 여야 합니다: {self.fallbacks!r}")
        if self.redaction not in {"mask", "off"}:
            raise ConfigError(f"DAI_REDACTION 값은 'mask' 또는 'off' 여야 합니다: {self.redaction!r}")
        if self.max_tokens < 256:
            raise ConfigError(f"DAI_MAX_TOKENS 값은 256 이상이어야 합니다: {self.max_tokens}")
        if self.max_iterations < 1:
            raise ConfigError(f"DAI_MAX_ITERATIONS 값은 1 이상이어야 합니다: {self.max_iterations}")
        if self.embeddings not in {"hash", "openai", "off"}:
            raise ConfigError(f"DAI_EMBEDDINGS 값은 'hash', 'openai', 'off' 중 하나여야 합니다: {self.embeddings!r}")
        if self.backend not in {"claude", "local"}:
            raise ConfigError(f"DAI_BACKEND 값은 'claude' 또는 'local' 이어야 합니다: {self.backend!r}")
        if self.local_tools not in {"auto", "off"}:
            raise ConfigError(f"DAI_LOCAL_TOOLS 값은 'auto' 또는 'off' 여야 합니다: {self.local_tools!r}")
        if self.local_rag_top_k < 0 or self.local_timeout <= 0:
            raise ConfigError("DAI_LOCAL_RAG_TOP_K 는 0 이상, DAI_LOCAL_TIMEOUT 은 0 보다 커야 합니다")
        if self.token_ttl_hours <= 0:
            raise ConfigError("DAI_TOKEN_TTL_HOURS 값은 0 보다 커야 합니다")
        if self.login_max_attempts < 1 or self.login_lockout_minutes < 1:
            raise ConfigError("DAI_LOGIN_MAX_ATTEMPTS 와 DAI_LOGIN_LOCKOUT_MINUTES 는 1 이상이어야 합니다")
        if self.local_num_ctx is not None and self.local_num_ctx < 512:
            raise ConfigError(f"DAI_LOCAL_NUM_CTX 값은 512 이상이어야 합니다: {self.local_num_ctx}")
        if self.local_max_tokens is not None and self.local_max_tokens < 1:
            raise ConfigError(f"DAI_LOCAL_MAX_TOKENS 값은 1 이상이어야 합니다: {self.local_max_tokens}")

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        """환경 변수에서 설정을 읽는다. 상대 경로는 `root` 기준으로 해석한다."""
        root = root or Path.cwd()

        def _path(name: str, default: str) -> Path:
            p = Path(os.environ.get(name, default))
            return p if p.is_absolute() else root / p

        try:
            max_classification = Classification.parse(os.environ.get("DAI_MAX_CLASSIFICATION", "RESTRICTED"))
        except ValueError as e:
            raise ConfigError(f"DAI_MAX_CLASSIFICATION 값이 올바르지 않습니다: {e}") from None

        return cls(
            backend=os.environ.get("DAI_BACKEND", "claude").strip().lower(),
            model=os.environ.get("DAI_MODEL", "claude-opus-5-5"),
            effort=_env_str("DAI_EFFORT", "high"),
            max_tokens=_env_int("DAI_MAX_TOKENS", "16000"),
            max_iterations=_env_int("DAI_MAX_ITERATIONS", "12"),
            fallbacks=_env_str("DAI_FALLBACKS", "default"),
            max_classification=max_classification,
            redaction=_env_str("DAI_REDACTION", "mask"),
            docs_dir=_path("DAI_DOCS_DIR", "data/docs"),
            glossary_path=_path("DAI_GLOSSARY_PATH", "data/glossary.json"),
            audit_log=_path("DAI_AUDIT_LOG", "audit/audit.jsonl"),
            synonyms_path=_path("DAI_SYNONYMS_PATH", "data/synonyms.json"),
            index_cache_dir=_path("DAI_INDEX_CACHE_DIR", "storage/index"),
            embeddings=os.environ.get("DAI_EMBEDDINGS", "hash").strip().lower(),
            embeddings_base_url=os.environ.get("DAI_EMBEDDINGS_BASE_URL", os.environ.get("DAI_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")),
            embeddings_model=os.environ.get("DAI_EMBEDDINGS_MODEL", "nomic-embed-text"),
            embeddings_api_key=os.environ.get("DAI_EMBEDDINGS_API_KEY") or os.environ.get("DAI_LOCAL_API_KEY") or None,
            db_path=_path("DAI_DB_PATH", "storage/defense.db"),
            token_ttl_hours=_env_float("DAI_TOKEN_TTL_HOURS", "12"),
            login_max_attempts=_env_int("DAI_LOGIN_MAX_ATTEMPTS", "5"),
            login_lockout_minutes=_env_int("DAI_LOGIN_LOCKOUT_MINUTES", "10"),
            cors_origins=tuple(o.strip() for o in os.environ.get("DAI_CORS_ORIGINS", "").split(",") if o.strip()),
            admin_password=os.environ.get("DAI_ADMIN_PASSWORD") or None,
            local_base_url=os.environ.get("DAI_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1"),
            local_model=os.environ.get("DAI_LOCAL_MODEL", "qwen2.5:7b"),
            local_api_key=os.environ.get("DAI_LOCAL_API_KEY") or None,
            local_tools=os.environ.get("DAI_LOCAL_TOOLS", "auto").strip().lower(),
            local_rag_top_k=_env_int("DAI_LOCAL_RAG_TOP_K", "4"),
            local_timeout=_env_float("DAI_LOCAL_TIMEOUT", "300"),
            local_temperature=_env_float("DAI_LOCAL_TEMPERATURE", "0.2"),
            local_num_ctx=_env_int("DAI_LOCAL_NUM_CTX", "0") or None,
            local_max_tokens=_env_int("DAI_LOCAL_MAX_TOKENS", "0") or None,
            trusted_proxies=os.environ.get("DAI_TRUSTED_PROXIES", ""),
        )


def build_store(settings: Settings):
    """설정에 맞는 DocumentStore 를 만든다 (동의어·임베딩·캐시 포함)."""
    from .knowledge import DocumentStore, OpenAIEmbedder, SynonymMap

    embedder: object = None
    if settings.embeddings == "off":
        embedder = False
    elif settings.embeddings == "openai":
        embedder = OpenAIEmbedder(settings.embeddings_base_url, settings.embeddings_model, settings.embeddings_api_key)
    return DocumentStore.from_directory(
        settings.docs_dir,
        synonyms=SynonymMap.load(settings.synonyms_path),
        embedder=embedder,  # type: ignore[arg-type]
        cache_dir=settings.index_cache_dir,
    )
