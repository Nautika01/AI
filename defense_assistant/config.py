"""환경 변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .security.classification import Classification

load_dotenv()

_VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}


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
    # --- 로컬 모델 (DAI_BACKEND=local) ---
    local_base_url: str = "http://127.0.0.1:11434/v1"  # Ollama 기본값. vLLM/llama-server 는 http://host:8000/v1 등
    local_model: str = "qwen2.5:7b"
    local_api_key: str | None = None
    local_tools: str = "auto"  # "auto" | "off"
    local_rag_top_k: int = 4
    local_timeout: float = 300.0
    local_temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.effort not in _VALID_EFFORT:
            raise ValueError(f"DAI_EFFORT must be one of {sorted(_VALID_EFFORT)}, got {self.effort!r}")
        if self.fallbacks not in {"default", "off"}:
            raise ValueError("DAI_FALLBACKS must be 'default' or 'off'")
        if self.redaction not in {"mask", "off"}:
            raise ValueError("DAI_REDACTION must be 'mask' or 'off'")
        if self.max_tokens < 256:
            raise ValueError("DAI_MAX_TOKENS must be >= 256")
        if self.max_iterations < 1:
            raise ValueError("DAI_MAX_ITERATIONS must be >= 1")
        if self.embeddings not in {"hash", "openai", "off"}:
            raise ValueError("DAI_EMBEDDINGS must be 'hash', 'openai' or 'off'")
        if self.backend not in {"claude", "local"}:
            raise ValueError("DAI_BACKEND must be 'claude' or 'local'")
        if self.local_tools not in {"auto", "off"}:
            raise ValueError("DAI_LOCAL_TOOLS must be 'auto' or 'off'")
        if self.local_rag_top_k < 0 or self.local_timeout <= 0:
            raise ValueError("DAI_LOCAL_RAG_TOP_K must be >= 0 and DAI_LOCAL_TIMEOUT > 0")
        if self.token_ttl_hours <= 0:
            raise ValueError("DAI_TOKEN_TTL_HOURS must be > 0")
        if self.login_max_attempts < 1 or self.login_lockout_minutes < 1:
            raise ValueError("DAI_LOGIN_MAX_ATTEMPTS and DAI_LOGIN_LOCKOUT_MINUTES must be >= 1")

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        """환경 변수에서 설정을 읽는다. 상대 경로는 `root` 기준으로 해석한다."""
        root = root or Path.cwd()

        def _path(name: str, default: str) -> Path:
            p = Path(os.environ.get(name, default))
            return p if p.is_absolute() else root / p

        return cls(
            backend=os.environ.get("DAI_BACKEND", "claude").strip().lower(),
            model=os.environ.get("DAI_MODEL", "claude-opus-5-5"),
            effort=os.environ.get("DAI_EFFORT", "high"),
            max_tokens=int(os.environ.get("DAI_MAX_TOKENS", "16000")),
            max_iterations=int(os.environ.get("DAI_MAX_ITERATIONS", "12")),
            fallbacks=os.environ.get("DAI_FALLBACKS", "default"),
            max_classification=Classification.parse(os.environ.get("DAI_MAX_CLASSIFICATION", "RESTRICTED")),
            redaction=os.environ.get("DAI_REDACTION", "mask"),
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
            token_ttl_hours=float(os.environ.get("DAI_TOKEN_TTL_HOURS", "12")),
            login_max_attempts=int(os.environ.get("DAI_LOGIN_MAX_ATTEMPTS", "5")),
            login_lockout_minutes=int(os.environ.get("DAI_LOGIN_LOCKOUT_MINUTES", "10")),
            cors_origins=tuple(o.strip() for o in os.environ.get("DAI_CORS_ORIGINS", "").split(",") if o.strip()),
            admin_password=os.environ.get("DAI_ADMIN_PASSWORD") or None,
            local_base_url=os.environ.get("DAI_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1"),
            local_model=os.environ.get("DAI_LOCAL_MODEL", "qwen2.5:7b"),
            local_api_key=os.environ.get("DAI_LOCAL_API_KEY") or None,
            local_tools=os.environ.get("DAI_LOCAL_TOOLS", "auto").strip().lower(),
            local_rag_top_k=int(os.environ.get("DAI_LOCAL_RAG_TOP_K", "4")),
            local_timeout=float(os.environ.get("DAI_LOCAL_TIMEOUT", "300")),
            local_temperature=float(os.environ.get("DAI_LOCAL_TEMPERATURE", "0.2")),
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
