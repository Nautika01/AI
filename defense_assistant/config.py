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

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        """환경 변수에서 설정을 읽는다. 상대 경로는 `root` 기준으로 해석한다."""
        root = root or Path.cwd()

        def _path(name: str, default: str) -> Path:
            p = Path(os.environ.get(name, default))
            return p if p.is_absolute() else root / p

        return cls(
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
        )
