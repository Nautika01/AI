from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from defense_assistant.assistant import DefenseAssistant
from defense_assistant.config import Settings
from defense_assistant.knowledge import DocumentStore
from defense_assistant.security import AuditLogger
from defense_assistant.tools.glossary import load_glossary

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def store(tmp_path_factory) -> DocumentStore:
    """실제 실행과 같은 구성(동의어 사전 + 해시 임베딩 + 캐시)의 지식 베이스."""
    from defense_assistant.config import build_store

    settings = Settings(docs_dir=ROOT / "data" / "docs", synonyms_path=ROOT / "data" / "synonyms.json", index_cache_dir=tmp_path_factory.mktemp("index"))
    return build_store(settings)


@pytest.fixture(scope="session")
def glossary() -> dict:
    return load_glossary(ROOT / "data" / "glossary.json")


def make_message(text: str, *, stop_reason: str = "end_turn", tool_uses: list[tuple[str, dict]] | None = None, model: str = "claude-opus-5-5", fallback: bool = False):
    content = [SimpleNamespace(type="text", text=text)] if text else []
    for i, (name, inp) in enumerate(tool_uses or []):
        content.append(SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=inp))
    iterations = [SimpleNamespace(type="fallback_message")] if fallback else None
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber", explanation="테스트") if stop_reason == "refusal" else None,
        model=model,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, iterations=iterations),
    )


class FakeStream:
    def __init__(self, message):
        self._message = message

    def __iter__(self):
        for block in self._message.content:
            if block.type == "text":
                for ch in block.text:
                    yield SimpleNamespace(type="text", text=ch)

    def get_final_message(self):
        return self._message


class FakeRunner:
    """미리 준비한 메시지 목록을 차례로 돌려주는 가짜 툴 러너. 도구는 실제로 실행한다."""

    def __init__(self, scripted, params):
        self.scripted = list(scripted)
        self.params = params
        self._pending = None
        self._tools = {t.name: t for t in params["tools"]}

    def __iter__(self):
        for msg in self.scripted:
            self._pending = msg
            yield FakeStream(msg)

    def generate_tool_call_response(self):
        msg = self._pending
        results = []
        for block in msg.content:
            if block.type == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": self._tools[block.name].call(block.input)})
        return {"role": "user", "content": results} if results else None


class FakeClient:
    def __init__(self, scripted):
        self.scripted = scripted
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **params):
        self.calls.append(params)
        return FakeRunner(self.scripted, params)


@pytest.fixture
def make_assistant(store, glossary):
    def _make(scripted=None, **overrides) -> tuple[DefenseAssistant, FakeClient]:
        settings = Settings(docs_dir=ROOT / "data" / "docs", glossary_path=ROOT / "data" / "glossary.json", **overrides)
        client = FakeClient(scripted or [make_message("응답")])
        assistant = DefenseAssistant(settings, client=client, store=store, glossary=glossary, audit=AuditLogger(None))
        return assistant, client

    return _make
