"""교차 검토 후 모듈 경계를 넘는 후속 수정의 회귀 테스트."""

import json

import pytest

from defense_assistant.config import ConfigError, Settings
from defense_assistant.security import AuditLogger, AuditRecord
from tests.conftest import ROOT, make_message


def _rec(**kw):
    base = dict(session_id="s", user_id="u", event="chat", input_sha256="h", input_chars=1, classification="UNCLASSIFIED")
    base.update(kw)
    return AuditRecord(**base)


def test_audit_memory_record_only_after_file_write(tmp_path):
    log = AuditLogger(tmp_path / "a.jsonl")
    log.path = tmp_path / "missing_dir" / "a.jsonl"  # 쓰기 실패 유도
    with pytest.raises(OSError):
        log.write(_rec())
    assert log.records == []  # 파일에 못 쓴 기록은 메모리에도 남지 않는다


def test_local_settings_fields_and_korean_errors(monkeypatch):
    s = Settings(local_num_ctx=8192, local_max_tokens=1024)
    assert s.local_num_ctx == 8192 and s.local_max_tokens == 1024
    with pytest.raises(ConfigError, match="DAI_LOCAL_NUM_CTX"):
        Settings(local_num_ctx=100)
    with pytest.raises(ConfigError, match="DAI_LOCAL_MAX_TOKENS"):
        Settings(local_max_tokens=0)
    monkeypatch.setenv("DAI_LOCAL_NUM_CTX", "16384")
    monkeypatch.setenv("DAI_TRUSTED_PROXIES", "10.0.0.0/8")
    env = Settings.from_env(ROOT)
    assert env.local_num_ctx == 16384 and env.local_max_tokens is None and env.trusted_proxies == "10.0.0.0/8"


def test_local_backend_uses_settings_num_ctx(store, glossary):
    from defense_assistant.backends.local import LocalBackend
    from defense_assistant.tools import build_tools

    b = LocalBackend(Settings(backend="local", local_num_ctx=16384), build_tools(store, glossary), store, "sys")
    assert b.num_ctx == 16384 and b.input_budget < 16384


def test_cli_reports_config_error_without_traceback(monkeypatch, capsys):
    from defense_assistant.cli import main

    monkeypatch.setenv("DAI_EFFORT", "turbo")
    assert main(["--root", str(ROOT), "search", "지혈대"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("설정 오류:") and "DAI_EFFORT" in err and "Traceback" not in err


def test_cli_serve_rejects_short_bootstrap_password(monkeypatch, tmp_path, capsys):
    pytest.importorskip("uvicorn")
    from defense_assistant.cli import main

    monkeypatch.setenv("DAI_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("DAI_ADMIN_PASSWORD", "short")
    monkeypatch.setenv("DAI_INDEX_CACHE_DIR", str(tmp_path / "idx"))
    assert main(["--root", str(ROOT), "serve"]) == 1
    assert "8자 이상" in capsys.readouterr().err


def _run_cli_chat(monkeypatch, capsys, assistant, line):
    import defense_assistant.cli as cli

    monkeypatch.setattr(cli, "DefenseAssistant", lambda settings: assistant)
    inputs = iter([line, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    assert cli.main(["--root", str(ROOT), "chat"]) == 0
    return capsys.readouterr().out


def test_cli_stream_shows_truncation_notice(monkeypatch, capsys, make_assistant):
    assistant, _ = make_assistant([make_message("긴 답변", stop_reason="max_tokens")])
    out = _run_cli_chat(monkeypatch, capsys, assistant, "질문")
    assert "◀ 긴 답변" in out and "잘렸습니다" in out


def test_cli_stream_shows_refusal_notice(monkeypatch, capsys, make_assistant):
    assistant, _ = make_assistant([make_message("부분", stop_reason="refusal")])
    out = _run_cli_chat(monkeypatch, capsys, assistant, "질문")
    assert "안전 정책" in out


def test_local_usage_summed_over_tool_iterations(store, glossary):
    pytest.importorskip("fastapi")
    from defense_assistant.assistant import DefenseAssistant
    from defense_assistant.backends.local import LocalBackend
    from defense_assistant.prompts import build_local_system_prompt
    from defense_assistant.tools import build_tools
    from tests.fake_openai_server import create_fake_server, run_in_thread

    base_url, stop = run_in_thread(create_fake_server())
    try:
        settings = Settings(backend="local", local_model="fake-model", local_base_url=base_url, local_timeout=20)
        backend = LocalBackend(settings, build_tools(store, glossary), store, build_local_system_prompt(store.titles))
        a = DefenseAssistant(settings, store=store, glossary=glossary, audit=AuditLogger(None), backend=backend)
        r = a.chat(a.new_session("kim"), "시간: 071430IOCT26")  # 도구 호출 → 요청 2번
        assert r.usage["input_tokens"] == 84 and r.usage["output_tokens"] == 14
    finally:
        stop()


def test_pyproject_dev_has_httpx_for_testclient():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"httpx"' in text
    json.dumps({})  # 형식 유지
