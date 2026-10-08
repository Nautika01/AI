"""교차 검토에서 확인된 서버·저장소·감사 로그·웹 UI 결함의 회귀 테스트."""

import json
import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from defense_assistant.assistant import AssistantError, Session  # noqa: E402
from defense_assistant.security import AuditLogger, AuditRecord, Classification  # noqa: E402
from defense_assistant.server import create_app  # noqa: E402
from defense_assistant.storage import NO_RESPONSE_TEXT, Database, transcript  # noqa: E402
from tests.conftest import make_message  # noqa: E402

INDEX_HTML = Path(__file__).resolve().parent.parent / "defense_assistant" / "static" / "index.html"


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    monkeypatch.delenv("DAI_TRUSTED_PROXIES", raising=False)


@pytest.fixture
def db():
    d = Database(":memory:")
    d.create_user("admin", "adminpass1", role="admin", clearance=Classification.RESTRICTED)
    d.create_user("kim", "kimpass123", clearance=Classification.UNCLASSIFIED)
    yield d
    d.close()


@pytest.fixture
def assistant(make_assistant):
    a, _ = make_assistant([make_message("서버 응답")])
    return a


def login(client, username, password, **kw):
    r = client.post("/auth/login", json={"username": username, "password": password}, **kw)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def sse_events(raw: str) -> list[tuple[str, object]]:
    out = []
    for e in raw.split("\n\n"):
        if not e.strip():
            continue
        kind = e.split("\n")[0].removeprefix("event: ")
        out.append((kind, json.loads(e.split("\ndata: ", 1)[1])))
    return out


# ---- 로그인 잠금: 신뢰 프록시 뒤 실제 클라이언트 기준 + 감사 기록 ---------------------
def test_lockout_keyed_by_forwarded_client_behind_trusted_proxy(assistant, db):
    c = TestClient(create_app(assistant, db, trusted_proxies="172.17.0.0/16"), client=("172.17.0.1", 40000))
    for i in range(5):
        r = c.post("/auth/login", json={"username": "kim", "password": "bad"}, headers={"X-Forwarded-For": "203.0.113.9"})
        assert r.status_code == 401
    # 같은 공격자는 잠기지만, 프록시를 거친 진짜 사용자는 영향을 받지 않는다
    assert c.post("/auth/login", json={"username": "kim", "password": "kimpass123"}, headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 429
    login(c, "kim", "kimpass123", headers={"X-Forwarded-For": "10.0.0.50"})
    events = [r.event for r in assistant.audit.records]
    assert events.count("login_failed") == 5 and events.count("login_locked") == 1
    rec = next(r for r in assistant.audit.records if r.event == "login_locked")
    assert rec.user_id == "kim" and rec.detail == "client=203.0.113.9"


def test_forwarded_for_spoofing_does_not_bypass_lockout(assistant, db):
    # 공격자가 X-Forwarded-For 왼쪽 값을 바꿔도 프록시가 덧붙인 실제 주소(가장 오른쪽)로 판단한다
    c = TestClient(create_app(assistant, db, trusted_proxies=["172.17.0.1"]), client=("172.17.0.1", 40000))
    for i in range(5):
        c.post("/auth/login", json={"username": "kim", "password": "bad"}, headers={"X-Forwarded-For": f"1.2.3.{i}, 203.0.113.9"})
    r = c.post("/auth/login", json={"username": "kim", "password": "kimpass123"}, headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9"})
    assert r.status_code == 429


def test_forwarded_for_ignored_without_trusted_proxy(assistant, db):
    c = TestClient(create_app(assistant, db))
    for i in range(5):
        c.post("/auth/login", json={"username": "kim", "password": "bad"}, headers={"X-Forwarded-For": f"203.0.113.{i}"})
    assert c.post("/auth/login", json={"username": "kim", "password": "kimpass123"}, headers={"X-Forwarded-For": "10.0.0.50"}).status_code == 429


def test_trusted_proxies_from_env(assistant, db, monkeypatch):
    monkeypatch.setenv("DAI_TRUSTED_PROXIES", "잘못된값, 172.17.0.0/16")
    c = TestClient(create_app(assistant, db), client=("172.17.0.1", 40000))
    for _ in range(5):
        c.post("/auth/login", json={"username": "kim", "password": "bad"}, headers={"X-Forwarded-For": "203.0.113.9"})
    login(c, "kim", "kimpass123", headers={"X-Forwarded-For": "10.0.0.50"})


# ---- 공백만 있는 메시지 ------------------------------------------------------------
@pytest.mark.parametrize("message", ["   ", "\n\n", "　"])
def test_blank_message_rejected_with_422(assistant, db, message):
    c = TestClient(create_app(assistant, db))
    h = login(c, "admin", "adminpass1")
    r = c.post("/chat", json={"message": message}, headers=h)
    assert r.status_code == 422 and "입력이 비어 있습니다" in r.text
    assert c.post("/chat/stream", json={"message": message}, headers=h).status_code == 422
    assert db.list_sessions("admin") == []


# ---- 응답 중 삭제한 대화가 되살아나지 않음 -------------------------------------------
def test_delete_during_turn_is_rejected_and_not_resurrected(assistant, db):
    c = TestClient(create_app(assistant, db))
    h = login(c, "admin", "adminpass1")
    sid = c.post("/sessions", headers=h).json()["session_id"]
    started, release = threading.Event(), threading.Event()
    real_chat = assistant.chat

    def slow_chat(*a, **kw):
        started.set()
        assert release.wait(5)
        return real_chat(*a, **kw)

    assistant.chat = slow_chat
    result = {}
    t = threading.Thread(target=lambda: result.update(r=c.post("/chat", json={"message": "첫 질문", "session_id": sid}, headers=h)))
    t.start()
    assert started.wait(5)
    r = c.delete(f"/sessions/{sid}", headers=h)
    release.set()
    t.join(5)
    assert r.status_code == 409 and "삭제할 수 없습니다" in r.json()["detail"]
    assert result["r"].status_code == 200
    # 턴이 끝난 뒤에는 정상 삭제되고 다시 생기지 않는다
    assert c.delete(f"/sessions/{sid}", headers=h).json() == {"deleted": True}
    assert db.list_sessions("admin") == []


def test_session_deleted_mid_turn_is_not_reinserted(assistant, db):
    # 세션을 읽은 직후 다른 경로로 삭제되어도 턴 종료 저장이 행을 다시 만들지 않는다
    c = TestClient(create_app(assistant, db))
    h = login(c, "admin", "adminpass1")
    sid = c.post("/chat", json={"message": "첫 질문"}, headers=h).json()["session_id"]
    real_chat = assistant.chat

    def chat_then_deleted(*a, **kw):
        out = real_chat(*a, **kw)
        db.delete_session(sid)
        return out

    assistant.chat = chat_then_deleted
    assert c.post("/chat", json={"message": "두 번째", "session_id": sid}, headers=h).status_code == 200
    assert db.list_sessions("admin") == [] and db.load_session(sid) is None


def test_save_session_update_only(db):
    s = Session(user_id="kim", messages=[{"role": "user", "content": "질문"}], turns=1)
    assert db.save_session(s, create=False) is False and db.load_session(s.session_id) is None
    assert db.save_session(s) is True
    s.turns = 2
    assert db.save_session(s, create=False) is True and db.load_session(s.session_id).turns == 2


# ---- 감사 로그 손상 허용 ----------------------------------------------------------
def _record(event="chat"):
    return AuditRecord(session_id="s", user_id="admin", event=event, input_sha256="x", input_chars=1, classification="UNCLASSIFIED")


def test_audit_read_all_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    good = json.dumps(_record().to_dict(), ensure_ascii=False)
    path.write_bytes(b"\xef\xbb\xbf" + good.encode() + b"\n" + good.encode()[:-5] + "\n".encode() + '{"detail": "한'.encode()[:-1] + b"\n" + good.encode() + b"\n")
    records = AuditLogger(path).read_all()
    assert [r["event"] for r in records] == ["chat", "corrupt", "corrupt", "chat"]
    assert records[1]["line_no"] == 2 and records[1]["raw"]


def test_audit_write_closes_truncated_last_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"session_id": "abc", "user_id": "ad', encoding="utf-8")  # 개행 없이 잘린 기록
    logger = AuditLogger(path)
    logger.write(_record("blocked"))
    logger.write(_record("chat"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[1])["event"] == "blocked"
    assert [r["event"] for r in logger.read_all()] == ["corrupt", "blocked", "chat"]


def test_admin_audit_endpoint_survives_corrupt_log(make_assistant, db, tmp_path):
    path = tmp_path / "audit.jsonl"
    a, _ = make_assistant([make_message("응답")])
    a.audit = AuditLogger(path)
    c = TestClient(create_app(a, db))
    h = login(c, "admin", "adminpass1")
    c.post("/chat", json={"message": "안녕"}, headers=h)
    with path.open("ab") as f:
        f.write(b'{"session_id": "trunc')
    c.post("/chat", json={"message": "다시"}, headers=h)
    r = c.get("/admin/audit", headers=h)
    assert r.status_code == 200
    assert [x["event"] for x in r.json()["records"]] == ["chat", "corrupt", "chat"]


# ---- 백엔드 오류 원문 노출 ---------------------------------------------------------
def test_backend_error_detail_hidden_from_regular_users(assistant, db):
    secret = "로컬 모델 서버에 연결할 수 없습니다: http://10.0.0.5:11434/v1. DAI_LOCAL_BASE_URL 이 맞는지 확인하십시오."

    def boom(*a, **kw):
        raise AssistantError(secret)

    assistant.chat = boom
    c = TestClient(create_app(assistant, db))
    hk = login(c, "kim", "kimpass123")
    r = c.post("/chat", json={"message": "안녕"}, headers=hk)
    assert r.status_code == 502 and "10.0.0.5" not in r.text and "일시적으로 사용할 수 없습니다" in r.json()["detail"]
    with c.stream("POST", "/chat/stream", json={"message": "안녕"}, headers=hk) as resp:
        raw = "".join(resp.iter_text())
    assert "10.0.0.5" not in raw and "DAI_LOCAL_BASE_URL" not in raw
    assert sse_events(raw)[-1][0] == "error"
    h = login(c, "admin", "adminpass1")  # 관리자는 원인 파악을 위해 원문을 본다
    assert c.post("/chat", json={"message": "안녕"}, headers=h).json()["detail"] == secret


# ---- 차단·실패 턴의 빈 세션 --------------------------------------------------------
def test_blocked_or_failed_first_turn_leaves_no_empty_session(assistant, db):
    c = TestClient(create_app(assistant, db))
    hk = login(c, "kim", "kimpass123")
    body = c.post("/chat", json={"message": "대외비 양식 설명"}, headers=hk).json()
    assert body["blocked"] is True and body["session_id"] is None
    with c.stream("POST", "/chat/stream", json={"message": "대외비 양식 설명"}, headers=hk) as resp:
        events = sse_events("".join(resp.iter_text()))
    assert events[-1][0] == "done" and events[-1][1]["session_id"] is None
    real_chat = assistant.chat
    assistant.chat = lambda *a, **kw: (_ for _ in ()).throw(AssistantError("오류"))
    assert c.post("/chat", json={"message": "안녕"}, headers=hk).status_code == 502
    assert db.list_sessions("kim") == []
    # 정상 턴은 저장되고, 기존 대화에서 차단된 턴은 대화를 지우지 않는다
    assistant.chat = real_chat
    sid = c.post("/chat", json={"message": "안녕"}, headers=hk).json()["session_id"]
    assert sid and c.post("/chat", json={"message": "대외비 양식", "session_id": sid}, headers=hk).json()["session_id"] == sid
    assert [s.session_id for s in db.list_sessions("kim")] == [sid]
    # 명시적으로 만든 새 대화는 그대로 저장된다
    assert c.post("/sessions", headers=hk).status_code == 200 and len(db.list_sessions("kim")) == 2


# ---- 만료 토큰 정리 ----------------------------------------------------------------
def test_login_purges_expired_tokens(assistant, db):
    for _ in range(3):
        db.issue_token("kim", ttl_hours=-1)
    c = TestClient(create_app(assistant, db))
    login(c, "kim", "kimpass123")
    with db._lock:
        rows = db._conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
    assert rows == 1


# ---- 세션 처리 표시가 쌓이지 않음 --------------------------------------------------
def test_active_turn_registry_is_cleared(assistant, db):
    app = create_app(assistant, db)
    c = TestClient(app)
    h = login(c, "admin", "adminpass1")
    for _ in range(5):
        sid = c.post("/chat", json={"message": "안녕"}, headers=h).json()["session_id"]
        c.delete(f"/sessions/{sid}", headers=h)
    with c.stream("POST", "/chat/stream", json={"message": "안녕"}, headers=h) as resp:
        "".join(resp.iter_text())
    assert app.state.active_turns == set()


# ---- 짧은 DAI_ADMIN_PASSWORD -------------------------------------------------------
def test_short_bootstrap_password_does_not_crash(make_assistant, caplog):
    a, _ = make_assistant(admin_password="short")
    d = Database(":memory:")
    with caplog.at_level(logging.ERROR, logger="defense_assistant.server"):
        app = create_app(a, d)
    assert d.count_users() == 0 and "8자 이상" in caplog.text
    assert TestClient(app).get("/health").status_code == 200
    d.close()


# ---- 대화 기록 복원 ----------------------------------------------------------------
def test_transcript_groups_turns_and_marks_empty_responses():
    messages = [
        {"role": "user", "content": "질문 1"},
        {"role": "assistant", "content": []},  # 출력 전 거부
        {"role": "user", "content": "질문 2"},
        {"role": "assistant", "content": [{"type": "text", "text": "검색해 보겠습니다."}, {"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "결과"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "답변"}]},
    ]
    assert transcript(messages) == [
        {"role": "user", "text": "질문 1"},
        {"role": "assistant", "text": NO_RESPONSE_TEXT},
        {"role": "user", "text": "질문 2"},
        {"role": "assistant", "text": "검색해 보겠습니다.답변"},
    ]


def test_refused_turn_reopens_with_notice(make_assistant, db):
    a, _ = make_assistant([make_message("", stop_reason="refusal")])
    c = TestClient(create_app(a, db))
    h = login(c, "admin", "adminpass1")
    sid = c.post("/chat", json={"message": "질문"}, headers=h).json()["session_id"]
    msgs = c.get(f"/sessions/{sid}", headers=h).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["text"] == NO_RESPONSE_TEXT


# ---- 웹 UI: 스트리밍 종료 안내, IME 조합 중 Enter ------------------------------------
def _js_function(name: str) -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    # 여러 줄 함수(`=> {` … `  };`) 또는 한 줄 함수
    m = re.search(rf"^  const {name} = [^\n]*\{{\n.*?^  \}};$", html, re.S | re.M) or re.search(rf"^  const {name} = [^\n]*;$", html, re.M)
    assert m, name
    return m.group(0)


def _run_node(script: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 가 없어 웹 UI 스크립트를 실행할 수 없습니다.")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def test_ui_stream_done_shows_notices():
    fn = _js_function("finalText")
    cases = [
        ["부분 응답", {"text": "부분 응답\n\n(응답이 최대 길이에 도달해 잘렸습니다. 이어서 답하도록 요청해 주십시오.)", "stop_reason": "max_tokens"}],
        ["검색해 보겠습니다.답", {"text": "답\n\n(도구 호출 횟수 한도에 도달해 중단했습니다.)", "stop_reason": "tool_use"}],
        ["부분 출력", {"text": "요청하신 내용은 안전 정책에 따라 답변할 수 없습니다.", "stop_reason": "refusal"}],
        ["", {"text": "차단 안내", "blocked": True}],
        ["정상 답변 (참고)", {"text": "정상 답변 (참고)", "stop_reason": "end_turn"}],
        ["이미 안내\n\n(잘렸습니다.)", {"text": "이미 안내\n\n(잘렸습니다.)", "stop_reason": "max_tokens"}],
    ]
    got = _run_node(fn + f"\nconsole.log(JSON.stringify({json.dumps(cases, ensure_ascii=False)}.map(([s, d]) => finalText(s, d))));")
    assert got == [
        "부분 응답\n\n(응답이 최대 길이에 도달해 잘렸습니다. 이어서 답하도록 요청해 주십시오.)",
        "검색해 보겠습니다.답\n\n(도구 호출 횟수 한도에 도달해 중단했습니다.)",
        "요청하신 내용은 안전 정책에 따라 답변할 수 없습니다.",
        "차단 안내",
        "정상 답변 (참고)",
        "이미 안내\n\n(잘렸습니다.)",
    ]


def test_ui_enter_during_ime_composition_does_not_submit():
    fn = _js_function("isSubmitKey")
    cases = [
        {"key": "Enter", "shiftKey": False, "isComposing": False, "keyCode": 13},
        {"key": "Enter", "shiftKey": False, "isComposing": True, "keyCode": 13},
        {"key": "Enter", "shiftKey": False, "isComposing": False, "keyCode": 229},
        {"key": "Enter", "shiftKey": True, "isComposing": False, "keyCode": 13},
    ]
    got = _run_node(fn + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(isSubmitKey)));")
    assert got == [True, False, False, False]
