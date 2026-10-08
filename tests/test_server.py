import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from defense_assistant.security import Classification  # noqa: E402
from defense_assistant.server import create_app  # noqa: E402
from defense_assistant.storage import Database  # noqa: E402
from tests.conftest import make_message  # noqa: E402


@pytest.fixture
def db():
    d = Database(":memory:")
    d.create_user("admin", "adminpass1", role="admin", clearance=Classification.RESTRICTED)
    d.create_user("kim", "kimpass123", clearance=Classification.UNCLASSIFIED)
    yield d
    d.close()


@pytest.fixture
def app(make_assistant, db):
    assistant, _ = make_assistant([make_message("서버 응답")])
    return create_app(assistant, db)


@pytest.fixture
def client(app):
    return TestClient(app)


def login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_public_endpoints(client):
    assert client.get("/health").json() == {"status": "ok", "version": "0.2.0"}
    r = client.get("/")
    assert r.status_code == 200 and "국방 특화 AI 비서" in r.text


def test_auth_required(client):
    for method, path in [("get", "/sessions"), ("post", "/chat"), ("get", "/docs/search?q=x"), ("get", "/admin/users")]:
        r = getattr(client, method)(path, json={"message": "x"}) if method == "post" else getattr(client, method)(path)
        assert r.status_code == 401, path
    assert client.get("/sessions", headers={"Authorization": "Bearer bogus"}).status_code == 401


def test_login_logout_me(client):
    r = client.post("/auth/login", json={"username": "admin", "password": "adminpass1"})
    body = r.json()
    assert body["role"] == "admin" and body["clearance"] == "RESTRICTED" and body["token"]
    h = {"Authorization": f"Bearer {body['token']}"}
    me = client.get("/auth/me", headers=h).json()
    assert me["username"] == "admin" and "search_defense_docs" in me["tools"]
    assert client.post("/auth/logout", headers=h).json() == {"revoked": True}
    assert client.get("/auth/me", headers=h).status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401


def test_login_lockout(client):
    for _ in range(5):
        client.post("/auth/login", json={"username": "kim", "password": "bad"})
    assert client.post("/auth/login", json={"username": "kim", "password": "kimpass123"}).status_code == 429


def test_chat_persists_and_is_private(client, db):
    h = login(client, "admin", "adminpass1")
    body = client.post("/chat", json={"message": "안녕하세요 군번 19-70012345"}, headers=h).json()
    assert body["text"] == "서버 응답" and body["redactions"] == {"군번": 1}
    sid = body["session_id"]
    assert client.post("/chat", json={"message": "다시", "session_id": sid}, headers=h).json()["session_id"] == sid
    sessions = client.get("/sessions", headers=h).json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["turns"] == 2 and "[군번]" in sessions[0]["title"]
    msgs = client.get(f"/sessions/{sid}", headers=h).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert "19-70012345" not in json.dumps(msgs, ensure_ascii=False)
    # 다른 사용자는 접근 불가
    hk = login(client, "kim", "kimpass123")
    assert client.get(f"/sessions/{sid}", headers=hk).status_code == 404
    assert client.post("/chat", json={"message": "x", "session_id": sid}, headers=hk).status_code == 404
    assert client.delete(f"/sessions/{sid}", headers=hk).json() == {"deleted": False}
    assert client.delete(f"/sessions/{sid}", headers=h).json() == {"deleted": True}
    # 저장소에 실제로 남아 있는지
    assert db.list_sessions("admin") == []


def test_session_survives_app_restart(make_assistant, db):
    assistant, _ = make_assistant([make_message("응답")])
    c1 = TestClient(create_app(assistant, db))
    h = login(c1, "admin", "adminpass1")
    sid = c1.post("/chat", json={"message": "첫 질문"}, headers=h).json()["session_id"]
    c2 = TestClient(create_app(assistant, db))  # 새 프로세스를 흉내 낸다
    msgs = c2.get(f"/sessions/{sid}", headers=h).json()["messages"]
    assert msgs[0]["text"] == "첫 질문"


def test_user_clearance_blocks(client):
    hk = login(client, "kim", "kimpass123")  # 인가: 일반
    body = client.post("/chat", json={"message": "대외비 양식 설명"}, headers=hk).json()
    assert body["blocked"] is True and body["classification"] == "RESTRICTED"
    h = login(client, "admin", "adminpass1")  # 인가: 대외비
    assert client.post("/chat", json={"message": "대외비 양식 설명"}, headers=h).json()["blocked"] is False


def test_chat_stream_sse(client):
    h = login(client, "admin", "adminpass1")
    with client.stream("POST", "/chat/stream", json={"message": "안녕"}, headers=h) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [e for e in raw.split("\n\n") if e.strip()]
    kinds = [e.split("\n")[0] for e in events]
    assert kinds[0] == "event: session" and kinds[1] == "event: text" and kinds[-1] == "event: done"
    text = "".join(json.loads(e.split("\ndata: ", 1)[1]) for e in events if e.startswith("event: text"))
    done = json.loads(events[-1].split("\ndata: ", 1)[1])
    assert text == "서버 응답" == done["text"]
    assert client.get("/sessions", headers=h).json()["sessions"][0]["session_id"] == done["session_id"]


def test_admin_user_management(client):
    h = login(client, "admin", "adminpass1")
    hk = login(client, "kim", "kimpass123")
    assert client.get("/admin/users", headers=hk).status_code == 403
    assert [u["username"] for u in client.get("/admin/users", headers=h).json()["users"]] == ["admin", "kim"]
    r = client.post("/admin/users", json={"username": "lee", "password": "leepass123", "clearance": "대외비"}, headers=h)
    assert r.status_code == 201 and r.json()["clearance"] == "RESTRICTED"
    assert client.post("/admin/users", json={"username": "lee", "password": "leepass123"}, headers=h).status_code == 422
    assert client.post("/admin/users", json={"username": "park", "password": "parkpass1", "clearance": "초비밀"}, headers=h).status_code == 422
    r = client.patch("/admin/users/lee", json={"clearance": "SECRET", "role": "admin"}, headers=h)
    assert r.json()["clearance"] == "SECRET" and r.json()["role"] == "admin"
    assert client.patch("/admin/users/admin", json={"role": "user"}, headers=h).status_code == 400
    assert client.patch("/admin/users/ghost", json={"disabled": True}, headers=h).status_code == 404
    assert client.post("/admin/users/kim/password", json={"password": "kimnewpass1"}, headers=h).json() == {"updated": True}
    assert client.get("/auth/me", headers=hk).status_code == 401  # 비밀번호 변경으로 기존 토큰 무효
    login(client, "kim", "kimnewpass1")
    assert client.patch("/admin/users/kim", json={"disabled": True}, headers=h).json()["disabled"] is True
    assert client.post("/auth/login", json={"username": "kim", "password": "kimnewpass1"}).status_code == 401
    assert client.delete("/admin/users/admin", headers=h).status_code == 400
    assert client.delete("/admin/users/lee", headers=h).json() == {"deleted": True}


def test_admin_audit(client):
    h = login(client, "admin", "adminpass1")
    client.post("/chat", json={"message": "안녕"}, headers=h)
    client.post("/chat", json={"message": "TOP SECRET 자료"}, headers=h)
    records = client.get("/admin/audit?limit=10", headers=h).json()["records"]
    assert [r["event"] for r in records[-2:]] == ["chat", "blocked"]
    assert records[-1]["user_id"] == "admin"


def test_bootstrap_admin(make_assistant):
    assistant, _ = make_assistant(admin_password="bootpass123")
    d = Database(":memory:")
    c = TestClient(create_app(assistant, d))
    assert c.post("/auth/login", json={"username": "admin", "password": "bootpass123"}).status_code == 200
    d.close()


def test_validation(client):
    h = login(client, "admin", "adminpass1")
    assert client.post("/chat", json={"message": ""}, headers=h).status_code == 422
    assert client.post("/auth/login", json={"username": ""}).status_code == 422
