import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from defense_assistant.server import create_app  # noqa: E402
from tests.conftest import make_message  # noqa: E402


@pytest.fixture
def client(make_assistant):
    assistant, _ = make_assistant([make_message("서버 응답")])
    return TestClient(create_app(assistant))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "search_defense_docs" in body["tools"]


def test_chat_creates_session_and_reuses_it(client):
    r = client.post("/chat", json={"message": "안녕하세요", "user_id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "서버 응답" and body["blocked"] is False
    sid = body["session_id"]
    r2 = client.post("/chat", json={"message": "다시", "session_id": sid})
    assert r2.json()["session_id"] == sid
    assert client.post("/chat", json={"message": "x", "session_id": "nope"}).status_code == 404
    assert client.delete(f"/sessions/{sid}").json() == {"deleted": True}


def test_chat_blocked(client):
    body = client.post("/chat", json={"message": "TOP SECRET 자료"}).json()
    assert body["blocked"] is True and body["classification"] == "TOP_SECRET"


def test_chat_stream_sse(client):
    with client.stream("POST", "/chat/stream", json={"message": "안녕"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [e for e in raw.split("\n\n") if e.strip()]
    kinds = [e.split("\n")[0] for e in events]
    assert kinds[0] == "event: text" and kinds[-1] == "event: done"
    text = "".join(json.loads(e.split("\ndata: ", 1)[1]) for e in events if e.startswith("event: text"))
    assert text == "서버 응답"
    done = json.loads(events[-1].split("\ndata: ", 1)[1])
    assert done["text"] == "서버 응답" and "session_id" in done


def test_docs_search(client):
    body = client.get("/docs/search", params={"q": "거수자", "top_k": 2}).json()
    assert body["hits"] and "거수자" in body["hits"][0]["ref"]


def test_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
