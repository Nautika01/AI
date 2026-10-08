import time

import pytest

from defense_assistant import auth
from defense_assistant.assistant import Session
from defense_assistant.security import Classification
from defense_assistant.storage import Database, transcript


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


def test_password_hash_and_verify():
    h = auth.hash_password("correct horse")
    assert h.startswith("scrypt$") and auth.verify_password("correct horse", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "garbage")
    with pytest.raises(ValueError):
        auth.hash_password("short")


def test_throttle():
    t = auth.LoginThrottle(max_attempts=2, lockout_seconds=0.2)
    t.record_failure("k")
    assert not t.is_locked("k")
    t.record_failure("k")
    assert t.is_locked("k")
    time.sleep(0.25)
    assert not t.is_locked("k")
    t.record_failure("k"); t.record_failure("k"); t.reset("k")
    assert not t.is_locked("k")


def test_users_crud(db):
    u = db.create_user("admin", "adminpass1", role="admin", clearance=Classification.SECRET)
    assert u.role == "admin" and u.clearance == Classification.SECRET and not u.disabled
    with pytest.raises(ValueError):
        db.create_user("admin", "adminpass1")
    with pytest.raises(ValueError):
        db.create_user("bad name!", "adminpass1")
    with pytest.raises(ValueError):
        db.create_user("x", "adminpass1", role="root")
    assert db.verify_credentials("admin", "adminpass1").username == "admin"
    assert db.verify_credentials("admin", "nope") is None
    assert db.verify_credentials("ghost", "adminpass1") is None
    assert db.update_user("admin", clearance=Classification.UNCLASSIFIED, role="user")
    assert db.get_user("admin").clearance == Classification.UNCLASSIFIED
    assert db.update_user("admin", disabled=True) and db.verify_credentials("admin", "adminpass1") is None
    assert not db.update_user("ghost", disabled=True)
    assert [x.username for x in db.list_users()] == ["admin"] and db.count_users() == 1
    assert db.delete_user("admin") and not db.delete_user("admin")


def test_tokens(db):
    db.create_user("kim", "kimpass123")
    token, exp = db.issue_token("kim", ttl_hours=1)
    p = db.resolve_token(token)
    assert p.username == "kim" and p.role == "user" and p.clearance == Classification.RESTRICTED and not p.is_admin
    assert db.resolve_token("nonsense") is None
    expired, _ = db.issue_token("kim", ttl_hours=-1)
    assert db.resolve_token(expired) is None
    assert db.revoke_token(token) and db.resolve_token(token) is None
    db.issue_token("kim", ttl_hours=1)
    db.set_password("kim", "newpass1234")  # 비밀번호 변경 시 토큰 전부 폐기
    assert db.purge_expired_tokens() == 0
    assert db.verify_credentials("kim", "newpass1234") is not None


def test_sessions_roundtrip_and_ownership(db):
    db.create_user("kim", "kimpass123", clearance=Classification.UNCLASSIFIED)
    db.create_user("lee", "leepass123")
    s = Session(user_id="kim", messages=[
        {"role": "user", "content": "경계 근무 교대 절차 알려줘"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": "다음과 같습니다."}]},
    ], turns=1)
    db.save_session(s)
    loaded = db.load_session(s.session_id, username="kim")
    assert loaded.messages == s.messages and loaded.turns == 1 and loaded.user_id == "kim"
    assert loaded.max_classification == Classification.UNCLASSIFIED
    assert db.load_session(s.session_id, username="lee") is None
    summaries = db.list_sessions("kim")
    assert len(summaries) == 1 and summaries[0].title == "경계 근무 교대 절차 알려줘"
    assert db.list_sessions("lee") == []
    assert transcript(s.messages) == [{"role": "user", "text": "경계 근무 교대 절차 알려줘"}, {"role": "assistant", "text": "다음과 같습니다."}]
    assert not db.delete_session(s.session_id, username="lee")
    assert db.delete_session(s.session_id, username="kim")
    # 사용자 삭제 시 세션도 함께 삭제
    db.save_session(Session(user_id="kim"))
    db.delete_user("kim")
    assert db.list_sessions("kim") == []


def test_assistant_messages_are_json_serializable(make_assistant):
    import json
    from tests.conftest import make_message

    assistant, _ = make_assistant([
        make_message("", stop_reason="tool_use", tool_uses=[("spell_phonetic", {"text": "AB"})]),
        make_message("Alfa Bravo"),
    ])
    session = assistant.new_session("kim")
    assistant.chat(session, "AB 음성문자")
    json.dumps(session.messages)  # SDK 객체가 남아 있으면 실패
    assert session.messages[1]["content"][0]["type"] == "tool_use"


def test_session_clearance_caps_global_setting(make_assistant):
    assistant, client = make_assistant()  # 전역 허용: 대외비
    s = assistant.new_session("kim", max_classification=Classification.UNCLASSIFIED)
    r = assistant.chat(s, "대외비 양식 알려줘")
    assert r.blocked and "일반" in r.text and client.calls == []
    s2 = assistant.new_session("admin", max_classification=Classification.TOP_SECRET)
    assert assistant.effective_max_classification(s2) == Classification.RESTRICTED  # 전역 설정이 상한
