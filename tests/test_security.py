import json

import pytest

from defense_assistant.security import AuditLogger, AuditRecord, Classification, classify_text, redact
from defense_assistant.security.classification import block_message


@pytest.mark.parametrize(
    "text,level",
    [
        ("오늘 점심 메뉴 알려줘", Classification.UNCLASSIFIED),
        ("비밀번호를 잊었습니다", Classification.UNCLASSIFIED),
        ("The secret garden", Classification.UNCLASSIFIED),
        ("대외비 자료입니다", Classification.RESTRICTED),
        ("FOUO 문서", Classification.RESTRICTED),
        ("III급 비밀 취급", Classification.CONFIDENTIAL),
        ("3급비밀", Classification.CONFIDENTIAL),
        ("CONFIDENTIAL", Classification.CONFIDENTIAL),
        ("군사기밀 보호", Classification.CONFIDENTIAL),
        ("II급 비밀 문서 요약", Classification.SECRET),
        ("2급 기밀", Classification.SECRET),
        ("SECRET//NOFORN", Classification.SECRET),
        ("I급 비밀", Classification.TOP_SECRET),
        ("1급비밀", Classification.TOP_SECRET),
        ("TOP SECRET", Classification.TOP_SECRET),
        ("TS//SCI", Classification.TOP_SECRET),
        ("극비 사항", Classification.TOP_SECRET),
        ("12급 비밀", Classification.UNCLASSIFIED),
        ("IV급", Classification.UNCLASSIFIED),
    ],
)
def test_classify_text(text, level):
    assert classify_text(text).level == level


def test_classification_parse_aliases():
    assert Classification.parse("대외비") == Classification.RESTRICTED
    assert Classification.parse("secret") == Classification.SECRET
    assert Classification.parse("top secret") == Classification.TOP_SECRET
    assert Classification.parse(2) == Classification.CONFIDENTIAL
    with pytest.raises(ValueError):
        Classification.parse("초비밀")


def test_allowed_under_and_block_message():
    r = classify_text("II급 비밀")
    assert not r.allowed_under(Classification.RESTRICTED)
    assert r.allowed_under(Classification.SECRET)
    msg = block_message(r, Classification.RESTRICTED)
    assert "II급 비밀" in msg and "대외비" in msg


def test_redact_masks_all_categories():
    text = "홍길동 900101-1234567 군번 19-70012345 / 95-12345 전화 010-1234-5678, 02-123-4567 메일 a.b@mnd.go.kr 좌표 52S CG 12345 67890 위치 37.5665, 126.9780 서버 192.168.10.3"
    r = redact(text)
    assert "[주민등록번호]" in r.text and "900101" not in r.text
    assert r.text.count("[군번]") == 2
    assert "[휴대전화]" in r.text and "[전화번호]" in r.text
    assert "[이메일]" in r.text
    assert r.text.count("[좌표]") == 2
    assert "[내부IP]" in r.text
    assert r.total == 9
    assert "주민등록번호 1건" in r.summary()


def test_redact_leaves_normal_text_and_dates():
    r = redact("2026-10-07 14:30 회의, 3개 소대 120명, 거리 12.5km, 공인 IP 8.8.8.8")
    assert not r.changed
    assert r.text.startswith("2026-10-07")


def test_redact_disabled():
    r = redact("010-1234-5678", enabled=False)
    assert r.text == "010-1234-5678" and not r.changed


def test_audit_logger_writes_jsonl(tmp_path):
    path = tmp_path / "audit" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write(AuditRecord(session_id="s1", user_id="u1", event="chat", input_sha256="abc", input_chars=3, classification="UNCLASSIFIED", tools_called=["x"]))
    logger.write(AuditRecord(session_id="s1", user_id="u1", event="blocked", input_sha256="def", input_chars=3, classification="SECRET"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["event"] == "chat" and rec["tools_called"] == ["x"] and "timestamp" in rec
    assert [r["event"] for r in logger.read_all()] == ["chat", "blocked"]
