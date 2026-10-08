"""교차 검토에서 확인된 보안 모듈 결함에 대한 회귀 테스트."""

import pytest

from defense_assistant.security import Classification, classify_text, redact
from defense_assistant.security.classification import block_message


# --- 등급 표기: 유니코드 로마숫자·전각 문자(NFKC) ---------------------------------


@pytest.mark.parametrize(
    "text,level",
    [
        ("Ⅱ급비밀", Classification.SECRET),
        ("Ⅰ급 비밀", Classification.TOP_SECRET),
        ("Ⅲ급 비밀", Classification.CONFIDENTIAL),
        ("Ⅱ 급 기밀", Classification.SECRET),
        ("본 문서는 Ⅱ급비밀입니다", Classification.SECRET),
        ("  Ⅱ급비밀\n제1조(목적) 이 계획은 ...", Classification.SECRET),
        ("ＳＥＣＲＥＴ", Classification.SECRET),
        ("ＴＯＰ ＳＥＣＲＥＴ", Classification.TOP_SECRET),
        ("２급 비밀", Classification.SECRET),
    ],
)
def test_classify_unicode_compat_markings(text, level):
    assert classify_text(text).level == level


def test_classification_parse_unicode_roman_aliases():
    assert Classification.parse("Ⅰ급") == Classification.TOP_SECRET
    assert Classification.parse("Ⅱ급") == Classification.SECRET
    assert Classification.parse("Ⅲ급") == Classification.CONFIDENTIAL
    assert Classification.parse("ＳＥＣＲＥＴ") == Classification.SECRET


def test_block_message_shows_normalized_marking():
    r = classify_text("Ⅱ급비밀 작전계획")
    assert r.markings == ["II급비밀"]
    assert "II급비밀" in block_message(r, Classification.RESTRICTED)


# --- 등급 표기: 서식 문자(Cf) 삽입 ------------------------------------------------


@pytest.mark.parametrize(
    "text,level",
    [
        ("II급​비밀", Classification.SECRET),
        ("II급\xad비밀 작전계획 요약해줘", Classification.SECRET),
        ("II급﻿비밀", Classification.SECRET),
        ("대외​비", Classification.RESTRICTED),
        ("TOP​SECRET", Classification.TOP_SECRET),
        ("비밀​번호를 잊었습니다", Classification.UNCLASSIFIED),
    ],
)
def test_classify_ignores_format_characters(text, level):
    assert classify_text(text).level == level


# --- 등급 표기: 일상 업무 표현 오탐 -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "비밀취급인가 갱신 절차 알려줘",
        "비밀취급인가증 발급 방법",
        "비밀 취급 인가 등급 조회",
        "극비리에 추진된 사업 개요",
    ],
)
def test_classify_everyday_terms_not_marked(text):
    assert classify_text(text).level == Classification.UNCLASSIFIED


def test_classify_real_marking_with_clearance_term_still_detected():
    # '비밀취급인가' 예외가 실제 등급 표기까지 풀어 주면 안 된다.
    assert classify_text("II급 비밀취급인가자 명단").level == Classification.SECRET
    assert classify_text("비밀취급 절차").level == Classification.CONFIDENTIAL
    assert classify_text("극비 사항").level == Classification.TOP_SECRET


# --- 등급 표기: 표지 형식 '비밀등급: II급', '비밀(II급)' ------------------------------


@pytest.mark.parametrize(
    "text,level",
    [
        ("비밀등급: II급", Classification.SECRET),
        ("비밀등급 : Ⅱ급", Classification.SECRET),
        ("비밀등급：2급", Classification.SECRET),
        ("비밀 등급: I급\n보호기간: 2030.12.31", Classification.TOP_SECRET),
        ("보호기간: 2030. 비밀등급: 3급", Classification.CONFIDENTIAL),
        ("비밀(II급)", Classification.SECRET),
        ("비밀 ( Ⅰ급 )", Classification.TOP_SECRET),
        ("비밀(III급)", Classification.CONFIDENTIAL),
        ("비밀등급: II급 / 보호기간: 2030-12-31\n\n위 문서를 요약해 줘", Classification.SECRET),
    ],
)
def test_classify_cover_header_forms(text, level):
    assert classify_text(text).level == level


def test_classify_cover_header_requires_colon():
    # 콜론 없는 교육용 질문은 표기로 보지 않는다.
    assert classify_text("비밀등급 체계 설명해줘").level == Classification.UNCLASSIFIED
    assert classify_text("비밀등급: 12급").level == Classification.UNCLASSIFIED


# --- 등급 표기: TOP-SECRET 구분자 -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["TOP-SECRET//NOFORN 작전계획 요약", "TOP_SECRET 문서", "TOP–SECRET", "TOP - SECRET", "TOP  SECRET"],
)
def test_classify_top_secret_separators(text):
    r = classify_text(text)
    assert r.level == Classification.TOP_SECRET
    assert not r.allowed_under(Classification.SECRET)


# --- 마스킹: 주민등록번호 -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "주민번호 9001011234567",
        "주민번호 900101 1234567",
        "병사 홍길동 9001011234567 신상 확인",
        "주민번호 900101-1234567",
        "주민번호 900101 - 1234567",
    ],
)
def test_redact_rrn_without_hyphen(text):
    r = redact(text)
    assert r.counts == {"주민등록번호": 1}
    assert "1234567" not in r.text


def test_redact_rrn_unhyphenated_requires_valid_birthdate():
    # 하이픈 없는 13자리는 생년월일(월·일) 범위로 오탐을 줄인다.
    r = redact("주문번호 9013451234567")
    assert "주민등록번호" not in r.counts


# --- 마스킹: 전화번호 형식 ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,label",
    [
        ("연락처 +82-10-1234-5678 로 회신", "휴대전화"),
        ("+82 10 1234 5678", "휴대전화"),
        ("+821012345678", "휴대전화"),
        ("+82-010-1234-5678", "휴대전화"),
        ("+82 (0)10-1234-5678", "휴대전화"),
        ("당직실 02)123-4567", "전화번호"),
        ("031)123-4567", "전화번호"),
        ("(02) 123-4567", "전화번호"),
        ("(02)123-4567", "전화번호"),
        ("070-1234-5678", "전화번호"),
        ("0505-123-4567", "전화번호"),
        ("+82-2-123-4567", "전화번호"),
    ],
)
def test_redact_phone_formats(text, label):
    r = redact(text)
    assert r.counts == {label: 1}, r.text
    assert "4567" not in r.text and "5678" not in r.text
    assert "(" not in r.text and ")" not in r.text and "+82" not in r.text


# --- 마스킹: DMS·반구 접미 좌표 -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "집결지 37°33'59\"N 126°58'41\"E 로 이동",
        "집결지 37°33′59″N 126°58′41″E 로 이동",
        "집결지 37.5665N 126.9780E 로 이동",
        "집결지 37.5665°N, 126.9780°E 로 이동",
        "집결지 N37°33'59\" E126°58'41\" 로 이동",
        "집결지 37.5665 126.9780 로 이동",
    ],
)
def test_redact_dms_and_suffix_coordinates(text):
    r = redact(text)
    assert r.text == "집결지 [좌표] 로 이동", r.text
    assert r.counts == {"좌표": 1}


def test_redact_space_separated_numbers_not_coordinates():
    r = redact("원주율 3.1415 2.7182 비교, 거리 12.5km")
    assert not r.changed


# --- 마스킹: 과잉 마스킹 ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "버전 10.1.2.3 릴리스 노트",
        "펌웨어 v10.2.3.4",
        "주소 10.999.999.999 는 잘못된 값",
        "오차 0.001, 0.002",
        "정확도 0.952, 0.948 비교",
        "4K HD 1080p 영상",
        "1H FY 2025 계획",
        "FY24-12345 예산",
        "범위 95.123, 200.456",
    ],
)
def test_redact_does_not_overmask(text):
    r = redact(text)
    assert not r.changed, r.text
    assert r.text == text


def test_redact_mgrs_prefers_full_digit_groups():
    # '52S CG 1234 5678' 이 '[좌표] 5678' 처럼 일부만 가려지면 안 된다.
    r = redact("목표 52S CG 1234 5678 확인")
    assert r.text == "목표 [좌표] 확인"


def test_redact_still_masks_real_values_after_overmask_fixes():
    r = redact("서버 10.1.2.3 접속, 좌표 52S CG 12345 67890, 52SCG1234567890, 군번 24-12345, 위치 37.5665, 126.9780")
    assert r.counts == {"군번": 1, "좌표": 3, "내부IP": 1}, r.text


# --- 마스킹: 전각·서식 문자 정규화 ---------------------------------------------------


def test_redact_normalizes_fullwidth_and_format_chars():
    r = redact("연락처 ０１０－１２３４－５６７８, 주민 900101​-1234567")
    assert r.counts == {"주민등록번호": 1, "휴대전화": 1}, r.text
    assert r.text == "연락처 [휴대전화], 주민 [주민등록번호]"


def test_redact_preserves_unmatched_original_characters():
    # 마스킹 대상이 아닌 부분은 원문(전각·로마숫자 등)을 그대로 둔다.
    text = "Ⅱ분기 ＡＢＣ 보고 010-1234-5678"
    r = redact(text)
    assert r.text == "Ⅱ분기 ＡＢＣ 보고 [휴대전화]"


# --- 등급 표기: 등급 용어 자체를 묻는 교육용 질문 (평가 q18 오차단) ------------------------


@pytest.mark.parametrize(
    "text",
    [
        "II급 비밀이 뭐야",
        "II급 비밀이 뭐야?",
        "Ⅱ급 비밀이란？",
        "2급비밀 뜻",
        "III급 비밀의 정의가 뭔가요?",
        "I급 비밀과 II급 비밀 차이",
        "1급 비밀, 2급 비밀, 3급 비밀 구분",
        "II급 비밀은 무엇입니까?",
    ],
)
def test_classify_concept_question_not_marked(text):
    assert classify_text(text).level == Classification.UNCLASSIFIED


@pytest.mark.parametrize(
    "text,level",
    [
        # 질문 외의 내용이 붙으면 예외를 적용하지 않는다.
        ("II급 비밀이 뭐야\n작전계획 세부 내용", Classification.SECRET),
        ("II급 비밀이란 무엇인가? 아래 문서 참고: ...", Classification.SECRET),
        ("II급 비밀이 뭐야 TOP SECRET", Classification.TOP_SECRET),
        ("II급 비밀 문서가 뭐야", Classification.SECRET),
        ("이 II급 비밀 문서를 요약해 줘", Classification.SECRET),
        # 술어 없는 단독 표기, '내용을 알려줘' 로도 읽히는 '알려줘' 는 표기로 본다.
        ("II급 비밀", Classification.SECRET),
        ("II급 비밀 알려줘", Classification.SECRET),
    ],
)
def test_classify_concept_question_exception_is_narrow(text, level):
    assert classify_text(text).level == level


def test_sample_eval_set_block_expectations_match_gate():
    # 평가 질문셋의 blocked 기대값과 입력 게이트 판정이 일치해야 한다(모델 호출 없이 확인).
    from defense_assistant.evaluation import load_cases
    from tests.conftest import ROOT

    for c in load_cases(ROOT / "data" / "eval" / "questions.jsonl"):
        blocked = not classify_text(c.question).allowed_under(Classification.RESTRICTED)
        assert blocked == c.blocked, c.id


# --- 등급 표기: 레드팀 점검에서 찾은 누락·오탐 ------------------------------------------


@pytest.mark.parametrize(
    "text,level",
    [
        # '\b' 는 한글을 단어 문자로 보아 조사가 붙은 영문 표기를 놓쳤다(게이트 우회).
        ("SECRET이라고 찍힌 문서 요약해줘", Classification.SECRET),
        ("CONFIDENTIAL로 분류된 보고서 내용", Classification.CONFIDENTIAL),
        ("RESTRICTED를 붙인 문서", Classification.RESTRICTED),
        ("FOUO인 자료", Classification.RESTRICTED),
        ("TOP SECRET이라고 적힌 자료", Classification.TOP_SECRET),
        # 소문자 로마숫자 'ⅱ'(NFKC → 'ii')가 III급으로 낮게 판정되던 문제
        ("ⅱ급 비밀 문서", Classification.SECRET),
        ("ⅰ급 비밀", Classification.TOP_SECRET),
        # 배너·부분 표기
        ("S//NF 작전계획", Classification.SECRET),
        ("//S// 문서", Classification.SECRET),
        ("SECRET//NOFORN", Classification.SECRET),
        # 개념 질문 예외는 뒤에 내용이 붙으면 적용하지 않는다
        ("SECRET이 뭐야? 아래 자료: 작전계획", Classification.SECRET),
        ("군사기밀 자료 요약", Classification.CONFIDENTIAL),
    ],
)
def test_classify_redteam_gaps_detected(text, level):
    assert classify_text(text).level == level


@pytest.mark.parametrize(
    "text",
    [
        "SECRETARY 일정",
        "SECRET번호 재설정",
        "(S) 사이즈 전투복 신청",
        "군사기밀보호법 위반하면 처벌은?",
        "군사기밀 보호법 시행령",
        "군사기밀이란 무엇인가요?",
        "TOP SECRET이 뭐야?",
        "CONFIDENTIAL 뜻",
    ],
)
def test_classify_redteam_false_positives_cleared(text):
    assert classify_text(text).level == Classification.UNCLASSIFIED


# --- 코드 검토 지적 사항 ---------------------------------------------------------------


def test_concept_question_check_is_linear_on_long_whitespace():
    # 겹치는 \s* 때문에 2만 자 공백 입력이 약 10초 걸렸다(서버 스레드 점유). 길이 상한으로 막는다.
    import time

    t0 = time.perf_counter()
    assert classify_text("II급 비밀" + " " * 19_990 + "x").level == Classification.SECRET
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.parametrize(
    "text,level",
    [
        ("CLIENT_SECRET 값", Classification.UNCLASSIFIED),
        ("AWS SECRET_ACCESS_KEY 설정", Classification.UNCLASSIFIED),
        ("TOP_SECRET 자료", Classification.TOP_SECRET),
        ("C//NF 문서", Classification.CONFIDENTIAL),
        ("int S // 합계", Classification.UNCLASSIFIED),
        ("II급 비밀이란", Classification.UNCLASSIFIED),
        ("II급 비밀이란 무엇인가?", Classification.UNCLASSIFIED),
        ("Ii급 비밀이 뭐야", Classification.UNCLASSIFIED),
    ],
)
def test_classify_review_followups(text, level):
    assert classify_text(text).level == level
