"""교차 검토에서 확인된 문서 적재(ingest) 결함의 회귀 테스트."""

import io
import struct
import zipfile

import pytest

from defense_assistant.ingest import convert_file, extract_paragraphs, ingest_paths, to_markdown
from defense_assistant.ingest.convert import _detect_mode, _structure
from defense_assistant.ingest.extract import ExtractError, _decode_para_text, extract_docx, extract_hwpx, extract_pdf, merge_pdf_lines, strip_pdf_page_numbers
from defense_assistant.knowledge import parse_front_matter

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _hwpx(section_body: str) -> bytes:
    xml = f'<?xml version="1.0" encoding="UTF-8"?><hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" xmlns:hp="{HP}">{section_body}</hs:sec>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", xml)
    return buf.getvalue()


def _docx(body: str) -> bytes:
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}" xmlns:mc="{MC}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _hp(text_xml: str) -> str:
    return f"<hp:p><hp:run><hp:t>{text_xml}</hp:t></hp:run></hp:p>"


# ---- HWP 5.0: 탭은 8 WCHAR 인라인 제어 문자, 코드 0 은 1 WCHAR --------------------
def _hwp_tab(width: int = 0) -> bytes:
    # 0x0009 + 정보 6 WCHAR(탭 폭 UINT32, 종류 등) + 0x0009 = 16바이트
    return struct.pack("<H", 9) + struct.pack("<I", width) + struct.pack("<H", 1) + b"\x00" * 6 + struct.pack("<H", 9)


@pytest.mark.parametrize("width", [0, 4000, 0x2000])
def test_hwp_tab_is_8_wchar_inline_control(width):
    payload = "이름".encode("utf-16-le") + _hwp_tab(width) + "홍길동 대위".encode("utf-16-le") + struct.pack("<H", 13)
    assert _decode_para_text(payload) == "이름 홍길동 대위 "


def test_hwp_nul_is_single_wchar():
    payload = "가".encode("utf-16-le") + struct.pack("<H", 0) + "나다라마바사아자".encode("utf-16-le")
    assert _decode_para_text(payload) == "가나다라마바사아자"


def test_hwp_surrogate_pair_decodes_to_one_char():
    payload = "성명 𠀋".encode("utf-16-le") + struct.pack("<H", 13)
    text = _decode_para_text(payload)
    assert text == "성명 𠀋 "
    text.encode("utf-8")  # 고립 서로게이트가 있으면 여기서 UnicodeEncodeError


def test_ingest_paths_write_failure_is_reported_per_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("가 문서\n본문이다.", encoding="utf-8")
    (src / "b.txt").write_text("나 문서\n본문이다.", encoding="utf-8")
    out = tmp_path / "docs"
    (out / "a.md").mkdir(parents=True)  # 쓰기가 실패하도록 같은 이름의 폴더를 만든다
    results = ingest_paths([src], out, overwrite=True)
    assert len(results) == 2
    assert results[0].error and results[1].ok
    assert (out / "b.md").read_text(encoding="utf-8").startswith("---\ntitle: 나 문서")


# ---- HWPX ---------------------------------------------------------------
def test_hwpx_mixed_content_tail_text_is_kept():
    body = _hp('제1조(목적)<hp:tab width="4000" leader="0" type="1"/>이 규정은 보안업무를 정한다.')
    body += _hp('<hp:markpenBegin color="#FFFF00"/>비밀취급인가자만<hp:markpenEnd/>열람할 수 있다.')
    body += _hp("첫째 줄<hp:lineBreak/>둘째 줄")
    body += "<hp:p><hp:run><hp:t>표 앞<hp:tab/>설명문</hp:t><hp:tbl><hp:tr><hp:tc><hp:subList>" + _hp("셀<hp:lineBreak/>내용") + "</hp:subList></hp:tc></hp:tr></hp:tbl></hp:run></hp:p>"
    paras = [p.strip() for p in extract_hwpx(_hwpx(body))]
    assert paras == ["제1조(목적) 이 규정은 보안업무를 정한다.", "비밀취급인가자만열람할 수 있다.", "첫째 줄 둘째 줄", "표 앞 설명문", "| 셀 내용 |"]


def test_hwpx_header_footer_footnote_not_merged_into_body(tmp_path):
    header = "<hp:ctrl><hp:header><hp:subList>" + _hp("Ⅲ급비밀 경계근무 규정") + "</hp:subList></hp:header></hp:ctrl>"
    footer = "<hp:ctrl><hp:footer><hp:subList>" + _hp("- 1 -") + "</hp:subList></hp:footer></hp:ctrl>"
    first = f"<hp:p><hp:run><hp:secPr/>{header}{footer}</hp:run><hp:run><hp:t>경계근무 규정</hp:t></hp:run></hp:p>"
    note = "<hp:ctrl><hp:footNote><hp:subList>" + _hp("1) 각주 내용이다.") + "</hp:subList></hp:footNote></hp:ctrl>"
    second = f"<hp:p><hp:run><hp:t>제1조(목적) 이 규정은 기준을 정한다.</hp:t>{note}</hp:run></hp:p>"
    third = _hp("제2조(정의) 용어의 뜻은 다음과 같다.")
    data = _hwpx(first + second + third)
    assert extract_hwpx(data) == ["경계근무 규정", "제1조(목적) 이 규정은 기준을 정한다.", "제2조(정의) 용어의 뜻은 다음과 같다."]
    p = tmp_path / "x.hwpx"
    p.write_bytes(data)
    doc = convert_file(p)
    assert doc.title == "경계근무 규정"
    assert doc.sections[0] == ("제1조(목적)", ["이 규정은 기준을 정한다."])


def test_hwpx_header_inside_table_paragraph_not_emitted():
    header = "<hp:ctrl><hp:header><hp:subList>" + _hp("머리말") + "</hp:subList></hp:header></hp:ctrl>"
    tbl = "<hp:tbl><hp:tr><hp:tc><hp:subList>" + _hp("a") + "</hp:subList></hp:tc></hp:tr></hp:tbl>"
    body = f"<hp:p><hp:run>{header}<hp:t>본문</hp:t>{tbl}</hp:run></hp:p>"
    assert extract_hwpx(_hwpx(body)) == ["본문", "| a |"]


def test_hwpx_nested_table_not_duplicated():
    inner = "<hp:tbl><hp:tr>" + "".join(f"<hp:tc><hp:subList>{_hp(c)}</hp:subList></hp:tc>" for c in ("내부1", "내부2")) + "</hp:tr></hp:tbl>"
    outer_cell2 = f"<hp:tc><hp:subList><hp:p><hp:run>{inner}</hp:run></hp:p></hp:subList></hp:tc>"
    outer = f"<hp:tbl><hp:tr><hp:tc><hp:subList>{_hp('외부A')}</hp:subList></hp:tc>{outer_cell2}</hp:tr></hp:tbl>"
    paras = extract_hwpx(_hwpx(f"<hp:p><hp:run>{outer}</hp:run></hp:p>"))
    assert paras == ["| 외부A |  |", "| 내부1 | 내부2 |"]


# ---- DOCX ---------------------------------------------------------------
def _wp(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def test_docx_nested_table_not_duplicated():
    inner = "<w:tbl><w:tr>" + "".join(f"<w:tc>{_wp(c)}</w:tc>" for c in ("내부1", "내부2")) + "</w:tr></w:tbl>"
    outer = f"<w:tbl><w:tr><w:tc>{_wp('외부A')}</w:tc><w:tc>{inner}{_wp('')}</w:tc></w:tr></w:tbl>"
    assert extract_docx(_docx(outer)) == ["| 외부A |  |", "| 내부1 | 내부2 |"]


def test_docx_table_rows_and_cells_in_sdt():
    tbl = f"<w:tbl><w:sdt><w:sdtContent><w:tr><w:tc>{_wp('가')}</w:tc><w:sdt><w:sdtContent><w:tc>{_wp('나')}</w:tc></w:sdtContent></w:sdt></w:tr></w:sdtContent></w:sdt></w:tbl>"
    assert extract_docx(_docx(tbl)) == ["| 가 | 나 |"]


def test_docx_alternate_content_fallback_not_duplicated():
    txbx = f"<w:txbxContent>{_wp('글상자 본문')}</w:txbxContent>"
    alt = f"<mc:AlternateContent><mc:Choice Requires=\"wps\"><w:drawing>{txbx}</w:drawing></mc:Choice><mc:Fallback><w:pict>{txbx}</w:pict></mc:Fallback></mc:AlternateContent>"
    body = f"<w:p><w:r>{alt}</w:r></w:p>"
    assert extract_docx(_docx(body)) == ["글상자 본문"]


# ---- PDF ----------------------------------------------------------------
def test_merge_pdf_lines_keeps_numeric_table_values():
    lines = ["| 구분 | 인원 |", "연번", "1", "소총수", "12", "", "2025", "", "3/4", "", "- 3 -"]
    merged = merge_pdf_lines(lines)
    assert merged == ["| 구분 | 인원 | 연번 1 소총수 12", "2025", "3/4"]


def test_strip_pdf_page_numbers_only_at_page_edges():
    pages = [["경계근무 규정", "연번", "1", "12", "- 1 -"], ["2", "본문", "3/4", "2 / 3"], ["끝 문단이다.", "2025"]]
    lines = strip_pdf_page_numbers(pages)
    assert [ln for ln in lines if ln] == ["경계근무 규정", "연번", "1", "12", "본문", "3/4", "끝 문단이다.", "2025"]


def _pdf(lines, *, encrypt=None) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    w = PdfWriter()
    page = w.add_blank_page(width=400, height=600)
    content = "BT /F1 12 Tf 40 560 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    page[NameObject("/Contents")] = w._add_object(stream)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    if encrypt:
        w.encrypt(encrypt, algorithm="RC4-128")
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def test_extract_pdf_keeps_mid_page_numbers():
    paras = extract_pdf(_pdf(["Personnel", "Rifleman", "12", "Total strength.", "3"]))
    assert "12" in " ".join(paras)
    assert not any(p.strip() == "3" for p in paras)


def test_encrypted_pdf_reports_korean_message():
    with pytest.raises(ExtractError, match="암호가 걸린 PDF"):
        extract_pdf(_pdf(["secret text"], encrypt="secret"))


# ---- 구조화 -------------------------------------------------------------
def test_chapter_requires_boundary_and_heading_form():
    paras = ["제1조(목적) 목적이다.", "제5조(보고) 보고한다.", "제2장의 규정을 준용한다.", "제3장비 점검", "제3절 제12조부터 제14조까지를 준용한다.", "제6조(보고) 보고한다."]
    names = [n for n, _ in _structure(paras, "article")]
    assert names == ["제1조(목적)", "제5조(보고)", "제6조(보고)"]
    assert _structure(["제2장 근무 편성", "제3조(편성) 편성한다."], "article")[0][0] == "제2장 근무 편성"


def test_article_branch_number_kept():
    secs = _structure(["제1장 총칙", "제5조(정의) 정의한다.", "제5조의2(정의의 특례) 특례를 둔다.", "제5조의 3 [특례] 둔다."], "article")
    assert [n for n, _ in secs] == ["제1장 총칙", "제1장 총칙 › 제5조(정의)", "제1장 총칙 › 제5조의2(정의의 특례)", "제1장 총칙 › 제5조의3(특례)"]


def test_article_reference_sentence_is_not_new_article():
    paras = ["제11조(지정) 지정한다.", "제12조(보고) 보고한다.", "제11조제2항에 따른 보고는 서면으로 한다.", "제6조에 따른 보고는 서면으로 한다.", "제5조 및 제6조에 따라 처리한다.", "제5조의 규정을 따른다."]
    secs = _structure(paras, "article")
    assert [n for n, _ in secs] == ["제11조(지정)", "제12조(보고)"]
    assert secs[1][1] == ["보고한다.", "제11조제2항에 따른 보고는 서면으로 한다.", "제6조에 따른 보고는 서면으로 한다.", "제5조 및 제6조에 따라 처리한다.", "제5조의 규정을 따른다."]
    assert _detect_mode(["교범", "제5조에 따라 지정된 부대는 보고한다.", "제6조에 따른 보고는 서면으로 한다."]) == "plain"
    # 라벨 없는 진짜 조문은 여전히 조문이다
    assert [n for n, _ in _structure(["제7조 삭제", "제8조 이 규정은 시행한다."], "article")] == ["제7조", "제8조"]


def test_numbered_heading_excludes_dates_and_measurements():
    assert _detect_mode(["개정 이력", "2024. 12. 31.", "2025. 3. 1.", "본문이다."]) == "plain"
    secs = _structure(["행군 계획", "1. 개요", "2.5 km 이동", "1.5 km 행군", "2. 절차", "2.1 GPS 점검"], "numbered")
    assert [n for n, _ in secs] == ["", "1. 개요", "2. 절차", "2.1 GPS 점검"]
    assert secs[1][1] == ["2.5 km 이동", "1.5 km 행군"]


def test_title_not_taken_from_structural_first_paragraph(tmp_path):
    p = tmp_path / "편성규정.txt"
    p.write_text("제1장 총칙\n제1조(목적) 목적이다.\n제2조(범위) 적용한다.\n제2장 편성\n제3조(편성) 편성한다.", encoding="utf-8")
    doc = convert_file(p)
    assert doc.title == "편성규정"
    assert [n for n, _ in doc.sections] == ["제1장 총칙", "제1장 총칙 › 제1조(목적)", "제1장 총칙 › 제2조(범위)", "제2장 편성", "제2장 편성 › 제3조(편성)"]
    q = tmp_path / "교범.txt"
    q.write_text("1. 개요\n개요다.\n2. 절차\n절차다.", encoding="utf-8")
    assert convert_file(q).title == "교범"


# ---- .md 머리말 재적재 ----------------------------------------------------
def test_reingest_markdown_with_front_matter(tmp_path):
    src = tmp_path / "기존.md"
    src.write_text("---\ntitle: 경계근무 규정\ncategory: 규정\neffective_date: 2025-03-01\nsource_file: 경계근무규정.hwp\ntags: 경계, 근무\n---\n# 경계근무 규정\n\n## 제1조(목적)\n\n이 규정은 기준을 정한다.\n", encoding="utf-8")
    assert extract_paragraphs(src)[0] == "# 경계근무 규정"
    doc = convert_file(src)
    assert doc.title == "경계근무 규정"
    assert doc.sections == [("제1조(목적)", ["이 규정은 기준을 정한다."])]
    meta, body = parse_front_matter(to_markdown(doc))
    assert meta.title == "경계근무 규정" and meta.category == "규정" and meta.effective_date == "2025-03-01"
    assert meta.source_file == "경계근무규정.hwp" and meta.tags == ("경계", "근무")
    assert body.startswith("# 경계근무 규정\n\n## 제1조(목적)\n\n이 규정은 기준을 정한다.")
    # CLI 에서 준 값은 기존 머리말보다 우선한다
    meta2, _ = parse_front_matter(to_markdown(doc, category="지침", tags=["새"]))
    assert meta2.category == "지침" and meta2.tags == ("새",) and meta2.effective_date == "2025-03-01"


def test_guess_title_skips_front_matter_like_lines(tmp_path):
    p = tmp_path / "구분선.md"
    p.write_text("---\n본문 첫 줄이다.\n", encoding="utf-8")
    assert convert_file(p).title == "구분선"


# ---- 출력 이름 충돌 -------------------------------------------------------
def test_ingest_output_name_collisions(tmp_path):
    src = tmp_path / "src"
    (src / "1대대").mkdir(parents=True)
    (src / "2대대").mkdir()
    (src / "1대대" / "경계지침.txt").write_text("1대대 지침\n1대대 본문이다.", encoding="utf-8")
    (src / "2대대" / "경계지침.txt").write_text("2대대 지침\n2대대 본문이다.", encoding="utf-8")
    (src / "보고서.md").write_text("보고서 md\nmd 본문이다.", encoding="utf-8")
    (src / "보고서.txt").write_text("보고서 txt\ntxt 본문이다.", encoding="utf-8")
    out = tmp_path / "docs"
    results = ingest_paths([src], out)
    assert all(r.ok and not r.skipped for r in results)
    outputs = [r.output for r in results]
    assert len(set(outputs)) == 4
    texts = {r.source.relative_to(src).as_posix(): r.output.read_text(encoding="utf-8") for r in results}
    assert "1대대 본문이다." in texts["1대대/경계지침.txt"] and "2대대 본문이다." in texts["2대대/경계지침.txt"]
    assert "md 본문이다." in texts["보고서.md"] and "txt 본문이다." in texts["보고서.txt"]
    assert sorted(p.name for p in out.glob("*.md")) == sorted(["경계지침.md", "2대대_경계지침.md", "보고서.md", "보고서_txt.md"])
    # 다시 실행하면 같은 이름이 배정되어 모두 '이미 있음' 으로 건너뛴다
    again = ingest_paths([src], out)
    assert [r.output for r in again] == outputs and all(r.skipped for r in again)
    forced = ingest_paths([src], out, overwrite=True)
    assert [r.output for r in forced] == outputs and not any(r.skipped for r in forced)


def test_ingest_same_file_twice_is_converted_once(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("가 문서\n본문이다.", encoding="utf-8")
    results = ingest_paths([f, f], tmp_path / "docs")
    assert len(results) == 1 and results[0].ok
