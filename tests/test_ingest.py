import io
import struct
import zipfile
import zlib

import pytest

from defense_assistant.ingest import convert_file, extract_paragraphs, ingest_paths, to_markdown
from defense_assistant.ingest.convert import _detect_mode, _structure
from defense_assistant.ingest.extract import ExtractError, _extract_hwp_from_ole, extract_docx, extract_hwpx, merge_pdf_lines, parse_hwp_bodytext
from defense_assistant.knowledge import DocumentStore, parse_front_matter

REGULATION = """경계근무 규정
제1장 총칙
제1조(목적) 이 규정은 경계근무의 기준을 정함을 목적으로 한다.
제2조(적용 범위) 모든 예하 부대에 적용한다.
다만, 특수 임무 부대는 별도 지침을 따른다.
제2장 근무 편성
제3조(근무 형태) 다음 각 호와 같이 편성한다.
1. 주간 근무: 2인 1조
2. 야간 근무: 1시간 교대
부칙
이 규정은 2025년 3월 1일부터 시행한다."""

MANUAL = """응급처치 교범
1. 개요
교전 상황에서 부상자를 처치하는 절차다.
2. 평가 순서
2.1 대량 출혈
지혈대를 적용하고 시각을 기록한다.
3. 후송
9-Line 양식으로 요청한다."""


def _docx_bytes(paragraphs, table=None) -> bytes:
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row) + "</w:tr>" for row in table)
        body += f"<w:tbl>{rows}</w:tbl>"
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _hwpx_bytes(paragraphs, table=None) -> bytes:
    HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
    body = "".join(f"<hp:p><hp:run><hp:t>{p}</hp:t></hp:run></hp:p>" for p in paragraphs)
    if table:
        rows = "".join("<hp:tr>" + "".join(f"<hp:tc><hp:subList><hp:p><hp:run><hp:t>{c}</hp:t></hp:run></hp:p></hp:subList></hp:tc>" for c in row) + "</hp:tr>" for row in table)
        body += f"<hp:p><hp:run><hp:tbl>{rows}</hp:tbl></hp:run></hp:p>"
    xml = f'<?xml version="1.0" encoding="UTF-8"?><hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" xmlns:hp="{HP}">{body}</hs:sec>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", xml)
    return buf.getvalue()


def _hwp_record(tag: int, payload: bytes) -> bytes:
    header = tag | (0 << 10) | (len(payload) << 20)
    return struct.pack("<I", header) + payload


def _para_text_payload(text: str, with_extended_ctrl: bool = False) -> bytes:
    data = b""
    if with_extended_ctrl:
        data += struct.pack("<H", 3) + b"\x00" * 14  # 8글자짜리 확장 제어 문자(필드 시작 등)
    data += text.encode("utf-16-le") + struct.pack("<H", 13)  # 문단 끝
    return data


def _pdf_bytes(lines: list[str]) -> bytes:
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
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


class FakeOle:
    def __init__(self, sections: dict[str, bytes], flags: int = 1):
        self._streams = {"FileHeader": b"\x00" * 36 + struct.pack("<I", flags) + b"\x00" * 216}
        self._streams.update({f"BodyText/{k}": v for k, v in sections.items()})

    def listdir(self):
        return [n.split("/") for n in self._streams]

    def openstream(self, name):
        return io.BytesIO(self._streams[name])


# ---- 추출 -------------------------------------------------------------
def test_extract_docx_with_table():
    paras = extract_docx(_docx_bytes(["제1조(목적) 목적이다.", "", "본문"], table=[["구분", "내용"], ["주간", "2인 1조"]]))
    assert paras == ["제1조(목적) 목적이다.", "", "본문", "| 구분 | 내용 |", "| 주간 | 2인 1조 |"]
    with pytest.raises(ExtractError):
        extract_docx(b"not a zip")


def test_extract_hwpx_with_table():
    paras = extract_hwpx(_hwpx_bytes(["경계근무 규정", "제1조(목적) 목적."], table=[["구분", "내용"], ["야간", "1시간 교대"]]))
    assert paras[:2] == ["경계근무 규정", "제1조(목적) 목적."]
    assert "| 구분 | 내용 |" in paras and "| 야간 | 1시간 교대 |" in paras
    with pytest.raises(ExtractError):
        extract_hwpx(b"zip?")


def test_parse_hwp_bodytext_and_ole():
    stream = _hwp_record(0x10 + 50, b"\x00" * 8)  # PARA_HEADER(무시)
    stream += _hwp_record(0x10 + 51, _para_text_payload("제1조(목적) 이 규정은", with_extended_ctrl=True))
    stream += _hwp_record(0x10 + 51, _para_text_payload("둘째 문단\t탭"))
    stream += _hwp_record(0x10 + 52, b"\x01\x02")  # PARA_CHAR_SHAPE(무시)
    paras = parse_hwp_bodytext(stream)
    assert paras == ["제1조(목적) 이 규정은 ", "둘째 문단 탭 "]
    compressed = zlib.compress(stream)[2:-4]  # raw deflate
    ole = FakeOle({"Section0": compressed, "Section1": zlib.compress(_hwp_record(0x10 + 51, _para_text_payload("두 번째 섹션")))[2:-4]})
    assert [p.strip() for p in _extract_hwp_from_ole(ole)] == ["제1조(목적) 이 규정은", "둘째 문단 탭", "두 번째 섹션"]
    assert [p.strip() for p in _extract_hwp_from_ole(FakeOle({"Section0": stream}, flags=0))][0] == "제1조(목적) 이 규정은"
    with pytest.raises(ExtractError, match="암호"):
        _extract_hwp_from_ole(FakeOle({"Section0": stream}, flags=3))
    with pytest.raises(ExtractError, match="배포용"):
        _extract_hwp_from_ole(FakeOle({"Section0": stream}, flags=5))
    with pytest.raises(ExtractError, match="본문"):
        _extract_hwp_from_ole(FakeOle({}))


def test_extract_hwp_rejects_non_ole(tmp_path):
    p = tmp_path / "x.hwp"
    p.write_bytes(b"HWP Document File V3.00")
    with pytest.raises(ExtractError, match="HWP 5.0"):
        extract_paragraphs(p)


def test_extract_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(_pdf_bytes(["Article 1 Purpose", "This regulation sets the", "standard for guard duty.", "2", "Article 2 Scope"]))
    paras = extract_paragraphs(p)
    assert paras[0].startswith("Article 1 Purpose")
    assert any("sets the standard for guard duty." in x for x in paras)  # 줄 병합
    assert not any(x == "2" for x in paras)  # 쪽 번호 제거


def test_merge_pdf_lines():
    lines = ["제1조(목적) 이 규정은", "기준을 정함을 목적으로 한다.", "제2조(범위) 적용한다.", "", "- 3 -", "다음 문단이다."]
    assert merge_pdf_lines(lines) == ["제1조(목적) 이 규정은 기준을 정함을 목적으로 한다.", "제2조(범위) 적용한다.", "다음 문단이다."]


def test_unsupported_extension(tmp_path):
    p = tmp_path / "a.xlsx"
    p.write_bytes(b"")
    with pytest.raises(ExtractError, match="지원하지 않는"):
        extract_paragraphs(p)


# ---- 구조화 -----------------------------------------------------------
def test_article_mode_structure(tmp_path):
    p = tmp_path / "경계근무규정.txt"
    p.write_text(REGULATION, encoding="utf-8")
    doc = convert_file(p)
    assert doc.title == "경계근무 규정" and doc.mode == "article"
    names = [n for n, _ in doc.sections]
    assert names == ["제1장 총칙", "제1장 총칙 › 제1조(목적)", "제1장 총칙 › 제2조(적용 범위)", "제2장 근무 편성", "제2장 근무 편성 › 제3조(근무 형태)", "부칙"]
    assert doc.sections[2][1] == ["모든 예하 부대에 적용한다.", "다만, 특수 임무 부대는 별도 지침을 따른다."]
    assert doc.sections[4][1] == ["다음 각 호와 같이 편성한다.", "1. 주간 근무: 2인 1조", "2. 야간 근무: 1시간 교대"]


def test_numbered_mode_structure(tmp_path):
    p = tmp_path / "교범.txt"
    p.write_text(MANUAL, encoding="utf-8")
    doc = convert_file(p, title="전투 응급처치 교범")
    assert doc.title == "전투 응급처치 교범" and doc.mode == "numbered"
    assert [n for n, _ in doc.sections] == ["", "1. 개요", "2. 평가 순서", "2.1 대량 출혈", "3. 후송"]
    assert doc.sections[0][1] == ["응급처치 교범"]  # --title 을 줬으므로 첫 줄은 본문으로 남는다


def test_plain_mode_and_markdown_headings():
    assert _detect_mode(["그냥 문단", "또 문단"]) == "plain"
    secs = _structure(["머리말", "## 둘째 절", "본문 1.", "# 또 제목", "본문 2."], "plain")
    assert secs == [("", ["머리말"]), ("둘째 절", ["본문 1."]), ("또 제목", ["본문 2."])]


def test_to_markdown_roundtrips_into_store(tmp_path):
    p = tmp_path / "경계근무규정.txt"
    p.write_text(REGULATION, encoding="utf-8")
    md = to_markdown(convert_file(p), category="규정", effective_date="2025-03-01", tags=["경계"])
    meta, body = parse_front_matter(md)
    assert meta.title == "경계근무 규정" and meta.category == "규정" and meta.effective_date == "2025-03-01" and meta.source_file == "경계근무규정.txt"
    assert body.startswith("# 경계근무 규정\n\n## 제1장 총칙")
    s = DocumentStore(embedder=False)
    s.add_document(md, source="r.md")
    assert s.search("야간 근무 교대", top_k=1)[0].chunk.ref == "경계근무 규정 › 제2장 근무 편성 › 제3조(근무 형태)"
    assert s.categories == ["규정"]


def test_ingest_paths_folder_skip_overwrite_dry_run(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "규정.txt").write_text(REGULATION, encoding="utf-8")
    (src / "교범.docx").write_bytes(_docx_bytes(["교범", "1. 개요", "내용이다.", "2. 절차", "절차다."]))
    (src / "표.hwpx").write_bytes(_hwpx_bytes(["표 문서", "본문"], table=[["a", "b"]]))
    (src / "무시.xlsx").write_bytes(b"")
    (src / "깨짐.pdf").write_bytes(b"%PDF-1.4 garbage")
    out = tmp_path / "docs"
    results = ingest_paths([src], out, category="규정", dry_run=True)
    assert not out.exists() and sum(1 for r in results if r.ok) == 3 and sum(1 for r in results if r.error) == 1
    results = ingest_paths([src], out, category="규정")
    outputs = sorted(p.name for p in out.glob("*.md"))
    assert outputs == ["교범.md", "규정.md", "표.md"]
    assert "| a | b |" in (out / "표.md").read_text(encoding="utf-8")
    again = ingest_paths([src / "규정.txt"], out)
    assert again[0].skipped
    forced = ingest_paths([src / "규정.txt"], out, overwrite=True, title="새 제목")
    assert not forced[0].skipped and (out / "규정.md").read_text(encoding="utf-8").startswith("---\ntitle: 새 제목")
    missing = ingest_paths([tmp_path / "없음.hwp"], out)
    assert missing[0].error == "파일이 없습니다."
    store = DocumentStore.from_directory(out, embedder=False)
    assert "새 제목" in store.titles and "교범" in store.titles and "표 문서" in store.titles
