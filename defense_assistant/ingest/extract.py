"""문서 파일에서 문단 텍스트를 뽑아낸다.

지원 형식: .hwp(한글 5.0 바이너리), .hwpx(한글 XML), .docx, .pdf, .txt, .md
표는 `| 셀 | 셀 |` 형태의 한 줄로 바꿔 검색·인용이 가능하게 한다. 그림·수식·각주·머리말·꼬리말은 버린다.
.md/.txt 맨 위의 머리말(front matter)은 본문에서 떼어 내고, 그 메타데이터는 `read_front_matter` 로 따로 읽는다.
모든 추출은 로컬에서만 이루어지며 외부 서비스를 쓰지 않는다.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from ..knowledge.metadata import parse_front_matter

SUPPORTED_EXTENSIONS = (".hwp", ".hwpx", ".docx", ".pdf", ".txt", ".md")


class ExtractError(RuntimeError):
    pass


def extract_paragraphs(path: Path | str) -> list[str]:
    """파일에서 문단 목록을 돌려준다. 표의 각 행은 `| a | b |` 한 문단으로 들어온다."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        _, body = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        return _clean(body.splitlines())
    if ext == ".docx":
        return _clean(extract_docx(path.read_bytes()))
    if ext == ".hwpx":
        return _clean(extract_hwpx(path.read_bytes()))
    if ext == ".hwp":
        return _clean(extract_hwp(path))
    if ext == ".pdf":
        return _clean(extract_pdf(path.read_bytes()))
    raise ExtractError(f"지원하지 않는 형식입니다: {path.suffix} (지원: {', '.join(SUPPORTED_EXTENSIONS)})")


def read_front_matter(path: Path | str) -> dict:
    """.md/.txt 맨 위 머리말(front matter)의 메타데이터. 머리말이 없거나 다른 형식이면 빈 dict."""
    path = Path(path)
    if path.suffix.lower() not in (".txt", ".md"):
        return {}
    meta, _ = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
    return meta.to_dict()


def _clean(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        p = re.sub(r"[ \t 　]+", " ", p).strip()
        if p:
            out.append(p)
    return out


# ---- DOCX ---------------------------------------------------------------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# 글상자 등은 mc:Choice 와 mc:Fallback(VML) 에 같은 글자가 두 번 들어 있으므로 Fallback 은 읽지 않는다
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def extract_docx(data: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        raise ExtractError("DOCX 파일을 열 수 없습니다 (손상되었거나 암호가 걸린 파일).") from e
    root = ET.fromstring(xml)
    body = root.find(f"{_W}body")
    if body is None:
        return []
    return _walk_docx(body)


def _walk_docx(parent: ET.Element) -> list[str]:
    out: list[str] = []
    for el in parent:
        if el.tag == f"{_W}p":
            out.append(_docx_para_text(el))
        elif el.tag == f"{_W}tbl":
            _docx_table(el, out)
        elif el.tag == f"{_W}sdt":  # 구조화 문서 태그 안의 내용
            content = el.find(f"{_W}sdtContent")
            if content is not None:
                out.extend(_walk_docx(content))
    return out


def _docx_blocks(el: ET.Element, tags: tuple[str, ...]):
    """el 의 직속 자식 중 tags 에 속한 것. 구조화 문서 태그(sdt)로 감싼 것도 풀어서 돌려준다."""
    for c in el:
        if c.tag in tags:
            yield c
        elif c.tag == f"{_W}sdt":
            content = c.find(f"{_W}sdtContent")
            if content is not None:
                yield from _docx_blocks(content, tags)


def _docx_table(tbl: ET.Element, out: list[str]) -> None:
    """표의 각 행을 한 줄로 낸다. 셀 안의 표(중첩 표)는 셀 글자에 섞지 않고 그 행 다음에 따로 낸다."""
    for tr in _docx_blocks(tbl, (f"{_W}tr",)):
        cells: list[str] = []
        nested: list[ET.Element] = []
        for tc in _docx_blocks(tr, (f"{_W}tc",)):
            texts: list[str] = []
            for block in _docx_blocks(tc, (f"{_W}p", f"{_W}tbl")):
                if block.tag == f"{_W}p":
                    texts.append(_docx_para_text(block))
                else:
                    nested.append(block)
            cells.append(" ".join(texts).strip())
        if any(cells):
            out.append("| " + " | ".join(cells) + " |")
        for t in nested:
            _docx_table(t, out)


def _docx_para_text(p: ET.Element) -> str:
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        for node in el:
            if node.tag == _MC_FALLBACK:  # mc:Choice 와 같은 내용의 대체본
                continue
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag in (f"{_W}tab", f"{_W}br", f"{_W}cr"):
                parts.append(" ")
            else:
                walk(node)

    walk(p)
    return "".join(parts)


# ---- HWPX ---------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_hwpx(data: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = sorted((n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n)), key=lambda n: int(re.search(r"(\d+)", n).group(1)))
            if not names:
                raise ExtractError("HWPX 안에 본문(Contents/section*.xml)이 없습니다.")
            out: list[str] = []
            for n in names:
                out.extend(_walk_hwpx(ET.fromstring(z.read(n))))
            return out
    except zipfile.BadZipFile as e:
        raise ExtractError("HWPX 파일을 열 수 없습니다 (손상되었거나 암호가 걸린 파일).") from e


# 머리말·꼬리말·각주·미주·숨은 설명은 호스트 문단의 hp:ctrl 아래에 들어 있으므로 본문에 섞지 않는다
_HWPX_SKIP = frozenset({"header", "footer", "footNote", "endNote", "hiddenComment", "secPr"})
# hp:t 안에 자식 요소로 들어오는 공백류. 그 뒤 글자는 자식의 .tail 에 있다
_HWPX_SPACE = frozenset({"tab", "lineBreak", "nbSpace", "fwSpace"})


def _hwpx_t_text(t: ET.Element) -> str:
    """hp:t 의 혼합 콘텐츠(text + 자식 요소 + 각 자식의 tail)를 문서 순서대로 잇는다."""
    parts = [t.text or ""]
    for c in t:
        tag = _local(c.tag)
        if tag in _HWPX_SPACE:
            parts.append(" ")
        elif tag not in _HWPX_SKIP:
            parts.append(_hwpx_t_text(c))  # 형광펜(markpenBegin/End) 같은 표시 요소는 빈 글자
        parts.append(c.tail or "")
    return "".join(parts)


def _hwpx_para_text(p: ET.Element, tables: list[ET.Element] | None = None) -> str:
    """문단 글자. 문단 안의 표는 글자에 넣지 않고 tables 에 모은다. 머리말·각주 등은 버린다."""
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        for c in el:
            tag = _local(c.tag)
            if tag in _HWPX_SKIP:
                continue
            if tag == "tbl":
                if tables is not None:
                    tables.append(c)
            elif tag == "t":
                parts.append(_hwpx_t_text(c))
            elif tag in _HWPX_SPACE:
                parts.append(" ")
            else:
                walk(c)

    walk(p)
    return "".join(parts)


def _hwpx_cell_paras(el: ET.Element):
    """셀 안의 최상위 문단들. 중첩 표 안의 문단은 빼고, 문단 안의 문단(글상자 등)은 그 문단이 함께 다룬다."""
    for c in el:
        tag = _local(c.tag)
        if tag == "p":
            yield c
        elif tag != "tbl" and tag not in _HWPX_SKIP:
            yield from _hwpx_cell_paras(c)


def _walk_hwpx(root: ET.Element) -> list[str]:
    out: list[str] = []

    def table(tbl: ET.Element) -> None:
        # 이 표에 직접 속한 행만 돈다. 중첩 표는 셀 글자에 섞지 않고 그 행 다음에 따로 낸다
        for tr in tbl:
            if _local(tr.tag) != "tr":
                continue
            cells: list[str] = []
            nested: list[ET.Element] = []
            for tc in tr:
                if _local(tc.tag) == "tc":
                    cells.append(" ".join(_hwpx_para_text(p, nested) for p in _hwpx_cell_paras(tc)).strip())
            if any(cells):
                out.append("| " + " | ".join(cells) + " |")
            for t in nested:
                table(t)

    def visit(el: ET.Element) -> None:
        for child in el:
            tag = _local(child.tag)
            if tag in _HWPX_SKIP:
                continue
            if tag == "tbl":
                table(child)
            elif tag == "p":
                # 문단 안의 표는 문단 글자 다음에 따로 낸다
                tables: list[ET.Element] = []
                text = _hwpx_para_text(child, tables)
                if tables:
                    if text.strip():
                        out.append(text)
                    for t in tables:
                        table(t)
                else:
                    out.append(text)
            else:
                visit(child)

    visit(root)
    return out


# ---- HWP 5.0 (바이너리) --------------------------------------------------
_HWPTAG_PARA_TEXT = 0x10 + 51
# HWP 5.0 스펙: 1글자(2바이트)만 차지하는 char 형 제어 문자는 0, 줄바꿈(10), 문단끝(13), 24~31 뿐이다.
# 탭(9)을 포함한 나머지 인라인·확장 제어 문자(1~9, 11, 12, 14~23)는 8글자(16바이트)를 차지한다
_HWP_SINGLE_CTRL = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}


def extract_hwp(path: Path) -> list[str]:
    try:
        import olefile
    except ImportError as e:  # pragma: no cover
        raise ExtractError("HWP 변환에는 olefile 패키지가 필요합니다: pip install olefile") from e
    if not olefile.isOleFile(str(path)):
        raise ExtractError("HWP 5.0 형식이 아닙니다. 한글에서 '다른 이름으로 저장 → HWPX' 로 바꾸거나 PDF 로 내보내 주십시오.")
    ole = olefile.OleFileIO(str(path))
    try:
        return _extract_hwp_from_ole(ole)
    finally:
        ole.close()


def _extract_hwp_from_ole(ole) -> list[str]:
    try:
        header = ole.openstream("FileHeader").read()
    except OSError as e:
        raise ExtractError("HWP 파일 헤더를 읽을 수 없습니다.") from e
    flags = struct.unpack("<I", header[36:40])[0]
    compressed = bool(flags & 0x1)
    if flags & 0x2:
        raise ExtractError("암호가 걸린 HWP 파일입니다. 암호를 해제한 뒤 다시 시도하십시오.")
    if flags & 0x4:
        raise ExtractError("배포용(DRM) HWP 문서는 본문을 읽을 수 없습니다.")
    sections = sorted((e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")), key=lambda e: int(re.sub(r"\D", "", e[1]) or 0))
    if not sections:
        raise ExtractError("HWP 안에 본문(BodyText) 이 없습니다.")
    out: list[str] = []
    for entry in sections:
        raw = ole.openstream("/".join(entry)).read()
        if compressed:
            raw = zlib.decompress(raw, -15)
        out.extend(parse_hwp_bodytext(raw))
    return out


def parse_hwp_bodytext(data: bytes) -> list[str]:
    """BodyText 섹션 스트림의 레코드를 훑어 PARA_TEXT 의 글자만 모은다."""
    out: list[str] = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        header = struct.unpack("<I", data[pos : pos + 4])[0]
        tag = header & 0x3FF
        size = (header >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            if pos + 4 > n:
                break
            size = struct.unpack("<I", data[pos : pos + 4])[0]
            pos += 4
        payload = data[pos : pos + size]
        pos += size
        if tag == _HWPTAG_PARA_TEXT:
            text = _decode_para_text(payload)
            if text.strip():
                out.append(text)
    return out


_UTF16_SPACE = " ".encode("utf-16-le")


def _decode_para_text(payload: bytes) -> str:
    # 글자 바이트를 모아 한 번에 풀어야 UTF-16 서로게이트 쌍(확장 한자 등)이 한 글자로 합쳐진다
    buf = bytearray()
    i = 0
    n = len(payload) - (len(payload) % 2)
    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        if code < 32:
            if code in (9, 10, 13):  # 탭·줄바꿈·문단끝
                buf += _UTF16_SPACE
            i += 2 if code in _HWP_SINGLE_CTRL else 16
            continue
        buf += payload[i : i + 2]
        i += 2
    return bytes(buf).decode("utf-16-le", errors="replace")


# ---- PDF ----------------------------------------------------------------
def extract_pdf(data: bytes) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ExtractError("PDF 변환에는 pypdf 패키지가 필요합니다: pip install pypdf") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")  # 실패해도 예외 없이 NOT_DECRYPTED(0) 을 돌려준다
            except Exception as e:  # noqa: BLE001
                raise ExtractError("암호가 걸린 PDF 입니다. 암호를 해제한 뒤 다시 시도하십시오.") from e
            if not decrypted:
                raise ExtractError("암호가 걸린 PDF 입니다. 암호를 해제한 뒤 다시 시도하십시오.")
        lines = strip_pdf_page_numbers([(page.extract_text() or "").splitlines() for page in reader.pages])
    except ExtractError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ExtractError(f"PDF 를 읽을 수 없습니다: {e}") from e
    if not any(ln.strip() for ln in lines):
        raise ExtractError("PDF 에서 글자를 찾지 못했습니다. 스캔 이미지 PDF 라면 OCR 이 필요합니다.")
    return merge_pdf_lines(lines)


_SENTENCE_END = re.compile(r"[.。:：;!?)\]」』”\"']\s*$|(?:다|음|함|됨|것|임|요)\.?\s*$")
_HEADING_START = re.compile(r"^(제\s*\d+\s*[조장절관편]|\d+(?:\.\d+)*[.)]\s|[가-힣][.)]\s|[①-⑳]|[IVX]+\.\s|부칙|별표|별지|\|)")


_PAGE_NUMBER_DECORATED = re.compile(r"[-–—]\s*\d{1,4}\s*[-–—]")  # "- 3 -" 꼴은 어디에 있어도 쪽 번호로 본다
_PAGE_NUMBER_EDGE = re.compile(r"[-–—]?\s*(\d{1,4})\s*[-–—]?|\d{1,4}\s*/\s*\d{1,4}|(?:[Pp]age|[Pp]\.)\s*\d{1,4}|\d{1,4}\s*쪽")


def _is_edge_page_number(line: str, page_count: int) -> bool:
    m = _PAGE_NUMBER_EDGE.fullmatch(line)
    if not m:
        return False
    if m.group(1) is not None and not re.search(r"[-–—]", line):
        # 장식 없는 숫자는 쪽 수보다 훨씬 크면(연도 등) 쪽 번호로 보지 않는다
        return int(m.group(1)) <= page_count + 100
    return True


def strip_pdf_page_numbers(pages: list[list[str]]) -> list[str]:
    """쪽마다 맨 첫 줄·맨 끝 줄의 쪽 번호만 지우고, 쪽 사이에 빈 줄을 넣어 한 목록으로 잇는다.

    쪽 중간에 있는 숫자만 있는 줄('1', '12', '2025', '3/4' 같은 표 셀 값)은 남긴다.
    """
    out: list[str] = []
    for page_lines in pages:
        lines = list(page_lines)
        filled = [i for i, ln in enumerate(lines) if ln.strip()]
        for i in sorted({filled[0], filled[-1]}) if filled else ():
            if _is_edge_page_number(lines[i].strip(), len(pages)):
                lines[i] = ""
        out.extend(lines)
        out.append("")
    return out


def merge_pdf_lines(lines: list[str]) -> list[str]:
    """PDF 는 줄 단위로 끊겨 나오므로, 문장이 끝나지 않은 줄은 다음 줄과 이어 붙인다."""
    out: list[str] = []
    buf = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if _PAGE_NUMBER_DECORATED.fullmatch(line):  # 쪽 번호. 숫자만 있는 줄은 표 값일 수 있어 남긴다
            continue
        if buf and not _HEADING_START.match(line) and not _SENTENCE_END.search(buf):
            buf = f"{buf} {line}"
        else:
            if buf:
                out.append(buf)
            buf = line
    if buf:
        out.append(buf)
    return out
