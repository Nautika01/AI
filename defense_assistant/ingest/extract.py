"""문서 파일에서 문단 텍스트를 뽑아낸다.

지원 형식: .hwp(한글 5.0 바이너리), .hwpx(한글 XML), .docx, .pdf, .txt, .md
표는 `| 셀 | 셀 |` 형태의 한 줄로 바꿔 검색·인용이 가능하게 한다. 그림·수식·각주는 버린다.
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

SUPPORTED_EXTENSIONS = (".hwp", ".hwpx", ".docx", ".pdf", ".txt", ".md")


class ExtractError(RuntimeError):
    pass


def extract_paragraphs(path: Path | str) -> list[str]:
    """파일에서 문단 목록을 돌려준다. 표의 각 행은 `| a | b |` 한 문단으로 들어온다."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _clean(path.read_text(encoding="utf-8", errors="replace").splitlines())
    if ext == ".docx":
        return _clean(extract_docx(path.read_bytes()))
    if ext == ".hwpx":
        return _clean(extract_hwpx(path.read_bytes()))
    if ext == ".hwp":
        return _clean(extract_hwp(path))
    if ext == ".pdf":
        return _clean(extract_pdf(path.read_bytes()))
    raise ExtractError(f"지원하지 않는 형식입니다: {path.suffix} (지원: {', '.join(SUPPORTED_EXTENSIONS)})")


def _clean(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        p = re.sub(r"[ \t 　]+", " ", p).strip()
        if p:
            out.append(p)
    return out


# ---- DOCX ---------------------------------------------------------------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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
            for tr in el.iter(f"{_W}tr"):
                cells = []
                for tc in tr.findall(f"{_W}tc"):
                    cells.append(" ".join(_docx_para_text(p) for p in tc.iter(f"{_W}p")).strip())
                if any(cells):
                    out.append("| " + " | ".join(cells) + " |")
        elif el.tag == f"{_W}sdt":  # 구조화 문서 태그 안의 내용
            content = el.find(f"{_W}sdtContent")
            if content is not None:
                out.extend(_walk_docx(content))
    return out


def _docx_para_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag in (f"{_W}tab",):
            parts.append(" ")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append(" ")
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


def _walk_hwpx(root: ET.Element) -> list[str]:
    out: list[str] = []

    def para_text(p: ET.Element) -> str:
        parts: list[str] = []
        for node in p.iter():
            if _local(node.tag) == "t" and node.text:
                parts.append(node.text)
            elif _local(node.tag) == "lineBreak":
                parts.append(" ")
        return "".join(parts)

    def visit(el: ET.Element) -> None:
        for child in el:
            tag = _local(child.tag)
            if tag == "tbl":
                for tr in child.iter():
                    if _local(tr.tag) != "tr":
                        continue
                    cells = []
                    for tc in tr:
                        if _local(tc.tag) == "tc":
                            cells.append(" ".join(para_text(p) for p in tc.iter() if _local(p.tag) == "p").strip())
                    if any(cells):
                        out.append("| " + " | ".join(cells) + " |")
            elif tag == "p":
                # 문단 안의 표·그림 개체는 따로 처리하고, 표 안의 문단은 위에서 이미 다뤘다
                if any(_local(d.tag) == "tbl" for d in child.iter()):
                    text = "".join((n.text or "") for n in child.iter() if _local(n.tag) == "t" and not _inside_tbl(child, n))
                    if text.strip():
                        out.append(text)
                    visit(child)
                else:
                    out.append(para_text(child))
            else:
                visit(child)

    visit(root)
    return out


def _inside_tbl(para: ET.Element, node: ET.Element) -> bool:
    for tbl in para.iter():
        if _local(tbl.tag) == "tbl" and any(n is node for n in tbl.iter()):
            return True
    return False


# ---- HWP 5.0 (바이너리) --------------------------------------------------
_HWPTAG_PARA_TEXT = 0x10 + 51
# 제어 문자 중 1글자(2바이트)만 차지하는 것: 탭(9), 줄바꿈(10), 문단끝(13), 24~31
# 나머지 제어 문자(1~8, 11, 12, 14~23)는 8글자(16바이트)를 차지한다
_HWP_SINGLE_CTRL = {9, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}


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


def _decode_para_text(payload: bytes) -> str:
    chars: list[str] = []
    i = 0
    n = len(payload) - (len(payload) % 2)
    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        if code < 32:
            if code in (9, 10):
                chars.append(" ")
            elif code == 13:
                chars.append(" ")
            i += 2 if code in _HWP_SINGLE_CTRL else 16
            continue
        chars.append(chr(code))
        i += 2
    return "".join(chars)


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
                reader.decrypt("")
            except Exception as e:  # noqa: BLE001
                raise ExtractError("암호가 걸린 PDF 입니다. 암호를 해제한 뒤 다시 시도하십시오.") from e
        lines: list[str] = []
        for page in reader.pages:
            lines.extend((page.extract_text() or "").splitlines())
            lines.append("")
    except ExtractError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ExtractError(f"PDF 를 읽을 수 없습니다: {e}") from e
    if not any(ln.strip() for ln in lines):
        raise ExtractError("PDF 에서 글자를 찾지 못했습니다. 스캔 이미지 PDF 라면 OCR 이 필요합니다.")
    return merge_pdf_lines(lines)


_SENTENCE_END = re.compile(r"[.。:：;!?)\]」』”\"']\s*$|(?:다|음|함|됨|것|임|요)\.?\s*$")
_HEADING_START = re.compile(r"^(제\s*\d+\s*[조장절관편]|\d+(?:\.\d+)*[.)]\s|[가-힣][.)]\s|[①-⑳]|[IVX]+\.\s|부칙|별표|별지|\|)")


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
        if re.fullmatch(r"-?\s*\d+\s*-?|\d+\s*/\s*\d+", line):  # 쪽 번호
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
