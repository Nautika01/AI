"""교차 검토에서 확인된 지식 베이스(knowledge) 결함의 회귀 테스트."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from defense_assistant.knowledge import DocumentStore, SynonymMap, parse_front_matter, tokenize
from defense_assistant.knowledge.store import _MAX_CHUNK_CHARS

ROOT = Path(__file__).resolve().parent.parent
BOM = "﻿"


class _FlakyEmbedder:
    """색인 때는 정상, 이후 `down=True` 면 서버 장애처럼 예외를 던지는 스텁 임베더."""

    name = "flaky"
    weight = 0.8

    def __init__(self) -> None:
        self.down = False
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.down:
            raise ConnectionError("[Errno 111] Connection refused")
        return [[1.0, 0.0] for _ in texts]


class _FixedEmbedder:
    """텍스트마다 미리 정한 벡터를 돌려주는 의미 임베더 스텁 (기본은 질의와 같은 방향)."""

    name = "fixed"
    weight = 0.8

    def __init__(self, table: dict[str, list[float]] | None = None, default: list[float] | None = None) -> None:
        self.table = table or {}
        self.default = default or [1.0, 0.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = next((vec for key, vec in self.table.items() if key in t), self.default)
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


# ---- 1. 질의 시점 임베딩 서버 장애 → BM25 로 계속 동작 ------------------------------
def test_query_embedding_failure_falls_back_to_bm25():
    emb = _FlakyEmbedder()
    s = DocumentStore(embedder=emb)
    s.add_document("# 경계 교범\n## 야간 경계\n야간 경계는 2인 1조로 운용한다.", source="a.md")
    s.add_document("# 응급처치\n## 지혈대\n지혈대는 상처 위쪽에 감는다.", source="b.md")
    assert s.search("지혈대", top_k=1)[0].vector_rank is not None  # 정상일 때는 하이브리드
    emb.down = True
    hits = s.search("지혈대", top_k=1)  # 예외가 전파되면 안 된다
    assert hits and hits[0].chunk.doc_id == "b.md" and hits[0].vector_rank is None
    assert s.embedding_error and "질의 임베딩 실패" in s.embedding_error
    assert s.embedder is emb  # 복구 후 재시도할 수 있도록 임베더는 유지
    # 장애 직후에는 매 검색마다 연결 대기를 반복하지 않는다
    calls = emb.calls
    assert s.search("야간 경계", top_k=1)[0].chunk.doc_id == "a.md"
    assert emb.calls == calls
    # 재시도 시각이 지나고 서버가 복구되면 다시 벡터 검색을 쓰고 오류 표시를 지운다
    emb.down = False
    s._embed_retry_at = 0.0
    assert s.search("지혈대", top_k=1)[0].vector_rank is not None
    assert s.embedding_error is None


def test_index_failure_message_not_cleared_by_query_recovery():
    """색인 실패(임베더 해제)의 사유는 질의 경로가 지우지 않는다."""
    emb = _FlakyEmbedder()
    emb.down = True
    s = DocumentStore(embedder=emb)
    s.add_document("# t\n## s\n본문 내용", source="t.md")
    s.ensure_vectors()
    assert s.embedder is None and "임베딩 계산 실패" in s.embedding_error
    assert s.search("본문", top_k=1)
    assert "임베딩 계산 실패" in s.embedding_error


# ---- 2. synonyms.json 의 UTF-8 BOM / 깨진 JSON ---------------------------------------
def test_synonyms_load_with_bom(tmp_path):
    p = tmp_path / "syn.json"
    p.write_bytes(BOM.encode("utf-8") + json.dumps({"경계근무": ["보초"]}, ensure_ascii=False).encode("utf-8"))
    m = SynonymMap.load(p)
    assert len(m) == 1 and "경계근무" in m.expand("보초")


def test_synonyms_load_broken_json_warns_and_continues(tmp_path, caplog):
    p = tmp_path / "syn.json"
    p.write_text('{"경계근무": ["보초",', encoding="utf-8")
    with caplog.at_level("WARNING"):
        m = SynonymMap.load(p)
    assert len(m) == 0
    assert any("synonyms" in r.getMessage() or "동의어" in r.getMessage() for r in caplog.records)


def test_bundled_synonyms_file_loads():
    assert len(SynonymMap.load(ROOT / "data" / "synonyms.json")) > 10


# ---- 3. 음차 '마치' → 'march' 오탐 ---------------------------------------------------
@pytest.mark.parametrize("q", ["마치 실전처럼 훈련하는 방법", "교육을 마치면 어떻게 보고하나", "훈련을 마치고 복귀 보고 요령"])
def test_common_word_machi_is_not_transliterated(q):
    assert "march" not in tokenize(q)


def test_machi_queries_do_not_pull_march_chunk(store):
    hits = store.search("마치 실전처럼 훈련하는 방법", top_k=1)
    assert not hits or "MARCH" not in hits[0].chunk.ref
    # 영문 'MARCH' 질의는 여전히 해당 섹션을 찾는다
    assert "MARCH" in store.search("MARCH 순서", top_k=1)[0].chunk.ref


def test_other_transliterations_still_work():
    assert "medevac" in tokenize("메데박을 요청")
    assert "medevac" in tokenize("메데박요청 절차")
    assert "sitrep" in tokenize("시트렙 양식")


# ---- 4. 하이픈·점·슬래시로 이어진 영숫자 -----------------------------------------------
def test_tokenize_hyphenated_latin_runs():
    toks = tokenize("K-9 자주포")
    assert {"k-9", "k9", "9"} <= set(toks)
    assert {"mett-tc", "mett", "tc", "metttc"} <= set(tokenize("METT-TC 분석"))
    assert {"9-line", "9line", "line"} <= set(tokenize("9-Line MEDEVAC"))
    assert {"pcc", "pci"} <= set(tokenize("PCC/PCI 점검"))
    date = tokenize("2025-03-01 시행")
    assert "2025-03-01" in date and "2025" in date
    assert "31" not in tokenize("버전 3.1")  # 숫자끼리의 결합형은 만들지 않는다


def test_k9_query_matches_hyphenated_doc():
    s = DocumentStore(embedder=False)
    s.add_document(
        "# 장비 제원\n## K-9 자주포\nK-9 자주포 제원: 155mm 곡사포.\n## K-2 전차\nK-2 전차 제원: 120mm 활강포.\n## F-35 전투기\nF-35 전투기 제원: 스텔스 전투기.",
        source="eq.md",
    )
    assert "K-9" in s.search("K9 제원", top_k=1)[0].chunk.section
    assert "K-2" in s.search("K2 제원", top_k=1)[0].chunk.section


def test_mett_query_finds_mett_tc_section(store):
    refs = [h.chunk.ref for h in store.search("METT 요소", top_k=3)]
    assert any("METT-TC" in r for r in refs), refs


# ---- 5. 벡터 후보가 1개/동점일 때 정규화 -----------------------------------------------
def test_single_vector_candidate_not_dropped():
    s = DocumentStore(embedder=_FixedEmbedder())
    s.add_document("# 경계 지침\n## 운용\n야간 2인 1조로 운용한다.", source="a.md")
    hits = s.search("저녁 사람 배치", top_k=1)  # 어휘 겹침 없음, 의미 벡터만 일치
    assert hits and hits[0].chunk.doc_id == "a.md" and hits[0].vector_rank == 1


def test_tied_vector_candidates_and_filter_not_dropped():
    s = DocumentStore(embedder=_FixedEmbedder())
    s.add_document("---\ncategory: 교범\n---\n# 경계 지침\n## 운용\n야간 2인 1조.", source="a.md")
    s.add_document("# 기타\n## 운용\n주간 근무.", source="b.md")
    assert s.search("저녁 사람 배치", top_k=2)  # 두 청크 동점
    hits = s.search("저녁 사람 배치", top_k=2, category="교범")  # 필터 뒤 1개
    assert [h.chunk.doc_id for h in hits] == ["a.md"]


def test_zero_query_vector_does_not_return_everything():
    """질의 벡터가 0(모든 코사인 0, 동점)이면 벡터 신호가 없으므로 아무것도 끌어올리지 않는다."""
    s = DocumentStore(embedder=_FixedEmbedder(table={"zzz": [0.0, 0.0]}, default=[1.0, 0.0]))
    s.add_document("# a\n## s\n경계 근무", source="a.md")
    s.add_document("# b\n## s\n지혈대", source="b.md")
    assert s.search("zzz", top_k=2) == []


# ---- 6. 머리말·본문의 UTF-8 BOM ---------------------------------------------------------
def test_bom_without_front_matter_keeps_h1_title():
    meta, body = parse_front_matter(BOM + "# 경계 교범\n\n## 요령\n내용")
    assert body.startswith("# 경계 교범")
    s = DocumentStore(embedder=False)
    s.add_document(BOM + "# 경계 교범\n\n## 요령\n내용", source="b.md", doc_id="b")
    assert s.chunks[0].title == "경계 교범"
    assert all("# 경계 교범" not in c.text and BOM not in c.text for c in s.chunks)


def test_bom_front_matter_without_trailing_newline():
    meta, body = parse_front_matter(BOM + "---\ntitle: T\n---")
    assert meta.title == "T" and body == ""
    meta, body = parse_front_matter(BOM + "---\ntitle: T\n---\n# 제목\n본문")
    assert meta.title == "T" and body == "# 제목\n본문"


def test_from_directory_reads_bom_file(tmp_path):
    (tmp_path / "경계교범.md").write_bytes((BOM + "# 경계 교범\n\n## 요령\n초병은 근무한다.").encode("utf-8"))
    s = DocumentStore.from_directory(tmp_path, embedder=False)
    assert s.chunks[0].title == "경계 교범"


# ---- 7. 빈 줄 없는 긴 섹션의 분할 -------------------------------------------------------
def test_long_section_without_blank_lines_is_split():
    items = "\n".join(f"- 항목 {i}: 장비 {i}번 점검 내용 " + "가나다라마 " * 10 for i in range(40))
    s = DocumentStore(embedder=False)
    n = s.add_document(f"# 정비 점검표\n## 점검 항목\n{items}", source="m.md")
    assert n >= 2
    assert all(len(c.text) <= _MAX_CHUNK_CHARS for c in s.chunks)
    # 원문 줄이 빠짐없이 보존된다
    assert sum(c.text.count("- 항목 ") for c in s.chunks) == 40
    hit = s.search("항목 39 장비 39번", top_k=1)
    assert "항목 39" in s.format_hits(hit)


def test_single_very_long_line_is_split():
    s = DocumentStore(embedder=False)
    s.add_document("# 긴 줄\n## 본문\n" + "가" * (_MAX_CHUNK_CHARS * 2 + 10), source="l.md")
    assert len(s.chunks) == 3 and all(len(c.text) <= _MAX_CHUNK_CHARS for c in s.chunks)
    assert "".join(c.text for c in s.chunks) == "가" * (_MAX_CHUNK_CHARS * 2 + 10)


# ---- 8. 두 번째 이후 H1 보존 -------------------------------------------------------------
def test_later_h1_lines_are_kept_as_sections():
    s = DocumentStore(embedder=False)
    s.add_document("# 경계근무 지침\n## 개요\n소개 문장.\n# 제2장 경계근무\n초소 운영 내용.\n# 제3장 순찰\n순찰 내용.", source="g.md")
    assert s.chunks[0].title == "경계근무 지침"
    sections = [c.section for c in s.chunks]
    assert sections == ["개요", "제2장 경계근무", "제3장 순찰"]
    assert "제2장" in s.search("제2장 경계근무", top_k=1)[0].chunk.section
    assert all("경계근무 지침" not in c.text for c in s.chunks)  # 제목 줄은 본문에서 빠진다


def test_h1_comment_inside_code_fence_is_kept():
    s = DocumentStore(embedder=False)
    s.add_document("# 운용 안내\n## 재시작\n```\n# 서버 재시작 방법\nsystemctl restart x\n```", source="c.md")
    assert "# 서버 재시작 방법" in s.chunks[0].text
    assert s.chunks[0].title == "운용 안내"


def test_format_hits_shows_window_around_query_terms():
    lines = "\n".join(f"- 항목 {i}: 장비 {i}번 점검" + " 가나다라마" * 8 for i in range(16))
    s = DocumentStore(embedder=False)
    s.add_document(f"# 점검표\n## 목록\n{lines}\n- 냉각팬 베어링 윤활", source="w.md")
    assert len(s.chunks) == 1 and 900 < len(s.chunks[0].text) <= _MAX_CHUNK_CHARS
    out = s.format_hits(s.search("냉각팬 베어링", top_k=1))
    assert "냉각팬 베어링" in out and "… " in out
    # 질의어가 앞부분에 있으면 기존처럼 앞에서부터 보여 준다
    out = s.format_hits(s.search("항목 0 장비", top_k=1))
    assert "- 항목 0:" in out
