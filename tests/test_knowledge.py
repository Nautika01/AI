from defense_assistant.knowledge import DocumentStore, tokenize


def test_tokenize_korean_bigrams_and_particles():
    toks = tokenize("경계근무는 OPORD 5단락")
    assert "경계근무" in toks and "경계" in toks and "근무" in toks
    assert "opord" in toks and "5" in toks
    assert "경계근무는" not in toks


def test_tokenize_transliteration():
    assert "medevac" in tokenize("메데박 요청")
    assert "sitrep" in tokenize("시트렙 양식")


def test_store_loads_sample_docs(store):
    assert len(store) > 20
    assert any("보고서" in t for t in store.titles)


def test_search_relevance(store):
    cases = {
        "9라인 메데박 항목": "9-Line MEDEVAC",
        "SALUTE 보고 형식": "SPOTREP",
        "야간 경계 요령": "야간 경계 요령",
        "작전명령 5단락": "OPORD",
        "DTG 형식": "DTG",
        "계급 체계": "계급 체계",
    }
    for query, expected in cases.items():
        hits = store.search(query, top_k=1)
        assert hits, query
        assert expected in hits[0].chunk.ref, (query, hits[0].chunk.ref)


def test_search_empty_and_format(store):
    assert store.search("") == []
    assert store.format_hits([]) == "관련 문서를 찾지 못했습니다."
    out = store.format_hits(store.search("지혈대", top_k=2))
    assert out.startswith("[1] ") and "출처:" in out


def test_add_document_splits_long_sections():
    s = DocumentStore()
    body = "# 긴 문서\n\n## 섹션\n\n" + "\n\n".join(f"문단 {i} " + "가나다라 " * 60 for i in range(8))
    n = s.add_document(body, source="long.md")
    assert n >= 2
    assert s.chunks[0].section.startswith("섹션")
