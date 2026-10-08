import json

import pytest

from defense_assistant.knowledge import DocMeta, DocumentStore, HashEmbedder, OpenAIEmbedder, SynonymMap, parse_front_matter, render_front_matter
from defense_assistant.knowledge.embeddings import EmbeddingCache, VectorIndex


def test_front_matter_roundtrip():
    content = "---\ntitle: 경계근무 규정\ncategory: 규정\neffective_date: 2025-03-01\nversion: 3.1\ntags: 경계, 근무\nunit: 1사단\n---\n# 무시되는 제목\n\n## 제1조\n본문"
    meta, body = parse_front_matter(content)
    assert meta.title == "경계근무 규정" and meta.category == "규정" and meta.tags == ("경계", "근무") and meta.extra == {"unit": "1사단"}
    assert body.startswith("# 무시되는 제목")
    assert parse_front_matter("# 제목\n본문")[0] == DocMeta()
    rendered = render_front_matter(meta.to_dict())
    assert rendered.startswith("---\ntitle: 경계근무 규정\n") and "tags: 경계, 근무" in rendered and "unit: 1사단" in rendered
    assert parse_front_matter(rendered + "## a\nb")[0].category == "규정"


def test_store_uses_front_matter_and_filters():
    s = DocumentStore(embedder=False)
    s.add_document("---\ntitle: 경계 규정\ncategory: 규정\ntags: 경계\n---\n## 제1조\n경계 근무는 2인 1조로 한다.", source="a.md", doc_id="a")
    s.add_document("# 경계 교범\n## 요령\n경계 근무 요령은 다음과 같다.", source="b.md", doc_id="b")
    assert s.titles == ["경계 규정", "경계 교범"] and s.categories == ["규정"]
    assert [h.chunk.doc_id for h in s.search("경계 근무", top_k=5)] == ["a", "b"] or len(s.search("경계 근무", top_k=5)) == 2
    assert [h.chunk.doc_id for h in s.search("경계 근무", category="규정")] == ["a"]
    assert [h.chunk.doc_id for h in s.search("경계 근무", tag="경계")] == ["a"]
    assert [h.chunk.doc_id for h in s.search("경계 근무", doc_id="b")] == ["b"]
    assert s.search("경계 근무", category="없음") == []
    docs = s.documents()
    assert docs[0]["meta"] == {"title": "경계 규정", "category": "규정", "tags": ["경계"]} and docs[0]["chunks"] == 1
    out = s.format_hits(s.search("경계", top_k=1, category="규정"))
    assert "규정" in out and "[1] 경계 규정 › 제1조" in out


def test_synonym_map_expand_and_merge(tmp_path):
    m = SynonymMap({"경계근무": ["보초", "불침번"], "MEDEVAC": ["메데박"]})
    assert len(m) == 2
    assert m.expand("보초 설 때") == "보초 설 때 경계근무 불침번"
    assert m.expand("아무 관계 없는 질문") == "아무 관계 없는 질문"
    assert m.expand("medevac 요청") == "medevac 요청 메데박"  # 대소문자 무시
    m.add_group(["불침번", "야간 근무"])  # 기존 그룹과 합쳐진다
    assert len(m) == 2 and "야간 근무" in m.expand("보초")
    p = tmp_path / "syn.json"
    p.write_text(json.dumps({"a": ["b"]}), encoding="utf-8")
    assert len(SynonymMap.load(p)) == 1 and len(SynonymMap.load(tmp_path / "none.json")) == 0
    p.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ValueError):
        SynonymMap.load(p)


def test_synonyms_improve_recall(store):
    plain = DocumentStore.from_directory(store.chunks[0].source and "data/docs", embedder=False)
    assert "거수자" not in plain.search("보초 설 때 수상한 사람", top_k=1)[0].chunk.section
    assert "거수자" in store.search("보초 설 때 수상한 사람", top_k=1)[0].chunk.section


def test_hash_embedder_and_vector_index():
    e = HashEmbedder(dim=64)
    v1, v2, v3 = e.embed(["경계 근무 교대", "경계근무 교대 절차", "지혈대 적용"])
    assert len(v1) == 64 and abs(sum(x * x for x in v1) - 1.0) < 1e-6
    idx = VectorIndex()
    idx.add([v1, v2, v3])
    top = idx.top(v1, 3)
    assert top[0][0] == 0 and top[1][0] == 1 and top[2][0] == 2  # 비슷한 문장이 먼저


def test_embedding_cache_persists(tmp_path):
    class Counting(HashEmbedder):
        name = "counting"
        calls = 0

        def embed(self, texts):
            Counting.calls += len(texts)
            return super().embed(texts)

    c = EmbeddingCache(tmp_path, "counting")
    c.get_many(["가", "나"], Counting())
    c.save()
    assert Counting.calls == 2 and (tmp_path / "embeddings-counting.json").exists()
    c2 = EmbeddingCache(tmp_path, "counting")
    c2.get_many(["가", "나", "다"], Counting())
    assert Counting.calls == 3  # '다' 만 새로 계산


def test_store_vectors_cached_between_instances(tmp_path):
    kw = dict(cache_dir=tmp_path)
    s1 = DocumentStore.from_directory("data/docs", **kw)
    files = list(tmp_path.glob("embeddings-*.json"))
    assert len(files) == 1 and len(json.loads(files[0].read_text())) == len(s1)
    s2 = DocumentStore.from_directory("data/docs", **kw)
    assert s2.search("지혈대", top_k=1)[0].vector_rank is not None


def test_hybrid_hits_carry_ranks(store):
    hits = store.search("작전명령 5단락", top_k=3)
    assert hits[0].bm25_rank == 1 and hits[0].vector_rank is not None and hits[0].score > hits[1].score
    assert DocumentStore(embedder=False).search("x") == []


def test_openai_embedder_against_fake_server():
    from fastapi import FastAPI
    from tests.fake_openai_server import run_in_thread

    app = FastAPI()
    seen = []

    @app.post("/v1/embeddings")
    def emb(body: dict):
        seen.append(body)
        return {"data": [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]}

    base, stop = run_in_thread(app)
    try:
        e = OpenAIEmbedder(base, "fake-embed", api_key="k", batch_size=2)
        vecs = e.embed(["a", "bb", "ccc"])
        assert len(vecs) == 3 and len(seen) == 2 and seen[0]["model"] == "fake-embed"
        assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-6 and e.weight == 0.8
        s = DocumentStore(embedder=e)
        s.add_document("# t\n## s\n본문", source="t.md")
        s.ensure_vectors()
        assert len(s._vectors) == 1
    finally:
        stop()


def test_build_store_modes(tmp_path):
    from defense_assistant.config import Settings, build_store

    base = dict(docs_dir="data/docs", synonyms_path="data/synonyms.json", index_cache_dir=tmp_path)
    assert build_store(Settings(embeddings="off", **base)).embedder is None
    assert isinstance(build_store(Settings(embeddings="hash", **base)).embedder, HashEmbedder)
    s = build_store(Settings(embeddings="openai", embeddings_model="m", embeddings_base_url="http://127.0.0.1:9/v1", **base))
    assert s.embedder is None and "임베딩 계산 실패" in s.embedding_error  # 서버가 없으면 BM25 만으로 계속 동작
    assert s.search("지혈대", top_k=1)[0].vector_rank is None
    with pytest.raises(ValueError):
        Settings(embeddings="bert")
