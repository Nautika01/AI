import json
from types import SimpleNamespace

import pytest

from defense_assistant.evaluation import EvalReport, compare_reports, evaluate_answers, evaluate_retrieval, load_cases
from tests.conftest import ROOT, make_message


def test_load_cases_validation(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('# 주석\n{"question": "a", "expected_refs": ["x"]}\n\n{"id": "q2", "question": "b"}\n', encoding="utf-8")
    cases = load_cases(p)
    assert [c.id for c in cases] == ["q001", "q2"] and cases[0].expected_refs == ["x"]
    p.write_text('{"id": "a", "question": "x"}\n{"id": "a", "question": "y"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        load_cases(p)
    p.write_text('{"nope": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_cases(p)
    p.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_cases(p)
    p.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="질문이 없습니다"):
        load_cases(p)


def test_sample_question_set_retrieval(store):
    cases = load_cases(ROOT / "data" / "eval" / "questions.jsonl")
    r = evaluate_retrieval(store, cases, k=4)
    assert r["cases"] == 18 and r["hit_at_k"] >= 0.9 and r["mrr"] >= 0.85
    skipped = [x for x in r["results"] if x["skipped"]]
    assert {x["id"] for x in skipped} == {"q14", "q15", "q19", "q20"}  # 도구 질문·차단 질문은 검색 평가 제외


def test_retrieval_metrics_math(store):
    cases = load_cases(ROOT / "data" / "eval" / "questions.jsonl")[:1]
    cases[0].expected_refs = ["존재하지 않는 섹션"]
    r = evaluate_retrieval(store, cases, k=2)
    assert r["hit_at_k"] == 0.0 and r["mrr"] == 0.0 and len(r["misses"]) == 1 and len(r["misses"][0]["got"]) == 2


def test_evaluate_answers_with_stub():
    cases = load_cases(ROOT / "data" / "eval" / "questions.jsonl")
    chosen = [c for c in cases if c.id in {"q01", "q14", "q19"}]

    def ask(q: str):
        if "비밀" in q:
            return SimpleNamespace(text="차단", blocked=True, tools_called=[], usage={})
        if "071430" in q:
            return SimpleNamespace(text="070530ZOCT26 입니다.", blocked=False, tools_called=["convert_military_time"], usage={"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3500})
        return SimpleNamespace(text="정지 명령 후 보고합니다. 사살은 안 됩니다.", blocked=False, tools_called=["search_defense_docs"], usage={"input_tokens": 20, "output_tokens": 8, "cache_read_input_tokens": None})

    seen = []
    a = evaluate_answers(ask, chosen, on_progress=lambda i, n, r: seen.append((i, n, r.id)))
    assert seen == [(1, 3, "q01"), (2, 3, "q14"), (3, 3, "q19")]
    by_id = {r["id"]: r for r in a["results"]}
    assert by_id["q19"]["passed"] and by_id["q19"]["blocked"]
    assert by_id["q14"]["passed"] and by_id["q14"]["keywords_hit"] == ["070530Z"]
    assert not by_id["q01"]["passed"] and by_id["q01"]["keywords_missed"] == ["신원"] and by_id["q01"]["forbidden_found"] == ["사살"]
    assert a["pass_rate"] == round(2 / 3, 4) and a["block_accuracy"] == 1.0 and a["forbidden_violations"] == 1
    assert a["total_input_tokens"] == 30 and a["errors"] == 0
    # 캐시 읽기 입력(시스템 프롬프트·도구 정의)은 input_tokens 와 별도로 합산해 보고한다.
    assert a["total_cache_read_tokens"] == 3500 and by_id["q14"]["cache_read_tokens"] == 3500
    line = next(x for x in EvalReport.build("q.jsonl", "m", None, a).summary_lines() if x.startswith("[답변]"))
    assert "토큰 입력 30 (+캐시 읽기 3500) / 출력 13" in line


def test_evaluate_answers_handles_errors_and_wrong_block():
    cases = load_cases(ROOT / "data" / "eval" / "questions.jsonl")
    chosen = [c for c in cases if c.id in {"q05", "q20"}]

    def ask(q: str):
        if "TOP SECRET" in q:
            return SimpleNamespace(text="여기 있습니다", blocked=False, tools_called=[], usage={})
        raise RuntimeError("모델 서버 다운")

    a = evaluate_answers(ask, chosen)
    assert a["errors"] == 1 and a["block_accuracy"] == 0.0 and a["pass_rate"] == 0.0
    assert any("모델 서버 다운" in f["error"] for f in a["failures"] if f["error"])


def test_report_save_load_compare(tmp_path, store):
    cases = load_cases(ROOT / "data" / "eval" / "questions.jsonl")
    r1 = evaluate_retrieval(store, cases, k=4)
    a1 = evaluate_answers(lambda q: SimpleNamespace(text="", blocked=("비밀" in q or "SECRET" in q), tools_called=[], usage={}), cases)
    rep1 = EvalReport.build("q.jsonl", "stub", r1, a1)
    path = rep1.save(tmp_path / "r1.json")
    loaded = EvalReport.load(path)
    assert loaded.retrieval["hit_at_k"] == r1["hit_at_k"] and loaded.answers["pass_rate"] == a1["pass_rate"]
    lines = loaded.summary_lines()
    assert lines[0].startswith("평가 시각") and any("[검색]" in ln for ln in lines) and any("[답변]" in ln for ln in lines)
    # 두 번째 실행: 모든 키워드를 담은 답변 → 개선
    a2 = evaluate_answers(lambda q: SimpleNamespace(text=" ".join(kw.split("|")[0] for c in cases for kw in c.expected_keywords), blocked=("비밀" in q or "SECRET" in q), tools_called=[], usage={}), cases)
    rep2 = EvalReport.build("q.jsonl", "stub", r1, a2)
    out = "\n".join(compare_reports(loaded, rep2))
    assert "답변 통과율" in out and "답변 개선:" in out and "퇴보" not in out
    assert "검색 hit@k" in out


def test_cli_eval_runs(tmp_path, monkeypatch):
    from defense_assistant.cli import main

    monkeypatch.setenv("DAI_INDEX_CACHE_DIR", str(tmp_path / "idx"))
    out = tmp_path / "r.json"
    assert main(["--root", str(ROOT), "eval", "--limit", "5", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["retrieval"]["cases"] <= 5 and data["answers"] is None
    assert main(["--root", str(ROOT), "eval", "--limit", "5", "--compare", str(out)]) == 0
    assert main(["--root", str(ROOT), "eval", "--cases", str(tmp_path / "없음.jsonl")]) == 1


def test_cli_eval_with_answers_uses_assistant(tmp_path, monkeypatch, make_assistant):
    """--answers 는 비서를 실제로 호출한다. 가짜 모델로 전체 경로만 확인한다."""
    import defense_assistant.cli as cli

    assistant, _ = make_assistant([make_message("정지 명령, 신원 확인, 보고")])
    monkeypatch.setattr(cli, "DefenseAssistant", lambda settings, store=None: assistant)
    monkeypatch.setenv("DAI_INDEX_CACHE_DIR", str(tmp_path / "idx"))
    out = tmp_path / "a.json"
    assert cli.main(["--root", str(ROOT), "eval", "--answers", "--category", "경계", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["answers"]["cases"] == 4 and data["model"] == "claude-opus-5-5"
    assert {r["id"]: r["passed"] for r in data["answers"]["results"]}["q01"] is True
