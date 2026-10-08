"""교차 검토에서 확인된 CLI·평가 결함의 회귀 테스트."""

import json
import shutil
from types import SimpleNamespace

import pytest

from defense_assistant.cli import _build_settings, build_parser, main
from defense_assistant.evaluation import EvalReport, compare_reports, evaluate_answers, load_cases
from defense_assistant.evaluation.runner import _keyword_present
from tests.conftest import ROOT


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in ("DAI_BACKEND", "DAI_MODEL", "DAI_LOCAL_MODEL", "DAI_DOCS_DIR", "DAI_EFFORT", "DAI_FALLBACKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DAI_INDEX_CACHE_DIR", str(tmp_path / "idx"))


# ---- 전역 옵션을 하위 명령 뒤에 써도 동작 -----------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--user", "홍길동", "-v"],
        ["search", "거수자", "--model", "m", "--backend", "local"],
        ["eval", "--effort", "low", "--no-fallback", "--root", "/tmp"],
        ["users", "add", "kim", "-v"],
        ["local-check", "--model", "m"],
    ],
)
def test_global_options_after_subcommand(argv):
    args = build_parser().parse_args(argv)
    for opt, attr in (("-v", "verbose"), ("--model", "model"), ("--backend", "backend"), ("--effort", "effort"), ("--no-fallback", "no_fallback"), ("--root", "root")):
        if opt in argv:
            assert getattr(args, attr) not in (None, False)


def test_global_options_before_subcommand_not_overwritten():
    """하위 파서의 기본값이 앞에 준 전역 옵션을 덮어쓰면 안 된다."""
    args = build_parser().parse_args(["-v", "--model", "m", "--backend", "local", "--root", "/r", "--effort", "low", "--no-fallback", "chat"])
    assert args.verbose is True and args.model == "m" and args.backend == "local" and args.root == "/r" and args.effort == "low" and args.no_fallback is True
    args = build_parser().parse_args(["chat"])
    assert args.verbose is False and args.model is None and args.backend is None and args.root is None and args.no_fallback is False


# ---- local-check 의 --model 은 로컬 모델에 적용 ------------------------------
def test_local_check_model_option_targets_local_model(clean_env, monkeypatch, capsys):
    import defense_assistant.backends.local as local_mod
    from defense_assistant.backends import BackendError

    seen = {}

    class FakeLocal:
        def __init__(self, settings, *a, **k):
            seen["local_model"] = settings.local_model
            seen["backend"] = settings.backend

        def check(self):
            raise BackendError("테스트 중단")

    monkeypatch.setattr(local_mod, "LocalBackend", FakeLocal)
    assert main(["--root", str(ROOT), "--model", "exaone3.5:7.8b", "local-check"]) == 1
    assert seen == {"local_model": "exaone3.5:7.8b", "backend": "local"}


def test_build_settings_backend_override_keeps_other_commands(clean_env):
    args = build_parser().parse_args(["--root", str(ROOT), "--model", "claude-x", "search", "q"])
    s = _build_settings(args)
    assert s.backend == "claude" and s.model == "claude-x" and s.local_model == "qwen2.5:7b"


# ---- --compare 에 보고서가 아닌 JSON ---------------------------------------
@pytest.mark.parametrize("content", ["{}", "[]", '{"created_at": "x"}', '"문자열"', '{"created_at": "x", "cases_file": "q", "retrieval": [1]}'])
def test_eval_report_load_rejects_non_report(tmp_path, content):
    p = tmp_path / "bad.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="평가 보고서 형식이 아닙니다"):
        EvalReport.load(p)


def test_cli_eval_compare_with_non_report_json(tmp_path, clean_env, capsys):
    bad = tmp_path / "other.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["--root", str(ROOT), "eval", "--limit", "2", "--compare", str(bad)]) == 1
    assert "비교 대상을 읽을 수 없습니다" in capsys.readouterr().err
    # 형식은 맞지만 내부 필드가 빠진 보고서도 트레이스백 없이 종료
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"created_at": "x", "cases_file": "q", "model": "m", "retrieval": {"k": 4}, "answers": None}), encoding="utf-8")
    assert main(["--root", str(ROOT), "eval", "--limit", "2", "--compare", str(partial)]) == 1
    assert "비교 대상을 읽을 수 없습니다" in capsys.readouterr().err


# ---- MRR·정수 지표 변화량 표시 --------------------------------------------
def _report(mrr, forbidden=0, pass_rate=0.5):
    retrieval = {"k": 4, "cases": 1, "hit_at_k": 1.0, "hit_at_1": 1.0, "mrr": mrr, "misses": [], "results": []}
    answers = {"pass_rate": pass_rate, "keyword_rate": 0.5, "forbidden_violations": forbidden, "results": []}
    return EvalReport("t", "q", "m", retrieval, answers)


def test_compare_reports_delta_precision():
    out = "\n".join(compare_reports(_report(0.9722, forbidden=1), _report(0.9306, forbidden=3)))
    assert "(-0.0416)" in out and "(-0.0)" not in out
    assert "(+2)" in out
    out = "\n".join(compare_reports(_report(0.9306, pass_rate=0.5), _report(0.9722, pass_rate=0.75)))
    assert "(+0.0416)" in out and "(+25.0%p)" in out


# ---- 차단 실패 사유 ------------------------------------------------------
def test_summary_shows_reason_when_block_expected_but_not_blocked():
    cases = [c for c in load_cases(ROOT / "data" / "eval" / "questions.jsonl") if c.id == "q20"]
    a = evaluate_answers(lambda q: SimpleNamespace(text="여기 있습니다", blocked=False, tools_called=[], usage={}), cases)
    lines = EvalReport("t", "q", "m", None, a).summary_lines()
    line = next(ln for ln in lines if "q20" in ln)
    assert "차단 안 됨" in line and "()" not in line


# ---- 한 글자 키워드 ------------------------------------------------------
def test_single_char_ascii_keyword_requires_boundary():
    assert not _keyword_present("2025년 기준 10개 항목", "1")
    assert not _keyword_present("Time zone 문자를 붙입니다", "I|India")
    assert not _keyword_present("Time zone 문자를 붙입니다", "Z|Zulu")
    assert _keyword_present("1~5번을 먼저 보냅니다", "1") and _keyword_present("1~5번을 먼저 보냅니다", "5")
    assert _keyword_present("한국은 I(India), 협정세계시는 Z", "I|India") and _keyword_present("한국은 I(India), 협정세계시는 Z", "Z|Zulu")
    assert _keyword_present("UTC+9는 I입니다", "I")
    # 여러 글자 키워드·한글 한 글자는 기존처럼 부분 문자열
    assert _keyword_present("070530ZOCT26 입니다", "070530Z")
    assert _keyword_present("적군 상황", "적")


def test_sample_set_q08_q16_judged_by_content():
    cases = {c.id: c for c in load_cases(ROOT / "data" / "eval" / "questions.jsonl")}
    # 숫자 한 글자만으로 된 키워드는 답변 품질과 무관하게 맞거나 틀린다
    for c in cases.values():
        for kw in c.expected_keywords:
            assert not all(len(alt.strip()) == 1 and alt.strip().isdigit() for alt in kw.split("|")), (c.id, kw)

    def run(case_id, text):
        return evaluate_answers(lambda q: SimpleNamespace(text=text, blocked=False, tools_called=[], usage={}), [cases[case_id]])["results"][0]["passed"]

    assert not run("q08", "2025년 기준 10개 항목을 모두 보내야 합니다.")
    assert run("q08", "9라인 메데박의 첫 다섯 항목(착륙 지점 위치, 무전 주파수·호출부호, 환자 수와 긴급도, 특수 장비, 환자 유형)을 먼저 보내야 합니다.")
    assert not run("q16", "DTG는 날짜와 시간을 쓰는 형식입니다. Time zone 문자를 붙입니다.")
    assert run("q16", "DTG 시간대 문자는 한국 표준시 I(India), 협정세계시 Z(Zulu) 입니다.")
    assert run("q16", "시간대 문자: 한국 표준시는 I, 협정세계시는 Z 를 씁니다.")


# ---- BOM ---------------------------------------------------------------
def test_load_cases_and_report_accept_bom(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "a", "question": "x"}\n', encoding="utf-8-sig")
    assert [c.id for c in load_cases(p)] == ["a"]
    r = tmp_path / "r.json"
    r.write_text(json.dumps({"created_at": "t", "cases_file": "q", "model": "m", "retrieval": None, "answers": None}), encoding="utf-8-sig")
    assert EvalReport.load(r).created_at == "t"


# ---- 기본 질문셋 경로 ------------------------------------------------------
def test_eval_default_cases_falls_back_to_root_data_eval(tmp_path, clean_env, monkeypatch, capsys):
    kb = tmp_path / "elsewhere" / "kb"
    shutil.copytree(ROOT / "data" / "docs", kb)
    monkeypatch.setenv("DAI_DOCS_DIR", str(kb))
    assert main(["--root", str(ROOT), "eval", "--limit", "2"]) == 0
    assert str(ROOT / "data" / "eval" / "questions.jsonl") in capsys.readouterr().out


def test_eval_default_cases_prefers_set_next_to_docs_dir(tmp_path, clean_env, monkeypatch, capsys):
    kb = tmp_path / "kb" / "docs"
    shutil.copytree(ROOT / "data" / "docs", kb)
    own = tmp_path / "kb" / "eval" / "questions.jsonl"
    own.parent.mkdir()
    own.write_text('{"id": "x1", "question": "거수자 조치", "expected_refs": ["거수자"]}\n', encoding="utf-8")
    monkeypatch.setenv("DAI_DOCS_DIR", str(kb))
    assert main(["--root", str(ROOT), "eval"]) == 0
    out = capsys.readouterr().out
    assert "질문 1개" in out and str(own) in out
