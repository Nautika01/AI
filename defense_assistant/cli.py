"""명령줄 인터페이스.

    defense-ai                 # 대화 (REPL)
    defense-ai chat --user 홍길동
    defense-ai search "거수자 조치"
    defense-ai serve --port 8000
    defense-ai local-check     # 로컬 모델 서버(DAI_BACKEND=local) 연결 점검
    defense-ai ingest 규정.hwp 교범폴더/ --category 규정   # HWP/HWPX/DOCX/PDF → data/docs/*.md
    defense-ai eval                      # 검색 평가 (모델 호출 없음)
    defense-ai eval --answers --out r.json --compare prev.json   # 답변까지 평가하고 이전과 비교
    defense-ai users add kim --role user --clearance RESTRICTED
    defense-ai users list | passwd kim | disable kim | enable kim | remove kim
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .assistant import AssistantError, DefenseAssistant
from .config import Settings, build_store

HELP = """명령:
  /help           도움말
  /reset          대화 기록 초기화
  /docs <검색어>   지식 베이스 직접 검색
  /status         설정·세션 상태
  /exit, /quit    종료"""


def _build_settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(root=Path(args.root).resolve() if args.root else None)
    if getattr(args, "backend", None):
        s.backend = args.backend
    if getattr(args, "model", None):
        if s.backend == "local":
            s.local_model = args.model
        else:
            s.model = args.model
    if getattr(args, "effort", None):
        s.effort = args.effort
    if getattr(args, "no_fallback", False):
        s.fallbacks = "off"
    s.__post_init__()
    return s


def run_chat(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    assistant = DefenseAssistant(settings)
    session = assistant.new_session(user_id=args.user)
    print(f"국방 특화 AI 비서 (모델 {assistant.model_label}, 추론 {settings.effort}, 허용 등급 {settings.max_classification.korean})")
    print(f"지식 베이스 {len(assistant.store)}개 청크 적재. /help 로 명령을 확인하십시오.\n")
    stream = not args.no_stream

    while True:
        try:
            line = input("▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            print("종료합니다.")
            return 0
        if line == "/help":
            print(HELP)
            continue
        if line == "/reset":
            session.reset()
            print("대화 기록을 초기화했습니다.")
            continue
        if line == "/status":
            print(f"세션 {session.session_id} / 사용자 {session.user_id} / 턴 {session.turns} / 메시지 {len(session.messages)}")
            print(f"백엔드 {assistant.backend.name}, 모델 {assistant.model_label}, 추론 {settings.effort}, 폴백 {settings.fallbacks}, 마스킹 {settings.redaction}, 감사 로그 {settings.audit_log}")
            continue
        if line.startswith("/docs"):
            q = line[5:].strip()
            if not q:
                print("사용법: /docs <검색어>")
                continue
            print(assistant.store.format_hits(assistant.store.search(q, top_k=3)))
            continue

        try:
            printed = False

            def on_text(chunk: str) -> None:
                nonlocal printed
                if not printed:
                    sys.stdout.write("◀ ")
                    printed = True
                sys.stdout.write(chunk)
                sys.stdout.flush()

            def on_tool(name: str, inp: dict) -> None:
                nonlocal printed
                if printed:
                    sys.stdout.write("\n")
                    printed = False
                print(f"  ⚙ 도구 호출: {name} {inp}")

            result = assistant.chat(session, line, on_text=on_text if stream else None, on_tool=on_tool if args.verbose else None)
            if result.blocked or not stream:
                print("◀ " + result.text)
            elif printed:
                print()
            elif result.text:
                print("◀ " + result.text)
            notes = []
            if result.redaction.changed:
                notes.append(f"마스킹: {result.redaction.summary()}")
            if result.fallback_used:
                notes.append(f"폴백 모델로 응답: {result.served_by}")
            if args.verbose and result.usage.get("input_tokens") is not None:
                notes.append(f"토큰 입력 {result.usage['input_tokens']} / 출력 {result.usage['output_tokens']} / 캐시 {result.usage['cache_read_input_tokens']}")
            if notes:
                print("  ℹ " + " | ".join(notes))
            print()
        except AssistantError as e:
            print(f"✖ {e}\n")
        except KeyboardInterrupt:
            print("\n(중단)\n")


def run_search(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    store = build_store(settings)
    print(store.format_hits(store.search(args.query, top_k=args.top_k, category=args.category)))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("서버 실행에는 `pip install 'defense-assistant[server]'` 가 필요합니다.", file=sys.stderr)
        return 1
    from .server import create_app
    from .storage import Database

    settings = _build_settings(args)
    db = Database(settings.db_path)
    if db.count_users() == 0 and not settings.admin_password:
        print("등록된 사용자가 없습니다. 먼저 `defense-ai users add <아이디> --role admin` 으로 관리자를 만들거나 DAI_ADMIN_PASSWORD 를 설정하십시오.", file=sys.stderr)
        return 1
    app = create_app(DefenseAssistant(settings), db)
    print(f"서버 시작: http://{args.host}:{args.port}  (사용자 DB: {settings.db_path})")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_local_check(args: argparse.Namespace) -> int:
    """로컬 모델 서버 연결·모델 존재·짧은 응답을 차례로 점검한다."""
    from .backends import BackendError
    from .backends.local import LocalBackend
    from .prompts import build_local_system_prompt
    from .tools import build_tools
    from .tools.glossary import load_glossary

    # --model 이 로컬 모델(local_model)에 들어가도록 설정을 만들기 전에 백엔드를 local 로 고정한다.
    args.backend = "local"
    settings = _build_settings(args)
    store = build_store(settings)
    backend = LocalBackend(settings, build_tools(store, load_glossary(settings.glossary_path)), store, build_local_system_prompt(store.titles))
    print(f"[1/3] 서버 연결: {settings.local_base_url}")
    try:
        info = backend.check()
    except BackendError as e:
        print(f"  ✖ {e}")
        return 1
    print(f"  ✔ 연결됨. 서버의 모델 {len(info['models'])}개: {', '.join(info['models'][:8]) or '(없음)'}")
    print(f"[2/3] 모델 확인: {settings.local_model}")
    if not info["model_available"]:
        print(f"  ✖ 서버에 '{settings.local_model}' 이(가) 없습니다. Ollama 라면 `ollama pull {settings.local_model}` 를 실행하거나 DAI_LOCAL_MODEL 을 위 목록 중 하나로 바꾸십시오.")
        return 1
    print("  ✔ 모델 있음")
    print("[3/3] 응답 시험: '경계 근무 교대 시 보고 형식은?'")
    try:
        chunks: list[str] = []
        turn = backend.run([{"role": "user", "content": "경계 근무 교대 시 보고 형식은?"}], user_text="경계 근무 교대 시 보고 형식은?", on_text=chunks.append, on_tool=None)
    except BackendError as e:
        print(f"  ✖ {e}")
        return 1
    preview = turn.text.strip().replace("\n", " ")
    print(f"  ✔ 응답 수신 ({len(turn.text)}자, 도구 {turn.tools_called}, 함수 호출 지원: {'예' if backend._tools_supported else '아니오/미확인'})")
    print(f"  ▶ {preview[:200]}{'…' if len(preview) > 200 else ''}")
    print("점검 완료. .env 에 DAI_BACKEND=local 을 설정하면 이 모델로 동작합니다.")
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    from .ingest import ingest_paths

    settings = _build_settings(args)
    out_dir = Path(args.out) if args.out else settings.docs_dir
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    results = ingest_paths(args.paths, out_dir, category=args.category or "", effective_date=args.effective_date or "", version=args.version or "", tags=tags, title=args.title, overwrite=args.overwrite, dry_run=args.dry_run)
    ok = sum(1 for r in results if r.ok and not r.skipped)
    for r in results:
        if r.error:
            print(f"✖ {r.source}: {r.error}")
        elif r.skipped:
            print(f"– {r.source}: 이미 있음 → {r.output} (--overwrite 로 덮어쓰기)")
        else:
            d = r.doc
            mode = {"article": "조문", "numbered": "번호 제목", "plain": "구조 없음"}[d.mode]
            print(f"✔ {r.source} ({d.format}, 문단 {d.paragraphs}개, 섹션 {d.section_count}개, {mode}) → {r.output}{' [미리보기]' if args.dry_run else ''}")
            if args.dry_run:
                for name, ps in d.sections[:12]:
                    print(f"    ## {name or '(머리말)'}  — {len(ps)}문단")
                if len(d.sections) > 12:
                    print(f"    … 섹션 {len(d.sections) - 12}개 더")
    print(f"완료: 변환 {ok}개, 건너뜀 {sum(1 for r in results if r.skipped)}개, 실패 {sum(1 for r in results if r.error)}개")
    if ok and not args.dry_run:
        print("변환된 문서를 열어 제목·섹션이 맞는지 확인하고, 머리말의 effective_date 등을 채운 뒤 비서를 재시작하십시오.")
    return 0 if not any(r.error for r in results) else 1


def _default_cases_path(args: argparse.Namespace, settings: Settings) -> Path:
    """기본 질문셋: 지식 베이스 폴더(DAI_DOCS_DIR) 옆의 eval/questions.jsonl, 없으면 <루트>/data/eval/questions.jsonl."""
    beside_docs = Path(settings.docs_dir).parent / "eval" / "questions.jsonl"
    if beside_docs.is_file():
        return beside_docs
    root = Path(args.root).resolve() if args.root else Path.cwd()
    return root / "data" / "eval" / "questions.jsonl"


def run_eval(args: argparse.Namespace) -> int:
    from .evaluation import EvalReport, compare_reports, evaluate_answers, evaluate_retrieval, load_cases

    settings = _build_settings(args)
    cases_path = Path(args.cases) if args.cases else _default_cases_path(args, settings)
    try:
        cases = load_cases(cases_path)
    except (OSError, ValueError) as e:
        print(f"✖ 질문셋을 읽을 수 없습니다: {e}", file=sys.stderr)
        return 1
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]
    print(f"질문 {len(cases)}개 ({cases_path})")

    store = build_store(settings)
    retrieval = evaluate_retrieval(store, cases, k=args.k)
    answers = None
    model = "(검색만)"
    if args.answers:
        assistant = DefenseAssistant(settings, store=store)
        model = assistant.model_label
        print(f"답변 평가 시작: 모델 {model}, 질문 {len(cases)}개 (각 질문은 새 세션)")

        def ask(q: str):
            return assistant.chat(assistant.new_session("eval"), q)

        def progress(i: int, n: int, r) -> None:
            mark = "✔" if r.passed else "✖"
            print(f"  {mark} [{i}/{n}] {r.id} {r.seconds}s" + (f"  오류: {r.error}" if r.error else ""))

        answers = evaluate_answers(ask, cases, on_progress=progress)

    report = EvalReport.build(str(cases_path), model, retrieval, answers)
    print("\n".join(report.summary_lines()))
    if args.out:
        print(f"결과 저장: {report.save(args.out)}")
    if args.compare:
        try:
            before = EvalReport.load(args.compare)
            compared = compare_reports(before, report)
        except (OSError, ValueError, KeyError, TypeError) as e:
            detail = f"필드 {e} 이(가) 없습니다" if isinstance(e, KeyError) else str(e)
            print(f"✖ 비교 대상을 읽을 수 없습니다: {detail}", file=sys.stderr)
            return 1
        print("\n[이전 결과와 비교]")
        print("\n".join(compared))
    return 0


def run_users(args: argparse.Namespace) -> int:
    from getpass import getpass

    from .security.classification import Classification
    from .storage import Database

    settings = _build_settings(args)
    db = Database(settings.db_path)

    def ask_password() -> str:
        pw = args.password or os.environ.get("DAI_NEW_PASSWORD")
        if pw:
            return pw
        pw1 = getpass("비밀번호 (8자 이상): ")
        pw2 = getpass("비밀번호 확인: ")
        if pw1 != pw2:
            raise SystemExit("비밀번호가 일치하지 않습니다.")
        return pw1

    try:
        if args.users_command == "add":
            u = db.create_user(args.username, ask_password(), role=args.role, clearance=Classification.parse(args.clearance))
            print(f"생성: {u.username} (역할 {u.role}, 인가 {u.clearance.korean})")
        elif args.users_command == "list":
            users = db.list_users()
            if not users:
                print("등록된 사용자가 없습니다.")
            for u in users:
                print(f"{u.username:20s} {u.role:6s} {u.clearance.korean:8s} {'비활성' if u.disabled else '활성'}  {u.created_at[:19]}")
        elif args.users_command == "passwd":
            if not db.set_password(args.username, ask_password()):
                raise SystemExit("사용자를 찾을 수 없습니다.")
            print(f"비밀번호 변경: {args.username} (기존 로그인은 모두 무효화됨)")
        elif args.users_command in {"disable", "enable"}:
            if not db.update_user(args.username, disabled=(args.users_command == "disable")):
                raise SystemExit("사용자를 찾을 수 없습니다.")
            print(f"{'비활성화' if args.users_command == 'disable' else '활성화'}: {args.username}")
        elif args.users_command == "set":
            ok = db.update_user(args.username, role=args.role, clearance=Classification.parse(args.clearance) if args.clearance else None)
            if not ok:
                raise SystemExit("사용자를 찾을 수 없습니다.")
            u = db.get_user(args.username)
            print(f"변경: {u.username} (역할 {u.role}, 인가 {u.clearance.korean})")
        elif args.users_command == "remove":
            if not db.delete_user(args.username):
                raise SystemExit("사용자를 찾을 수 없습니다.")
            print(f"삭제: {args.username} (대화 기록 포함)")
        else:
            print("사용법: defense-ai users {add,list,passwd,set,disable,enable,remove}", file=sys.stderr)
            return 1
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()
    return 0


def _add_global_options(p: argparse.ArgumentParser, default: object = None) -> None:
    """모든 명령에 공통인 옵션. `default` 가 주어지면 각 옵션의 기본값으로 쓴다(하위 파서용 SUPPRESS)."""
    kw = {} if default is None else {"default": default}
    p.add_argument("--root", help="프로젝트 루트 (data/, audit/ 상대 경로 기준)", **kw)
    p.add_argument("--backend", choices=["claude", "local"], help="모델 백엔드 (기본: DAI_BACKEND 또는 claude)", **kw)
    p.add_argument("--model", help="모델 ID (claude: 기본 claude-opus-5-5 / local: DAI_LOCAL_MODEL)", **kw)
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="추론 깊이", **kw)
    p.add_argument("--no-fallback", action="store_true", help="서버측 안전장치 폴백 비활성화", **kw)
    p.add_argument("-v", "--verbose", action="store_true", help="도구 호출·토큰 사용량 표시", **kw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="defense-ai", description="국방 특화 생성형 AI 비서")
    _add_global_options(p)
    # 전역 옵션은 하위 명령 뒤에 써도 된다 (예: defense-ai chat --user 홍길동 -v).
    # 하위 파서 쪽은 기본값을 SUPPRESS 로 두어, 앞에 준 값을 하위 파서 기본값이 덮어쓰지 않게 한다.
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, default=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", parents=[common], help="대화 (기본)")
    c.add_argument("--user", default="anonymous", help="감사 로그에 기록할 사용자 ID")
    c.add_argument("--no-stream", action="store_true", help="스트리밍 출력 끄기")

    s = sub.add_parser("search", parents=[common], help="지식 베이스 검색")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=4)
    s.add_argument("--category", help="문서 분류(front matter category)로 필터")

    v = sub.add_parser("serve", parents=[common], help="HTTP API 서버 실행")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    sub.add_parser("local-check", parents=[common], help="로컬 모델 서버 연결 점검")

    g = sub.add_parser("ingest", parents=[common], help="HWP/HWPX/DOCX/PDF/TXT 문서를 지식 베이스 Markdown 으로 변환")
    g.add_argument("paths", nargs="+", help="파일 또는 폴더")
    g.add_argument("--out", help="출력 폴더 (기본: DAI_DOCS_DIR)")
    g.add_argument("--category", help="문서 분류 (예: 규정, 교범, 지침)")
    g.add_argument("--effective-date", help="시행일 (YYYY-MM-DD)")
    g.add_argument("--version", help="문서 버전")
    g.add_argument("--tags", help="태그, 쉼표 구분")
    g.add_argument("--title", help="문서 제목 (파일 1개일 때만)")
    g.add_argument("--overwrite", action="store_true", help="같은 이름의 .md 가 있으면 덮어쓰기")
    g.add_argument("--dry-run", action="store_true", help="저장하지 않고 구조만 미리보기")

    e = sub.add_parser("eval", parents=[common], help="질문셋으로 검색·답변 품질 평가")
    e.add_argument("--cases", help="질문셋 JSONL (기본: DAI_DOCS_DIR 옆 eval/questions.jsonl, 없으면 data/eval/questions.jsonl)")
    e.add_argument("--answers", action="store_true", help="모델을 호출해 답변까지 평가 (비용 발생)")
    e.add_argument("--k", type=int, default=4, help="검색 평가 시 상위 k")
    e.add_argument("--category", help="질문셋의 category 로 필터")
    e.add_argument("--limit", type=int, help="앞에서부터 N개만")
    e.add_argument("--out", help="결과 JSON 저장 경로")
    e.add_argument("--compare", help="비교할 이전 결과 JSON")

    u = sub.add_parser("users", parents=[common], help="사용자 관리")
    usub = u.add_subparsers(dest="users_command")
    ua = usub.add_parser("add", parents=[common], help="사용자 추가")
    ua.add_argument("username")
    ua.add_argument("--role", choices=["user", "admin"], default="user")
    ua.add_argument("--clearance", default="RESTRICTED", help="인가 등급: UNCLASSIFIED | RESTRICTED | CONFIDENTIAL | SECRET | TOP_SECRET")
    ua.add_argument("--password", help="비밀번호 (생략 시 프롬프트, 또는 DAI_NEW_PASSWORD 환경 변수)")
    usub.add_parser("list", parents=[common], help="사용자 목록")
    up = usub.add_parser("passwd", parents=[common], help="비밀번호 변경")
    up.add_argument("username")
    up.add_argument("--password")
    us = usub.add_parser("set", parents=[common], help="역할·인가 등급 변경")
    us.add_argument("username")
    us.add_argument("--role", choices=["user", "admin"])
    us.add_argument("--clearance")
    for name, help_text in (("disable", "계정 비활성화"), ("enable", "계정 활성화"), ("remove", "계정 삭제")):
        usub.add_parser(name, parents=[common], help=help_text).add_argument("username")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "chat"
        args.user = "anonymous"
        args.no_stream = False
    if args.command == "chat":
        return run_chat(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "users":
        return run_users(args)
    if args.command == "local-check":
        return run_local_check(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "eval":
        return run_eval(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
