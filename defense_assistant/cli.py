"""명령줄 인터페이스.

    defense-ai                 # 대화 (REPL)
    defense-ai chat --user 홍길동
    defense-ai search "거수자 조치"
    defense-ai serve --port 8000
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
from .config import Settings

HELP = """명령:
  /help           도움말
  /reset          대화 기록 초기화
  /docs <검색어>   지식 베이스 직접 검색
  /status         설정·세션 상태
  /exit, /quit    종료"""


def _build_settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(root=Path(args.root).resolve() if args.root else None)
    if getattr(args, "model", None):
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
    print(f"국방 특화 AI 비서 (모델 {settings.model}, 추론 {settings.effort}, 허용 등급 {settings.max_classification.korean})")
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
            print(f"모델 {settings.model}, 추론 {settings.effort}, 폴백 {settings.fallbacks}, 마스킹 {settings.redaction}, 감사 로그 {settings.audit_log}")
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
    from .knowledge import DocumentStore

    store = DocumentStore.from_directory(settings.docs_dir)
    print(store.format_hits(store.search(args.query, top_k=args.top_k)))
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="defense-ai", description="국방 특화 생성형 AI 비서")
    p.add_argument("--root", help="프로젝트 루트 (data/, audit/ 상대 경로 기준)")
    p.add_argument("--model", help="모델 ID (기본 claude-opus-5-5)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="추론 깊이")
    p.add_argument("--no-fallback", action="store_true", help="서버측 안전장치 폴백 비활성화")
    p.add_argument("-v", "--verbose", action="store_true", help="도구 호출·토큰 사용량 표시")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="대화 (기본)")
    c.add_argument("--user", default="anonymous", help="감사 로그에 기록할 사용자 ID")
    c.add_argument("--no-stream", action="store_true", help="스트리밍 출력 끄기")

    s = sub.add_parser("search", help="지식 베이스 검색")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=4)

    v = sub.add_parser("serve", help="HTTP API 서버 실행")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    u = sub.add_parser("users", help="사용자 관리")
    usub = u.add_subparsers(dest="users_command")
    ua = usub.add_parser("add", help="사용자 추가")
    ua.add_argument("username")
    ua.add_argument("--role", choices=["user", "admin"], default="user")
    ua.add_argument("--clearance", default="RESTRICTED", help="인가 등급: UNCLASSIFIED | RESTRICTED | CONFIDENTIAL | SECRET | TOP_SECRET")
    ua.add_argument("--password", help="비밀번호 (생략 시 프롬프트, 또는 DAI_NEW_PASSWORD 환경 변수)")
    usub.add_parser("list", help="사용자 목록")
    up = usub.add_parser("passwd", help="비밀번호 변경")
    up.add_argument("username")
    up.add_argument("--password")
    us = usub.add_parser("set", help="역할·인가 등급 변경")
    us.add_argument("username")
    us.add_argument("--role", choices=["user", "admin"])
    us.add_argument("--clearance")
    for name, help_text in (("disable", "계정 비활성화"), ("enable", "계정 활성화"), ("remove", "계정 삭제")):
        usub.add_parser(name, help=help_text).add_argument("username")
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
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
