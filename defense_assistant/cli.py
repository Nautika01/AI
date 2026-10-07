"""명령줄 인터페이스.

    defense-ai                 # 대화 (REPL)
    defense-ai chat --user 홍길동
    defense-ai search "거수자 조치"
    defense-ai serve --port 8000
"""

from __future__ import annotations

import argparse
import logging
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

    app = create_app(DefenseAssistant(_build_settings(args)))
    uvicorn.run(app, host=args.host, port=args.port)
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
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
