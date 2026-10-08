"""FastAPI HTTP 서버 (다중 사용자).

인증
    POST /auth/login            아이디·비밀번호 → 베어러 토큰
    POST /auth/logout           토큰 폐기
    GET  /auth/me               현재 사용자
대화 (인증 필요, 본인 세션만 접근 가능)
    GET  /sessions              내 대화 목록
    POST /sessions              새 대화
    GET  /sessions/{id}         대화 내용(사용자/비서 텍스트)
    DELETE /sessions/{id}       대화 삭제
    POST /chat                  한 턴 처리 (JSON)
    POST /chat/stream           한 턴 처리 (Server-Sent Events)
    GET  /docs/search?q=        지식 베이스 검색
관리자
    GET/POST /admin/users, PATCH/DELETE /admin/users/{username}, POST /admin/users/{username}/password
    GET  /admin/audit           감사 로그 최근 N건
기타
    GET  /                      웹 채팅 UI
    GET  /health                상태 확인 (인증 불필요, 최소 정보)

운영 시에는 HTTPS 종단(리버스 프록시) 뒤에 두고, 감사 로그를 보안 저장소로 전송한다.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .assistant import AssistantError, DefenseAssistant, Session
from .auth import LoginThrottle, Principal
from .security.classification import Classification
from .storage import Database, transcript

log = logging.getLogger(__name__)
_STATIC = Path(__file__).parent / "static"
_bearer = HTTPBearer(auto_error=False)


# ---- 요청 본문 ---------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    session_id: str | None = None


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=8, max_length=256)
    role: str = "user"
    clearance: str = "RESTRICTED"


class UserUpdate(BaseModel):
    role: str | None = None
    clearance: str | None = None
    disabled: bool | None = None


class PasswordChange(BaseModel):
    password: str = Field(..., min_length=8, max_length=256)


def _parse_clearance(value: str) -> Classification:
    try:
        return Classification.parse(value)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def create_app(assistant: DefenseAssistant | None = None, db: Database | None = None) -> FastAPI:
    app = FastAPI(title="국방 특화 생성형 AI 비서", version="0.2.0")
    session_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    state: dict[str, Any] = {"assistant": assistant, "db": db, "throttle": None}

    def get_assistant() -> DefenseAssistant:
        if state["assistant"] is None:
            state["assistant"] = DefenseAssistant()
        return state["assistant"]

    def get_db() -> Database:
        if state["db"] is None:
            settings = get_assistant().settings
            state["db"] = Database(settings.db_path)
            _bootstrap_admin(state["db"], settings.admin_password)
        return state["db"]

    def get_throttle() -> LoginThrottle:
        if state["throttle"] is None:
            s = get_assistant().settings
            state["throttle"] = LoginThrottle(s.login_max_attempts, s.login_lockout_minutes * 60)
        return state["throttle"]

    settings = get_assistant().settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])
    if db is not None:
        _bootstrap_admin(db, settings.admin_password)

    # ---- 인증 의존성 ---------------------------------------------------
    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.", headers={"WWW-Authenticate": "Bearer"})
        principal = get_db().resolve_token(creds.credentials)
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰이 유효하지 않거나 만료되었습니다.", headers={"WWW-Authenticate": "Bearer"})
        return principal

    def admin_user(user: Principal = Depends(current_user)) -> Principal:
        if not user.is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="관리자 권한이 필요합니다.")
        return user

    def owned_session(session_id: str | None, user: Principal) -> Session:
        if session_id is None:
            session = get_assistant().new_session(user_id=user.username, max_classification=user.clearance)
            get_db().save_session(session)
            return session
        session = get_db().load_session(session_id, username=user.username)
        if session is None:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
        session.max_classification = user.clearance
        return session

    # ---- 기본 ----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": app.version}

    # ---- 인증 ----------------------------------------------------------
    @app.post("/auth/login")
    def login(req: LoginRequest, request: Request) -> dict[str, Any]:
        throttle = get_throttle()
        client = request.client.host if request.client else "unknown"
        key = f"{client}|{req.username}"
        if throttle.is_locked(key):
            raise HTTPException(status_code=429, detail="로그인 실패가 반복되어 잠시 잠겼습니다. 잠시 후 다시 시도하십시오.")
        user = get_db().verify_credentials(req.username, req.password)
        if user is None:
            throttle.record_failure(key)
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
        throttle.reset(key)
        token, exp = get_db().issue_token(user.username, get_assistant().settings.token_ttl_hours)
        return {"token": token, "token_type": "bearer", "expires_at": exp.isoformat(), "username": user.username, "role": user.role, "clearance": user.clearance.name, "clearance_korean": user.clearance.korean}

    @app.post("/auth/logout")
    def logout(creds: HTTPAuthorizationCredentials | None = Depends(_bearer), user: Principal = Depends(current_user)) -> dict[str, bool]:
        return {"revoked": get_db().revoke_token(creds.credentials) if creds else False}

    @app.get("/auth/me")
    def me(user: Principal = Depends(current_user)) -> dict[str, Any]:
        a = get_assistant()
        return {"username": user.username, "role": user.role, "clearance": user.clearance.name, "clearance_korean": user.clearance.korean, "model": a.settings.model, "chunks": len(a.store), "tools": [t.name for t in a.tools]}

    # ---- 대화 ----------------------------------------------------------
    @app.get("/sessions")
    def list_sessions(user: Principal = Depends(current_user)) -> dict[str, Any]:
        return {"sessions": [s.to_dict() for s in get_db().list_sessions(user.username)]}

    @app.post("/sessions")
    def create_session(user: Principal = Depends(current_user)) -> dict[str, str]:
        return {"session_id": owned_session(None, user).session_id}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, user: Principal = Depends(current_user)) -> dict[str, Any]:
        session = owned_session(session_id, user)
        return {"session_id": session.session_id, "turns": session.turns, "messages": transcript(session.messages)}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str, user: Principal = Depends(current_user)) -> dict[str, bool]:
        return {"deleted": get_db().delete_session(session_id, username=user.username)}

    def _run_turn(session: Session, message: str, **callbacks: Any) -> dict[str, Any]:
        lock = session_locks[session.session_id]
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="이 대화는 이미 처리 중입니다. 응답이 끝난 뒤 다시 보내 주십시오.")
        try:
            result = get_assistant().chat(session, message, **callbacks)
            get_db().save_session(session)
            return {"session_id": session.session_id, **result.to_dict()}
        finally:
            lock.release()

    @app.post("/chat")
    def chat(req: ChatRequest, user: Principal = Depends(current_user)) -> dict[str, Any]:
        session = owned_session(req.session_id, user)
        try:
            return _run_turn(session, req.message)
        except AssistantError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, user: Principal = Depends(current_user)) -> StreamingResponse:
        session = owned_session(req.session_id, user)
        q: "queue.Queue[tuple[str, Any] | None]" = queue.Queue()

        def worker() -> None:
            try:
                payload = _run_turn(
                    session, req.message,
                    on_text=lambda t: q.put(("text", t)),
                    on_tool=lambda name, inp: q.put(("tool", {"name": name, "input": inp})),
                )
                q.put(("done", payload))
            except AssistantError as e:
                q.put(("error", {"detail": str(e)}))
            except HTTPException as e:
                q.put(("error", {"detail": e.detail}))
            except Exception as e:  # noqa: BLE001 - 스트림을 반드시 닫아야 한다
                log.exception("chat_stream worker failed")
                q.put(("error", {"detail": f"내부 오류: {type(e).__name__}"}))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def events() -> Iterator[str]:
            yield f"event: session\ndata: {json.dumps({'session_id': session.session_id})}\n\n"
            while True:
                item = q.get()
                if item is None:
                    break
                kind, payload = item
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/docs/search")
    def docs_search(q: str, top_k: int = 4, user: Principal = Depends(current_user)) -> dict[str, Any]:
        hits = get_assistant().store.search(q, top_k=max(1, min(top_k, 8)))
        return {"query": q, "hits": [{"ref": h.chunk.ref, "source": h.chunk.source, "score": round(h.score, 3), "text": h.chunk.text} for h in hits]}

    # ---- 관리자 --------------------------------------------------------
    @app.get("/admin/users")
    def admin_list_users(admin: Principal = Depends(admin_user)) -> dict[str, Any]:
        return {"users": [u.to_dict() for u in get_db().list_users()]}

    @app.post("/admin/users", status_code=201)
    def admin_create_user(req: UserCreate, admin: Principal = Depends(admin_user)) -> dict[str, Any]:
        try:
            return get_db().create_user(req.username, req.password, role=req.role, clearance=_parse_clearance(req.clearance)).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.patch("/admin/users/{username}")
    def admin_update_user(username: str, req: UserUpdate, admin: Principal = Depends(admin_user)) -> dict[str, Any]:
        if username == admin.username and (req.disabled or (req.role and req.role != "admin")):
            raise HTTPException(status_code=400, detail="자기 자신의 관리자 권한을 해제하거나 비활성화할 수 없습니다.")
        try:
            ok = get_db().update_user(username, role=req.role, clearance=_parse_clearance(req.clearance) if req.clearance else None, disabled=req.disabled)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        if not ok:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        return get_db().get_user(username).to_dict()  # type: ignore[union-attr]

    @app.post("/admin/users/{username}/password")
    def admin_set_password(username: str, req: PasswordChange, admin: Principal = Depends(admin_user)) -> dict[str, bool]:
        if not get_db().set_password(username, req.password):
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        return {"updated": True}

    @app.delete("/admin/users/{username}")
    def admin_delete_user(username: str, admin: Principal = Depends(admin_user)) -> dict[str, bool]:
        if username == admin.username:
            raise HTTPException(status_code=400, detail="자기 자신은 삭제할 수 없습니다.")
        return {"deleted": get_db().delete_user(username)}

    @app.get("/admin/audit")
    def admin_audit(limit: int = 100, admin: Principal = Depends(admin_user)) -> dict[str, Any]:
        records = get_assistant().audit.read_all()
        return {"records": records[-max(1, min(limit, 1000)):]}

    return app


def _bootstrap_admin(db: Database, admin_password: str | None) -> None:
    if db.count_users() == 0 and admin_password:
        db.create_user("admin", admin_password, role="admin", clearance=Classification.RESTRICTED)
        log.warning("사용자가 없어 DAI_ADMIN_PASSWORD 로 admin 계정을 생성했습니다. 생성 후 환경 변수를 비우십시오.")
