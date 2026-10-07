"""FastAPI HTTP 서버.

    POST /sessions            새 세션 생성
    POST /chat                한 턴 처리 (JSON 응답)
    POST /chat/stream         한 턴 처리 (Server-Sent Events 스트리밍)
    GET  /docs/search?q=...   지식 베이스 검색
    GET  /health              상태 확인

운영 시에는 부대 인증 체계(SSO 등) 뒤에 두고, 감사 로그를 보안 저장소로 전송한다.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .assistant import AssistantError, DefenseAssistant, Session


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    session_id: str | None = None
    user_id: str = "anonymous"


class SessionRequest(BaseModel):
    user_id: str = "anonymous"


def create_app(assistant: DefenseAssistant | None = None) -> FastAPI:
    app = FastAPI(title="국방 특화 생성형 AI 비서", version="0.1.0")
    sessions: dict[str, Session] = {}
    lock = threading.Lock()

    def get_assistant() -> DefenseAssistant:
        nonlocal assistant
        if assistant is None:
            assistant = DefenseAssistant()
        return assistant

    def get_session(req: ChatRequest) -> Session:
        with lock:
            if req.session_id:
                s = sessions.get(req.session_id)
                if s is None:
                    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
                return s
            s = get_assistant().new_session(user_id=req.user_id)
            sessions[s.session_id] = s
            return s

    @app.get("/health")
    def health() -> dict[str, Any]:
        a = get_assistant()
        return {"status": "ok", "model": a.settings.model, "chunks": len(a.store), "tools": [t.name for t in a.tools]}

    @app.post("/sessions")
    def create_session(req: SessionRequest) -> dict[str, str]:
        s = get_assistant().new_session(user_id=req.user_id)
        with lock:
            sessions[s.session_id] = s
        return {"session_id": s.session_id}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        with lock:
            return {"deleted": sessions.pop(session_id, None) is not None}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        session = get_session(req)
        try:
            result = get_assistant().chat(session, req.message)
        except AssistantError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return {"session_id": session.session_id, **result.to_dict()}

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        session = get_session(req)
        q: "queue.Queue[tuple[str, Any] | None]" = queue.Queue()

        def worker() -> None:
            try:
                result = get_assistant().chat(
                    session,
                    req.message,
                    on_text=lambda t: q.put(("text", t)),
                    on_tool=lambda name, inp: q.put(("tool", {"name": name, "input": inp})),
                )
                q.put(("done", {"session_id": session.session_id, **result.to_dict()}))
            except AssistantError as e:
                q.put(("error", {"detail": str(e)}))
            except Exception as e:  # noqa: BLE001 - 스트림을 반드시 닫아야 한다
                q.put(("error", {"detail": f"내부 오류: {type(e).__name__}"}))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def events() -> Iterator[str]:
            while True:
                item = q.get()
                if item is None:
                    break
                kind, payload = item
                data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False) if kind == 'text' else data}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/docs/search")
    def docs_search(q: str, top_k: int = 4) -> dict[str, Any]:
        a = get_assistant()
        hits = a.store.search(q, top_k=max(1, min(top_k, 8)))
        return {"query": q, "hits": [{"ref": h.chunk.ref, "source": h.chunk.source, "score": round(h.score, 3), "text": h.chunk.text} for h in hits]}

    return app
