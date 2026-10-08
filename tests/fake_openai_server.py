"""테스트용 가짜 OpenAI 호환 서버 (Ollama / vLLM / llama-server 의 /v1/chat/completions 흉내).

- 스트리밍 SSE 로 응답하며, 마지막 사용자 메시지 내용에 따라 동작을 바꾼다.
- "시간:" 으로 시작하면 convert_military_time 도구를 호출하고, 결과를 받은 뒤 최종 답을 낸다.
- "길게" 가 들어 있으면 finish_reason=length.
- `reject_tools=True` 면 tools 가 포함된 요청에 400 을 돌려준다 (함수 호출 미지원 서버).
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_fake_server(*, reject_tools: bool = False, model: str = "fake-model") -> FastAPI:
    app = FastAPI()
    app.state.requests = []  # 받은 요청 본문 기록 (테스트에서 검사)

    def sse(chunks: list[dict[str, Any]]) -> Iterator[str]:
        for c in chunks:
            yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def delta(d: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model, "object": "model"}, {"id": "other-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        app.state.requests.append(body)
        if body.get("model") != model:
            return JSONResponse({"error": {"message": f"model '{body.get('model')}' not found"}}, status_code=404)
        if reject_tools and body.get("tools"):
            return JSONResponse({"error": {"message": "tools are not supported"}}, status_code=400)
        last = body["messages"][-1]
        chunks: list[dict[str, Any]]
        if last["role"] == "tool":
            chunks = [delta({"role": "assistant"}), delta({"content": "도구 결과: "}), delta({"content": last["content"].splitlines()[0]}), delta({}, "stop")]
        else:
            text = last["content"] if isinstance(last["content"], str) else ""
            question = text.split("\n\n[참고 문서]")[0]
            if question.startswith("시간:") and body.get("tools"):
                chunks = [
                    delta({"role": "assistant", "content": None}),
                    delta({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "convert_military_time", "arguments": ""}}]}),
                    delta({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"value": question[3:].strip(), "to_zone": "Z"})}}]}),
                    delta({}, "tool_calls"),
                ]
            elif "길게" in question:
                chunks = [delta({"role": "assistant"}), delta({"content": "잘린 "}), delta({"content": "답변"}), delta({}, "length")]
            else:
                has_ref = "[참고 문서]" in text and "[1]" in text
                reply = f"질문 '{question}' 에 대한 답입니다." + (" [1]" if has_ref else "")
                chunks = [delta({"role": "assistant"})] + [delta({"content": ch}) for ch in reply] + [delta({}, "stop")]
        chunks.append({"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": model, "choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}})
        return StreamingResponse(sse(chunks), media_type="text/event-stream")

    return app


def run_in_thread(app: FastAPI):
    """가짜 서버를 백그라운드 스레드의 uvicorn 으로 띄우고 (base_url, stop) 을 돌려준다."""
    import socket
    import threading
    import time

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("fake server failed to start")

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=5)

    return f"http://127.0.0.1:{port}/v1", stop
