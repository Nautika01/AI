# 국방 특화 생성형 AI 비서 (Defense Assistant)

Claude API 위에 구축한 **국방 업무 특화 생성형 AI 비서**입니다. 군 규정·절차 질의응답, 보고서 작성 지원, 군사 용어·약어 설명, 일시군(DTG)·단위 변환 등 일상 업무를 돕되, 비밀 자료와 민감정보가 외부로 나가지 않도록 **입력 단계에서 보안 통제**를 적용합니다.

> ⚠️ 이 프로젝트는 **일반(평문)·대외비 이하 자료**만 다루도록 설계되었습니다. 비밀 등급 자료는 인가된 보안망·장비에서 승인된 도구로만 취급하십시오. 동봉된 지식 베이스 문서는 공개 군사 상식을 요약한 **샘플**이며 실제 규정이 아닙니다.

## 주요 기능

| 영역 | 내용 |
|---|---|
| 지식 베이스 질의응답 (RAG) | `data/docs/` 의 Markdown 문서를 BM25로 검색해 출처와 함께 답변. 외부 임베딩·벡터 DB 불필요 → 망 분리 환경 적합 |
| 보고서 작성 지원 | SITREP, SPOTREP(SALUTE), 9-Line MEDEVAC, WARNORD, OPORD 양식 생성·채움 |
| 군사 도구 | 일시군(DTG) 시간대 변환(I/Z 등), 현재 DTG, NATO 음성 문자, 단위 변환(밀↔도, 노트↔km/h 등), 군사 약어 사전(56개) |
| 비밀 등급 게이트 | I·II·III급 비밀, 대외비, TOP SECRET/SECRET/CONFIDENTIAL 등 표기를 탐지해 허용 등급 초과 시 **모델 호출 전 차단** |
| 민감정보 마스킹 | 주민등록번호, 군번, 전화번호, 이메일, MGRS·위경도 좌표, 내부망 IP를 자리표시자로 치환한 뒤 전송 |
| 감사 로그 | 모든 턴을 JSONL로 기록 (원문 대신 SHA-256 해시, 등급 판정, 마스킹 건수, 호출 도구, 토큰 사용량) |
| 인터페이스 | 터미널 REPL(스트리밍), 웹 채팅 UI, FastAPI HTTP API(JSON + SSE 스트리밍) |
| 다중 사용자 | 아이디·비밀번호 로그인(scrypt 해시, 베어러 토큰), 사용자별 비밀취급인가 등급, 대화 기록 SQLite 저장, 관리자 API, 로그인 실패 잠금 |

## 아키텍처

```
사용자 입력
   │
   ├─ 1. 비밀 등급 표기 탐지 ──▶ 허용 등급 초과 → 차단 + 감사 로그 (모델 호출 없음)
   ├─ 2. 민감정보 마스킹      ──▶ [군번], [좌표] 등 자리표시자
   ├─ 3. Claude 툴 러너 (스트리밍, adaptive thinking, 서버측 안전장치 폴백)
   │        ├─ search_defense_docs      (BM25 지식 베이스)
   │        ├─ lookup_military_term     (약어 사전)
   │        ├─ convert_military_time / get_current_dtg
   │        ├─ spell_phonetic
   │        ├─ generate_report_template
   │        └─ convert_unit
   ├─ 4. 종료 사유 처리 (refusal / max_tokens / 도구 반복 한도)
   └─ 5. 감사 로그 기록
```

```
defense_assistant/
├── assistant.py      핵심 처리 흐름 (DefenseAssistant, Session, ChatResult)
├── prompts.py        시스템 프롬프트 (프롬프트 캐싱을 위해 가변 값 미포함)
├── config.py         환경 변수 설정 (Settings)
├── cli.py            터미널 REPL / 검색 / 서버 실행 / 사용자 관리
├── server.py         FastAPI 앱 (인증, 세션 영속화, 관리자 API, 웹 UI)
├── auth.py           비밀번호 해시(scrypt), 토큰, 로그인 실패 제한
├── storage.py        SQLite 저장소 (사용자·토큰·대화 세션)
├── static/index.html 웹 채팅 UI
├── security/         classification.py(등급 탐지) · redaction.py(마스킹) · audit.py(감사 로그)
├── knowledge/        tokenizer.py(한국어 바이그램) · store.py(BM25)
└── tools/            Claude 도구 정의 (@beta_tool) 와 순수 함수 구현
data/docs/            샘플 지식 베이스 (Markdown)
data/glossary.json    군사 약어 사전
tests/                단위·통합 테스트 (네트워크 불필요)
```

## 설치

```bash
git clone <repo> && cd AI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"
cp .env.example .env     # ANTHROPIC_API_KEY 등 설정
```

Python 3.10 이상이 필요합니다. API 키 대신 `ant auth login` 프로필도 사용할 수 있습니다.

## 사용법

### 터미널 대화
```bash
defense-ai                      # 또는 python -m defense_assistant
defense-ai chat --user 홍길동 -v   # 사용자 ID 기록, 도구 호출·토큰 표시
```
REPL 명령: `/help`, `/reset`, `/docs <검색어>`, `/status`, `/exit`

예시:
```
▶ 거수자 발견 시 조치 절차 알려줘
◀ 거수자 조치는 다음 4단계로 진행합니다. [1]
   1. 정지 명령 …
   출처: [1] 경계근무 일반 지침 (샘플) › 거수자 조치 절차

▶ 071430IOCT26 를 줄루 시간으로
◀ 070530ZOCT26 입니다 (UTC+0).

▶ 이 II급 비밀 문서를 요약해 줘
◀ ⚠️ 입력에서 II급 비밀 등급 표기('II급 비밀')가 감지되었습니다. 이 비서는 대외비 등급까지만 처리할 수 있으며 …
```

### 지식 베이스 검색만
```bash
defense-ai search "9라인 메데박" --top-k 3
```

### 웹 서비스 / HTTP API (다중 사용자)
```bash
defense-ai users add admin --role admin      # 관리자 계정 생성 (비밀번호 프롬프트)
defense-ai serve --host 0.0.0.0 --port 8000  # 브라우저에서 http://서버IP:8000
```
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/auth/login` | `{username, password}` → `{token, ...}` 이후 `Authorization: Bearer <token>` |
| POST | `/auth/logout` · GET `/auth/me` | 토큰 폐기 · 내 정보 |
| GET/POST | `/sessions` | 내 대화 목록 · 새 대화 |
| GET/DELETE | `/sessions/{id}` | 대화 내용 조회 · 삭제 (본인 것만) |
| POST | `/chat` | `{message, session_id?}` → 응답 JSON |
| POST | `/chat/stream` | 같은 입력, SSE (`session` / `text` / `tool` / `done` / `error`) |
| GET | `/docs/search?q=` | 지식 베이스 검색 |
| GET/POST/PATCH/DELETE | `/admin/users…` | 사용자 관리 (관리자) |
| GET | `/admin/audit` | 감사 로그 조회 (관리자) |
| GET | `/health` | 상태 확인 (인증 불필요) |

```bash
TOKEN=$(curl -s localhost:8000/auth/login -H 'content-type: application/json' \
  -d '{"username":"admin","password":"..."}' | jq -r .token)
curl -s localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"SPOTREP 양식 만들어 줘. 규모는 차량 3대, 시각 071430IOCT26"}' | jq .text
```
Docker 실행, 보안 체크리스트, 사용자 관리 명령 전체는 [docs/운영가이드.md](docs/운영가이드.md)를 참고하십시오.

### 파이썬에서 직접 사용
```python
from defense_assistant import DefenseAssistant  # 또는 from defense_assistant.assistant import DefenseAssistant

assistant = DefenseAssistant()                      # 설정은 환경 변수에서
session = assistant.new_session(user_id="홍길동")
result = assistant.chat(session, "SITREP 항목이 뭐야?", on_text=print)
print(result.tools_called, result.usage)
```

## 설정 (환경 변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | API 키 (또는 `ant auth login`) |
| `DAI_MODEL` | `claude-opus-5-5` | 모델 ID |
| `DAI_EFFORT` | `high` | 추론 깊이 `low/medium/high/xhigh/max` |
| `DAI_MAX_TOKENS` | `16000` | 응답 최대 토큰 |
| `DAI_MAX_ITERATIONS` | `12` | 한 턴 최대 도구 호출 반복 |
| `DAI_FALLBACKS` | `default` | 서버측 안전장치 폴백. Claude API 직결이 아니면 `off` |
| `DAI_MAX_CLASSIFICATION` | `RESTRICTED` | 처리 허용 최대 등급 (`UNCLASSIFIED`~`TOP_SECRET`) |
| `DAI_REDACTION` | `mask` | 민감정보 마스킹 `mask/off` |
| `DAI_DOCS_DIR` | `data/docs` | 지식 베이스 디렉터리 |
| `DAI_GLOSSARY_PATH` | `data/glossary.json` | 약어 사전 |
| `DAI_AUDIT_LOG` | `audit/audit.jsonl` | 감사 로그 경로 |
| `DAI_DB_PATH` | `storage/defense.db` | 사용자·토큰·대화 기록 DB (서버) |
| `DAI_TOKEN_TTL_HOURS` | `12` | 로그인 토큰 유효 시간 (서버) |
| `DAI_LOGIN_MAX_ATTEMPTS` / `DAI_LOGIN_LOCKOUT_MINUTES` | `5` / `10` | 로그인 실패 잠금 (서버) |
| `DAI_CORS_ORIGINS` | (없음) | 허용 출처, 쉼표 구분 (서버) |
| `DAI_ADMIN_PASSWORD` | (없음) | 사용자 0명일 때 admin 자동 생성, 사용 후 제거 (서버) |

모델 호출은 **adaptive thinking + `output_config.effort`**, 시스템 프롬프트 **프롬프트 캐싱**, 스트리밍을 사용합니다. 기본적으로 **서버측 안전장치 폴백**(`fallbacks: "default"`)이 켜져 있어 안전 분류기가 요청을 거부하면 같은 호출 안에서 대체 모델로 재시도합니다. 원치 않으면 `DAI_FALLBACKS=off` 또는 `--no-fallback`으로 끌 수 있습니다.

## 지식 베이스 추가

`data/docs/` 에 UTF-8 Markdown을 넣으면 재시작 시 자동 색인됩니다. 첫 줄 `# 제목`이 문서 제목, `## 소제목` 단위가 검색 청크가 됩니다. 비밀·대외비 표기가 있는 자료는 적재하지 마십시오. 약어는 `data/glossary.json`에 `{"약어": {"full": ..., "ko": ..., "desc": ...}}` 형식으로 추가합니다.

## 보안 설계 메모

- **차단은 모델 호출 전에** 이루어지므로 비밀 등급 표기가 있는 텍스트는 네트워크로 나가지 않습니다. 단, 표기 없이 입력된 비밀 *내용*은 기술적으로 탐지할 수 없으므로 사용자 교육과 망 분리가 병행되어야 합니다.
- 마스킹된 자리표시자는 복원되지 않으며, 시스템 프롬프트가 모델에게 복원 시도를 금지합니다.
- 감사 로그에는 질문·답변 원문이 아닌 해시와 길이만 저장됩니다. 원문 보존이 필요하면 `AuditRecord`를 확장하고 저장소 암호화를 적용하십시오.
- 실패한 턴(네트워크 오류 등)은 세션 기록에 남기지 않아 대화 이력이 오염되지 않습니다.
- 서버 모드에서는 비밀번호를 scrypt로 해시하고 토큰은 SHA-256 해시만 저장합니다. 대화 기록은 마스킹된 상태로 저장되어 민감정보 원문이 DB에 남지 않습니다. 사용자별 인가 등급과 전역 허용 등급 중 낮은 쪽이 적용됩니다.
- Amazon Bedrock·Vertex AI 등 다른 플랫폼에서 구동하려면 `DefenseAssistant(client=...)`에 해당 플랫폼 클라이언트를 주입하고 `DAI_FALLBACKS=off`로 두십시오.

## 테스트

```bash
pytest          # 70개 테스트, 네트워크·API 키 불필요
```
가짜 툴 러너로 전체 처리 흐름(등급 차단, 마스킹, 도구 실행, 폴백 감지, 거부 처리, SSE 스트리밍)과 인증·세션 영속화·관리자 API를 검증합니다.

## 한계와 향후 과제

- 비밀 *표기* 탐지만 수행하며 내용 기반 비밀성 판단은 하지 않습니다.
- BM25 키워드 검색이므로 의미적으로 유사하지만 어휘가 다른 질의는 놓칠 수 있습니다. 망 분리 환경용 로컬 임베딩 모델 결합을 고려할 수 있습니다.
- 서버는 단일 인스턴스 기준입니다(SQLite, 메모리 잠금). 여러 대로 늘리려면 저장소를 PostgreSQL 등으로 바꾸고 로그인 잠금을 공유 저장소로 옮겨야 합니다.
- 전송 구간 암호화(HTTPS)는 리버스 프록시에서 처리해야 하며, 감사 로그 보안 저장은 배포 환경에서 추가해야 합니다.
