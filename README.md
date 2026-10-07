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
| 인터페이스 | 터미널 REPL(스트리밍), FastAPI HTTP API(JSON + SSE 스트리밍) |

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
├── cli.py            터미널 REPL / 검색 / 서버 실행
├── server.py         FastAPI 앱
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

### HTTP API
```bash
defense-ai serve --host 127.0.0.1 --port 8000
```
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 상태·모델·적재 문서 수 |
| POST | `/sessions` | 새 세션 `{user_id}` → `{session_id}` |
| POST | `/chat` | `{message, session_id?, user_id?}` → 응답 JSON |
| POST | `/chat/stream` | 같은 입력, SSE 스트리밍 (`text` / `tool` / `done` / `error` 이벤트) |
| GET | `/docs/search?q=` | 지식 베이스 검색 |
| DELETE | `/sessions/{id}` | 세션 삭제 |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"SPOTREP 양식 만들어 줘. 규모는 차량 3대, 시각 071430IOCT26"}' | jq .text
```
운영 시에는 부대 인증(SSO) 뒤에 배치하고 세션 저장소·감사 로그를 보안 저장소로 교체하십시오.

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

모델 호출은 **adaptive thinking + `output_config.effort`**, 시스템 프롬프트 **프롬프트 캐싱**, 스트리밍을 사용합니다. 기본적으로 **서버측 안전장치 폴백**(`fallbacks: "default"`)이 켜져 있어 안전 분류기가 요청을 거부하면 같은 호출 안에서 대체 모델로 재시도합니다. 원치 않으면 `DAI_FALLBACKS=off` 또는 `--no-fallback`으로 끌 수 있습니다.

## 지식 베이스 추가

`data/docs/` 에 UTF-8 Markdown을 넣으면 재시작 시 자동 색인됩니다. 첫 줄 `# 제목`이 문서 제목, `## 소제목` 단위가 검색 청크가 됩니다. 비밀·대외비 표기가 있는 자료는 적재하지 마십시오. 약어는 `data/glossary.json`에 `{"약어": {"full": ..., "ko": ..., "desc": ...}}` 형식으로 추가합니다.

## 보안 설계 메모

- **차단은 모델 호출 전에** 이루어지므로 비밀 등급 표기가 있는 텍스트는 네트워크로 나가지 않습니다. 단, 표기 없이 입력된 비밀 *내용*은 기술적으로 탐지할 수 없으므로 사용자 교육과 망 분리가 병행되어야 합니다.
- 마스킹된 자리표시자는 복원되지 않으며, 시스템 프롬프트가 모델에게 복원 시도를 금지합니다.
- 감사 로그에는 질문·답변 원문이 아닌 해시와 길이만 저장됩니다. 원문 보존이 필요하면 `AuditRecord`를 확장하고 저장소 암호화를 적용하십시오.
- 실패한 턴(네트워크 오류 등)은 세션 기록에 남기지 않아 대화 이력이 오염되지 않습니다.
- Amazon Bedrock·Vertex AI 등 다른 플랫폼에서 구동하려면 `DefenseAssistant(client=...)`에 해당 플랫폼 클라이언트를 주입하고 `DAI_FALLBACKS=off`로 두십시오.

## 테스트

```bash
pytest          # 57개 테스트, 네트워크·API 키 불필요
```
가짜 툴 러너로 전체 처리 흐름(등급 차단, 마스킹, 도구 실행, 폴백 감지, 거부 처리, SSE 스트리밍)을 검증합니다.

## 한계와 향후 과제

- 비밀 *표기* 탐지만 수행하며 내용 기반 비밀성 판단은 하지 않습니다.
- BM25 키워드 검색이므로 의미적으로 유사하지만 어휘가 다른 질의는 놓칠 수 있습니다. 망 분리 환경용 로컬 임베딩 모델 결합을 고려할 수 있습니다.
- 서버의 세션은 메모리에만 저장됩니다. 다중 인스턴스 운영 시 외부 저장소가 필요합니다.
- 접근 통제(인증·권한), 전송 구간 암호화, 로그 보안 저장은 배포 환경에서 추가해야 합니다.
