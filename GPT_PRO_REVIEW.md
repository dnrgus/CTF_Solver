# CTF Solver Agent — GPT Pro 3rd Code Review Request

## 프로젝트 개요
Multi-model CTF Solver Agent. Python 3.12+, asyncio, Click CLI, Textual TUI, Pydantic, aiodocker.

3개 AI Provider (Claude/Codex/z.ai)를 병렬로 돌려서 CTF 문제를 푸는 에이전트.
각 solver는 Docker 샌드박스에서 격리되어 실행, 도구(bash, 파일 I/O, web_fetch 등)를 사용.

## 수정 이력

### 1차 리뷰 (30개 이슈 → 전부 수정)
- ToolCall/ToolResult 필드 승격, Claude/ZAI/Codex 프로토콜 수정
- HostSandbox 신규 생성, 첫 LLM 메시지 수정
- httpx 재사용, session.close(), EventBus 개선
- SSRF DNS 차단, hint 서버 파싱, SIGINT 안전

### 2차 리뷰 (26개 이슈 → 전부 수정)
- Codex JSON-RPC deadlock 해결 (서버 request id 보존)
- Codex tool parsing (params.tool은 문자열)
- Codex _rpc() race + error handling
- HostSandbox /challenge/ 경로 매핑
- view_image ToolResult로 실제 이미지 전달 (Claude base64 / ZAI data URL)
- 비용 추적: 실제 모델명 사용 (provider/default 대신)
- regex-only flag: 기본 candidate-only (accept_regex_flag_as_verified=False)
- Docker 보안: CapDrop ALL + 최소 CapAdd + PidsLimit + no-new-privileges
- Docker exec stderr 분리 (msg.stream), write_file mkdir, TarFile close
- Solver cleanup 예외 안전, 에러 전파 보존
- Claude is_error 플래그, ZAI 이미지 분리 user message
- CLI try/finally cleanup, coordinator 이벤트 발행
- EventBus snapshot, hint 서버 검증 강화
- SSRF: scheme 화이트리스트, multicast/unspecified 차단, verify=True, follow_redirects=False
- 프롬프트 이미지 markdown, writeup 비용/solver 수

## 검증 요청사항
다음 항목을 집중적으로 검증해주세요:

1. **실제 실행 경로**: 단위 테스트는 통과하지만, 실제 Claude/Z.AI/Codex API 호출 시 프로토콜이 정확한지
2. **경쟁 상태 (Race Condition)**: asyncio 기반 병렬 실행에서 lock, event 순서
3. **리소스 누수**: Docker 컨테이너, httpx client, 파일 핸들러, EventBus queue
4. **에러 처리**: 예외가 적절히 전파되는지, solver가 조용히 죽지 않는지
5. **보안**: SSRF, 명령어 인젝션, 컨테이너 탈출, Docker capability
6. **비용 추적 정확성**: token counting, pricing 계산, model 이름 매핑
7. **Codex JSON-RPC 프로토콜**: request/response id 매칭, deadlock 가능성
8. **메모리**: EventBus queue limit, message_bus findings trim
9. **TUI/CLI 통합**: 이벤트 라우팅, coordinator 연결, cleanup 순서
10. **타입 안전성**: dataclass 필드, Protocol 준수
11. **Codex `send()` flow**: tool_results만 있을 때 turn을 시작하지 않고 바로 응답하는지
12. **Z.AI multimodal**: tool message 다음에 image user message가 올바르게 들어가는지
13. **Claude `is_error`**: error 있는 ToolResult가 `is_error: true`로 전달되는지
14. **HostSandbox**: `/challenge/distfiles/x`가 실제 host 경로로 올바르게 매핑되는지
15. **Config 우선순위**: CLI > env > TOML > Settings default

## 테스트 상태
- 121 passed, 1 skipped (Docker 통합 테스트 17개 포함)
- ruff 0 violations
- 커버리지: ~47% (TUI/swarm 낮음)

## 파일 구조
```
src/ctf_solver/
├── __init__.py
├── cli.py                    # Click CLI + TUI 분기
├── config.py                 # Pydantic Settings
├── events.py                 # EventBus (pub/sub)
├── hint_server.py            # HTTP hint server
├── prompts.py                # System prompt builder
├── tracing.py                # JSONL tracer
├── session_state.py          # Session persistence
├── writeup.py                # Writeup generator
├── collaboration/
│   ├── loop_detect.py        # Loop detection
│   └── message_bus.py        # Inter-solver messaging
├── providers/
│   ├── __init__.py           # Provider registry
│   ├── base.py               # Protocol + data models
│   ├── claude.py             # Anthropic Messages API
│   ├── codex.py              # JSON-RPC subprocess
│   └── zai.py                # OpenAI-compatible REST
├── sandbox/
│   ├── __init__.py           # SandboxProtocol
│   ├── docker.py             # DockerSandbox
│   └── host.py               # HostSandbox
├── solver/
│   ├── __init__.py
│   ├── coordinator.py        # CoordinatorAgent
│   ├── solver_base.py        # Result types
│   └── swarm.py              # ChallengeSwarm
├── tools/
│   ├── core.py               # bash, file I/O, web, flag
│   ├── flag.py               # Flag extraction
│   └── vision.py             # Image viewing
├── tracking/
│   ├── circuit_breaker.py    # Per-provider breaker
│   └── cost_tracker.py       # Token/cost tracking
└── tui/                      # Textual TUI
tests/                        # 121 tests
```
