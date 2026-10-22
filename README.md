# CTF Solver Agent

문제 하나에 대해 여러 LLM "솔버" 인스턴스를 병렬로 경쟁시키는 멀티모델 CTF 자동 풀이 에이전트입니다. 각 인스턴스는 자기만의 Docker 샌드박스 안에서 돌고, 서로 중간 findings를 공유하며, 먼저 플래그를 확정한 쪽이 이깁니다.

이 저장소는 두 프로젝트를 포크하고 합친 것입니다.

- **[yakisoba0728/ctf-solver-agent](https://github.com/yakisoba0728/ctf-solver-agent)** — 이 저장소의 기반. 여러 프로바이더(Claude/Codex/z.ai)를 하나의 인터페이스로 다루고, CTFd 같은 플랫폼 자동 연동 없이 문제를 수동으로 넣는 방식, Docker 샌드박스, Textual TUI.
- **[verialabs/ctf-agent](https://github.com/verialabs/ctf-agent)** — yakisoba가 "영감을 받았다"고 밝힌 원본. BSidesSF 2026 CTF 1위(52/52 전문제 해결). 이 원본의 로직 일부(재시도/bump, 서비스 접속 안내 프롬프트)를 다시 가져와 합쳤습니다 — 자세한 목록은 `MERGE_NOTES.md`.

## 브랜치

| 브랜치 | 내용 |
|---|---|
| `merge-v-y` *(기본 브랜치)* | 이 포크의 실제 작업물: verialabs의 재시도/bump 로직과 프롬프트 개선을 가져오고, yakisoba 쪽 버그 여러 개를 수정(`MERGE_NOTES.md` 참고)했고, 아래의 Codex 구독 인증 수정도 포함. **이걸 쓰세요.** |
| `main` | `yakisoba0728/ctf-solver-agent` 원본 그대로, 비교/벤치마크용 기준선으로 보존. |
| `bench/codex-subscription-auth` | `main` + Codex 구독 인증 버그 수정만 적용(재시도·프롬프트 변경 없음) — `merge-v-y`와 공정하게 비교하기 위한 기준선. |

## 동작 방식

- `ChallengeSwarm`(`src/ctf_solver/solver/swarm.py`)이 문제 하나를 전담합니다. 에이전트 1개로 시작하고, 필요할 때만 최대 `--max-agents`(1–3)까지 늘립니다(아래 "적응형 스케줄링").
- 각 인스턴스는 고정된 작은 도구 집합(`bash`, `read_file`, `write_file`, `list_files`, `view_image`, `web_fetch`, `webhook_create`, `submit_flag`, `check_findings`, `notify_coordinator`)만 자기 컨테이너에 대해 호출합니다. 도구 자체는 범용이고, 실제 능력(pwntools, radare2, SageMath, RsaCtfTool, volatility3, steghide, zsteg 등)은 전부 `sandbox/Dockerfile`이 깔아놓은 것입니다. 어떤 문제 카테고리에 어떤 도구를 써야 하는지는 시스템 프롬프트가 알려줍니다.
- 5스텝마다 인스턴스들이 프로세스 내 메시지 버스로 findings를 주고받습니다. 이 findings는 LLM의 주장이며 검증되지 않은 것으로 표시됩니다. 플래그가 **검증되면**(아래 "플래그 검증 상태") 나머지 인스턴스는 취소됩니다. 검증되지 않은 후보만으로는 실행이 끝나지 않습니다.
- 안전장치: 실행당 `--timeout`, `--max-steps`, `--max-cost`, 프로바이더별 서킷 브레이커, 같은 도구 호출을 반복하면 끼어들고(경고) 결국 끊어버리는 루프 감지기.

Docker로 격리하는 이유는 모델이 **자율적으로** 셸 명령을 만들어 알려지지 않은 바이너리와 실제 원격 서비스에 대해 직접 실행하는 에이전트이기 때문입니다 — 이걸 호스트에서 그대로 돌리면 안 됩니다. `asyncio`는 N개 인스턴스의 네트워크 대기 작업(LLM 호출, `docker exec`)이 한 프로세스 안에서 OS 스레드 N개 없이도 겹쳐 돌아가게 해주는 것입니다.

## 설치

```bash
# 의존성 설치 (editable install; uv든 pip든 됩니다)
uv sync                      # 또는: pip install -e .

# 샌드박스 이미지 (브랜치끼리 Dockerfile이 갈리지 않아서 공유 가능)
docker build -t ctf-sandbox -f sandbox/Dockerfile .
```

Docker와 Python 3.12+가 필요합니다.

## 플래그 검증 상태

정규식에 맞는 문자열은 성공이 아닙니다. 상태는 다섯 가지이며, 실행을 끝내는 것은 아래 두 가지뿐입니다.

| 상태 | 뜻 | 실행 종료 |
|---|---|---|
| `candidate` | 대회 형식에 맞는 문자열 (검증 안 됨) | 아니오 |
| `locally_reproduced` | `solve.py`를 새 샌드박스에서 다시 실행해 플래그가 출력됨. 소스에 플래그 문자열이 없어야 함 | 예 (`reproduced`) |
| `remote_confirmed` | `--verifier-cmd`로 지정한 권위 있는 검증기가 수락 (exit 0) | 예 (`solved`) |
| `failed` | 검증기가 거부(exit 1) 또는 재현 실패 | 아니오 |
| `unknown` | 검증기 장애·타임아웃 (거부로 취급하지 않음) | 아니오 |

- 공식 검증기가 설정되지 않으면 `remote_confirmed`는 기록되지 않습니다. 이 경우 성공은 `locally_reproduced`뿐입니다.
- 검증기는 호스트에서 실행되고, 플래그는 `CTF_FLAG` 환경변수로 전달됩니다(명령 문자열에 넣지 않음). 제출 횟수는 `submit_flag` 호출에만 따릅니다. 텍스트 속 플래그 언급은 후보로만 기록되고 검증기를 호출하지 않습니다.
- 대회 형식이 기본 패턴과 다르면 `--flag-pattern`이나 `challenge.yml`의 `flag_pattern`으로 지정하세요. 기본 패턴은 `POCTF{…}` 안의 `CTF{…}`를 잡지 않습니다.
- `--strict-flags`는 더 이상 필요 없습니다(기본 동작). 호환을 위해 남겨 둔 no-op입니다.

## 적응형 스케줄링

- 에이전트 1개가 항상 먼저 시작합니다. `--max-agents`(기본 1, 최대 3)는 문제당 시작할 수 있는 에이전트 수입니다. 풀은 `--claude/--codex/--zai` 개수에서 옵니다.
- 2번째 이후 에이전트는 기록된 증거를 기준으로 시작합니다. 시간 경과로는 시작하지 않습니다.
  - `refuted`: 실행 중인 에이전트가 모두 끝났고 성공이 없음
  - `stalled`: 실행 중인 에이전트 모두가 새 출력(novel output) 없이 `--stall-steps`(기본 8) 호출을 넘김
  - `verify`: 검증되지 않은 후보가 있고 실행 중인 에이전트가 없음
- 남은 비용 예산이 `--min-budget-fraction`(기본 0.3) 아래이거나 검증이 끝났으면 시작하지 않습니다.
- 새 에이전트는 기록된 상태(기각된 가설, 관찰된 사실, 후보, 건드린 경로)를 받고, 기각된 가설을 다시 시도하지 않도록 지시받습니다.

## 증거 기록과 복구

- 모든 도구 호출과 `record_experiment` 호출(가설, 판정, 근거, 실패 원인, 다음 실험)은 `execution_log.jsonl`에 한 줄씩 추가됩니다. 이 파일이 원본이며, 재개 시 여기서 상태를 다시 만듭니다. 잘린 마지막 줄은 무시합니다.
- 종료 코드가 0이 아닌 명령을 같은 입력으로 다시 실행하려고 하면, 새 증거가 생기기 전까지 실행하지 않고 이전 출력을 돌려줍니다.
- `kb_search`(선택): `--kb-dir`에 `ctf-knowledge-base` 체크아웃을 지정하면 활성화됩니다. KB 자체의 리트리버를 호스트에서 실행하며, 레벨 B를 씁니다(KB 벤치마크에서 top-k가 가장 좋음). 결과는 가설로만 취급합니다.

## 산출물

실행이 끝나면 출력 디렉터리(`--output-dir`, 기본은 문제 폴더)에 다음을 씁니다.

- `verification.json`: 결과(solved / reproduced / unsolved), 모든 후보의 상태와 근거, 에이전트 목록, 한계
- `solve.py`: **검증된 플래그가 있을 때만**. 재현에 성공한 바로 그 스크립트입니다. 검증되지 않은 실행은 `solve_candidate.py`(초안, 재현되지 않음)만 남깁니다.
- `writeup.md`: 한국어 보고서. 기록된 가설과 실험, 관찰된 사실, 미검증 주장, 한계를 담습니다.
- `evidence.json`, `execution_log.jsonl`: 요약과 원본 로그

## 인증

| 프로바이더 | API 키 | 또는: CLI 구독 로그인 |
|---|---|---|
| `--claude N` | `ANTHROPIC_API_KEY` | 아직 미지원 — `providers/claude.py`가 Messages API를 직접 호출함 |
| `--codex N` | `OPENAI_API_KEY` | **가능** — `OPENAI_API_KEY`가 없으면 `codex` CLI가 이미 로그인된 것(ChatGPT/Codex Plus/Pro/Team 구독)을 그대로 물려받아서, 키 없이 돌아갑니다 |
| `--zai N` | `ZAI_API_KEY` | 해당 없음 |

Codex는 Docker 샌드박스 안에서만 실행됩니다. `--no-docker`로 호스트에서 돌리면 codex의 네이티브 도구가 실제 파일시스템을 읽을 수 있어서 거부합니다. `--no-docker`는 Claude와 z.ai에서만 씁니다.

Codex 구독 경로는 실제 `codex app-server`를 직접 찍어가며 확인한 끝에 yakisoba 코드에서 버그 3개를 고쳐야 했습니다(`merge-v-y`와 `bench/codex-subscription-auth` 둘 다 포함).

1. `CodexProvider.validate_config()`가 `openai_api_key`를 무조건 요구했는데, 실제로는 `_start_process()`가 키가 없으면 이미 CLI 자체 로그인을 그대로 물려받게 짜여 있었습니다. 이 체크 하나가 그 경로를 아예 못 타게 막고 있었습니다.
2. 기본값으로 박혀있던 `codex_model`(`gpt-5.4`)은 ChatGPT 플랜 로그인에서 그냥 거부됩니다(`"model is not supported when using Codex with a ChatGPT account"`). 모든 계정에 통하는 모델명은 없으므로, 환경변수 `CODEX_MODEL`(`--codex-model` 같은 CLI 플래그는 없음)을 자신의 `~/.codex/config.toml`의 `model =` 줄에 적힌 값으로 맞춰야 합니다.
3. 실제 app-server가 도구 호출 없이 텍스트로만 답할 때, 응답이 `agentMessage` 아이템의 `item.text`에 직접 들어 있는데, 기존 파서는 `item.content[]` 배열만 봐서 항상 빈 값으로 떨어졌습니다. 그래서 Codex가 텍스트로만 답하는 턴은 전부 `response.text`가 비어서, 텍스트 안의 플래그를 스캔하는 경로가 조용히 죽어 있었습니다.

자격증명은 `.env`(`pydantic-settings`가 읽음)나 해당 환경변수로 설정하세요.

## 사용법

```bash
# 디렉터리 형태 문제 (description.md를 읽고 distfiles/를 마운트)
ctf-solve --challenge-dir ./chall/ --claude 1 --codex 1 --timeout 600 --no-tui

# 파일 + 설명을 직접 지정, Codex는 구독 로그인으로(API 키 없이) —
# CODEX_MODEL은 자기 계정 플랜이 실제로 받아주는 모델명으로 맞춰야 함
CODEX_MODEL=gpt-5.6-sol ctf-solve --files chall.bin --desc "pwn, remote at 1.2.3.4:1337" \
  --category pwn --codex 2 --no-tui

# 플래그 없이 멈춘 솔버를 형제 findings와 함께 최대 2번까지 재시도
ctf-solve --challenge-dir ./chall/ --codex 1 --max-bumps 2

# 당신만의 하네스 문서 같은 추가 작업 규율 텍스트를 모든 솔버 시스템 프롬프트에 주입(A/B 비교용)
ctf-solve --challenge-dir ./chall/ --codex 1 --harness-file ./my-harness.md

# 검증기를 지정하고(exit 0 수락, 1 거부, 그 밖은 unknown), 에이전트를 최대 2개까지 적응형으로
ctf-solve --challenge-dir ./chall/ --claude 1 --codex 1 --max-agents 2 --verifier-cmd './submit_to_ctfd.sh'

# 대회 형식을 명시(기본 패턴이 아닌 경우)
ctf-solve --challenge-dir ./chall/ --claude 1 --flag-pattern 'POCTF\{[^}]+\}'

# 지식 베이스 힌트 사용
ctf-solve --challenge-dir ./chall/ --claude 1 --kb-dir ~/ctf-knowledge-base
```

`--no-tui`는 Textual 대시보드 대신 일반 CLI 로그만 출력합니다 — CI나 스크립트 실행에 쓰세요. `ctf-msg --port <N> "힌트"`로 돌고 있는 `ctf-solve` 인스턴스에 실시간 힌트를 보낼 수 있습니다(포트는 시작할 때 출력됩니다).

전체 옵션: `ctf-solve --help`.

## 테스트

```bash
uv run pytest -q --ignore=tests/test_docker_integration.py   # Docker 필요 없음
uv run ruff check src tests
```

`test_docker_integration.py`는 실제 Docker 데몬과 빌드된 `ctf-sandbox` 이미지가 있어야 합니다.

## 여기 없는 것

- CTFd(또는 다른 플랫폼) 자동 수집/자동 제출 — yakisoba 원본과 똑같이 의도적으로 뺐습니다. 문제는 수동으로 넣습니다. 원격 검증이 필요하면 `--verifier-cmd`로 직접 제공하세요. 이 저장소는 공식 채점기를 흉내 내지 않습니다.
- verialabs의 쿼터 폴백(구독 기반 프로바이더가 레이트리밋 걸리면 API 기반으로 바꿔 타는 것)과 CTFd 클라이언트는 포팅하지 않았습니다.
- Claude 구독 인증(verialabs의 `ClaudeSolver`처럼 raw API 키 대신 `claude` CLI/Claude Agent SDK를 타는 것) — Codex 쪽은 이제 됩니다만 Claude 쪽은 아직 없습니다.

verialabs에서 가져온 것과 yakisoba 쪽에서 고친 것의 전체 목록은 `MERGE_NOTES.md`를 보세요.
