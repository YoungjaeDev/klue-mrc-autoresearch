# Claude Code로 연구 시작하기

Claude Code가 로그를 읽고 가설을 세워 `train.py`를 수정합니다. 학습되는 Qwen 모델과 연구를 지휘하는 Claude는 별개입니다. `program.md`는 연구 지침이며, Python 스크립트가 다음 실험을 스스로 선택하지는 않습니다.

## 준비와 시작

[README](../README.md)의 CPU 준비와 [Windows GPU 안내](windows-runtime.md)를 먼저 따릅니다. GPU 경로는 기존에 정상 동작하는 Studio 환경과 고정 revision의 모델 캐시를 전제로 합니다. CPU 의존성 설치만으로 GPU 준비가 끝나는 것은 아닙니다.

Claude Code 설치·로그인을 마친 뒤, 전체 저장소를 clone한 폴더에서 PowerShell로 실행합니다.

```powershell
claude --permission-mode default
```

대화에서 `/context`를 실행해 Memory files의 `CLAUDE.md`를 확인합니다. 이 파일은 `@AGENTS.md`로 공통 규칙을 가져옵니다. `README.md`·`prepare.py`·`program.md`·`train.py`가 수강생이 먼저 읽을 네 파일이며, `CLAUDE.md`는 에이전트에 지침을 연결하는 추가 파일입니다. [공식 지침 문서](https://code.claude.com/docs/en/memory), [CLI 문서](https://code.claude.com/docs/en/cli-reference)

첫 요청은 다음처럼 전달합니다.

> AGENTS.md, README.md, program.md와 Windows·평가 안내를 읽고 autoresearch 실습을 진행해 주세요. CPU 검사, 데이터 manifest와 기존 GPU 환경을 확인한 뒤, 새 출력 폴더에서 진단·30-step SDK reference·search 평가를 실행하세요. 완료된 결과를 읽고 한 번에 한 가설을 세워 train.py를 수정하세요. 각 후보는 원 모델의 새 LoRA에서 시작하고, program.md의 데이터·배치·seed·평가 조건을 유지하세요. GPU는 같은 state 경로와 supervisor로 직렬 실행하며 오류 중단 규칙을 따르세요. EM이 엄격히 오를 때만 채택하고, 최대 10후보 또는 탐색 6시간 중 먼저 도달하면 선택을 끝내세요. 그 뒤에만 final을 평가하세요.

## 장부와 재개

첫 후보 작업을 시작할 때 UTC 시작 시각과 6시간 deadline을 장부에 남깁니다. 후보 번호·가설·코드와 adapter hash·점수·keep/discard를 기록하고 실패한 후보도 횟수에 포함합니다. 세션이 바뀌면 기존 장부를 읽고 같은 deadline과 출력 경로를 이어 확인합니다. 완료한 후보를 다시 실행하거나 선택 종료 뒤 새 후보를 추가하지 않습니다.

Windows 가이드는 PowerShell 문법입니다. 실제 명령 도구의 셸을 확인하고, 호출 사이에 임시 변수가 남아 있다고 가정하지 마세요. 의존하는 명령은 같은 셸 호출에서 실행하거나 변수를 다시 설정합니다. 개별 프로세스의 timeout·GPU lock과 전체 연구의 시각·횟수 관리는 구분합니다.

## 확인한 범위

공개 helper의 GPU 검증은 기존 Studio 환경의 1-step 학습·저장과 별도 3문항 재로드입니다. 비공개 10후보 캠페인은 Codex가 지휘했습니다. 2026-09-13 제작 PC에서 Claude Code `2.1.269`의 버전과 CLI 옵션은 확인했지만, Claude Code의 로그인·지침 로드·연구 실행·중단 후 재개 리허설은 아직 하지 않았습니다. Claude로 실행하기 전에는 현재 GPU 작업이 끝났는지 먼저 확인합니다.
