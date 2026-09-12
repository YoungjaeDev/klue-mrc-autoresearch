# 로컬 기록, W&B, PNG·SVG

학습 코드가 쓴 `events.jsonl`을 별도 CPU 프로세스가 읽습니다. 추적이 실패해도 학습 파일은 그대로 남습니다. 기본값은 **local-only**이며 W&B를 import하지 않습니다. 공개 저장소에서는 CPU 계약 검사와 로컬 그래프 생성을 확인했고, 아래 온라인 명령으로 새 W&B run을 생성하는 검증은 아직 하지 않았습니다.

## 선택적 CPU 패키지

저장소의 uv 환경은 Studio 학습 환경과 별개입니다. 여기에만 tracking extra를 추가합니다. Studio Python, Transformers overlay, 시스템 Python에 패키지를 설치하지 마세요.

```console
uv sync --locked --extra tracking --python 3.11
uv run --extra tracking python -m unittest discover -s tests -v
```

extra에는 `wandb==0.23.1`, `matplotlib==3.10.8`이 있습니다. 온라인 전송 없이 그래프만 그릴 때도 같은 extra를 사용할 수 있습니다. 코드는 W&B를 명시적으로 요청할 때만 import합니다.

## SDK 학습 기록 수집

```console
uv run --extra tracking python -m autoresearch_lab.tracking collect --group my-study --run-name reference --events outputs/reference-01/train/events.jsonl --status outputs/reference-01/train/status.json --metadata outputs/reference-01/train/run.json
```

한 번 읽고 종료합니다. 학습이 진행되는 동안 갱신하려면 `--follow`를 붙입니다. 후보에는 `--run-name candidate-01 --candidate-index 1`처럼 이름과 번호를 지정합니다. 진단과 reference에는 후보 번호를 지정하지 않습니다. 모든 명령에서 같은 연구에는 같은 `--group`을 쓰세요. 이름은 영문·숫자·점·밑줄·하이픈을 사용합니다.

`outputs/tracking/my-study/runs/<run-name>/`에 다음 파일을 씁니다.

- `sanitized-events.jsonl`: 허용한 loss·LR·gradient norm·step·시간·VRAM scalar만 보존합니다.
- `source-status.json`: 원 실행의 상태, 학습 호출 시간, 프로세스 전체 시간, VRAM입니다.
- `metadata.json`: 모델 revision, data/code/adapter hash, 가설, diff hash 등 허용 필드입니다.
- `tracking-status.json`: local-only/온라인 상태와 오류 유형입니다. 원문 예외 메시지나 비밀값은 기록하지 않습니다.

완성된 JSONL 행만 읽고 0부터 연속인 event index를 검사합니다. 같은 로컬 기록을 재수집하면 중복해서 붙이지 않습니다. 원본을 중간에 바꾼 경우에는 실패합니다. `--tracking-dir`로 기록 위치를 바꿀 수 있습니다.

## 공식 점수 연결

`--scores`에는 evaluator가 만든 `scores.json`을 전달합니다. 이 파일과 같은 폴더의 `status.json`, `run.json`, `raw_predictions.jsonl`이 필요합니다. 수집기는 완료 상태와 **status→scores→run·예측 hash**를 검사하고 공식 scorer hash를 확인합니다. 진단용 prefix 점수는 거부합니다. 예측은 로컬에서 hash만 계산하고 W&B에 보내지 않습니다.

SDK reference·candidate·selected는 **`--training-dir`가 필수**입니다. 해당 폴더의 `status.json`, `run.json`, `events.jsonl`, `artifacts.json`, `adapter/`가 있어야 합니다. 완료된 30-step 기록, 모델 revision, 학습 run/event hash, 저장 artifact의 실제 hash를 확인하고, 평가 `run.json`의 adapter 파일별 hash와 정확히 대조합니다. 다른 adapter의 점수를 해당 후보로 등록할 수 없습니다. adapter를 복사해 평가했어도 파일 이름과 내용이 같으면 허용합니다. 기록에는 평가한 adapter identity와 검증한 학습 출처를 보존합니다. `--metadata`가 검증된 모델·데이터·코드 hash를 덮어쓰려 하면 실패합니다.

먼저 SDK reference의 전체 search 점수를 등록합니다. 이것이 best curve의 0번입니다.

```console
uv run --extra tracking python -m autoresearch_lab.tracking score --group my-study --run-name reference --kind reference --scores outputs/reference-search/eval/scores.json --training-dir outputs/reference-01/train --metadata outputs/reference-01/train/run.json
```

후보의 학습과 전체 search 평가가 완료된 뒤 판정을 연결합니다. 아래 `discard`는 명령 형식의 예시이며 실제 점수로 정해야 합니다.

```console
uv run --extra tracking python -m autoresearch_lab.tracking score --group my-study --run-name candidate-01 --kind candidate --candidate-index 1 --decision discard --scores outputs/candidate-01-search/eval/scores.json --training-dir outputs/candidate-01/train --metadata outputs/candidate-01/train/run.json
```

에이전트가 정한 판정이 **현재 best보다 EM이 엄격히 높으면 keep, 동률·하락이면 discard**라는 규칙과 맞는지 검사합니다. ROUGE-W가 오른 동률도 discard입니다. reference와 생성 조건 hash가 다르거나, 이미 기록한 후보의 내용을 바꾸려 하면 실패합니다. reference 없이 후보 best curve를 만들지 않습니다. 실패한 학습은 `collect`로 상태만 남기며 점수를 만들어 넣지 않습니다.

최종 선택 완료 뒤 평가한 4개 arm은 `--kind base`, `studio`, `reference`, `selected`로 각각 등록합니다. final 평가는 기존 evaluator의 selection receipt가 필요합니다. 예를 들어 base는 다음과 같습니다.

```console
uv run --extra tracking python -m autoresearch_lab.tracking score --group my-study --run-name base-final --kind base --scores outputs/base-final/eval/scores.json
```

학습이 없는 base에는 학습 출처 옵션을 붙이지 않습니다. 기존 Studio는 `--kind studio --adapter PATH_TO_YOUR_STUDIO_ADAPTER`로 사용자가 실제 adapter 폴더를 지정하고 평가 adapter와 hash를 대조합니다. SDK reference는 reference 학습 폴더를, selected는 선택된 candidate 또는 reference의 실제 학습 폴더를 `--training-dir`로 지정합니다. 평가 status를 학습 status로 전달하는 별도 옵션은 제공하지 않습니다. 최종 arm끼리도 같은 생성 조건을 검사합니다. Studio adapter가 없으면 해당 arm을 비워 둡니다.

## 기존 Studio loss는 선택 입력

```console
uv run --extra tracking python -m autoresearch_lab.tracking historical-import --group my-study --trainer-state PATH_TO_YOUR_TRAINER_STATE_JSON
```

사용자가 지정한 trainer state에서 빠짐없는 step별 loss와 과거 eval loss만 가져옵니다. 기본적으로 로컬에만 저장합니다. 원 실행 경계 시각·전체 소요 시간이 있다면 별도 JSON을 `--metadata`로 전달할 수 있습니다(`source_run_started_at`, `source_run_ended_at`, `source_duration_seconds`, `source_time_evidence`). import 시각을 과거 학습 시각으로 만들지 않으며, 과거 전체 실행 시간을 순수 학습 시간으로 환산하지 않습니다. 이전 전체 validation 평가 이력은 캠페인 장부에 별도로 명시하세요.

## PNG와 SVG 만들기

```console
uv run --extra tracking python -m autoresearch_lab.plotting --group my-study --figures-dir outputs/figures/my-study
```

학습 loss·과거 eval loss, 후보 공식 EM·ROUGE-W와 best curve, 학습 호출 시간·VRAM, 최종 4-arm 점수의 4종 그래프를 PNG와 SVG로 저장합니다. 없는 값은 빈 상태로 표시합니다. 출력할 때도 기록된 점수를 원래 봉인된 평가 파일과 다시 비교합니다. 원 평가 파일을 옮겼다면 기존 추적 기록을 덮어 고치지 말고 새 group에 다시 등록하세요.

시간 막대는 **`train_seconds`만** 사용합니다. 첫 실행의 컴파일을 포함할 수 있는 SDK 학습 호출 시간입니다. Studio 전체 duration이나 평가 시간을 같은 축에 넣지 않습니다. VRAM은 학습 프로세스가 기록한 peak allocated 값이며 작업 관리자 전체 GPU 사용량과 다릅니다. Windows는 맑은 고딕, 다른 환경은 설치된 NanumGothic을 우선 사용합니다. 한국어 글꼴이 없으면 해당 환경에서 글꼴을 별도로 준비해야 합니다.

## W&B는 명시적으로 선택

W&B에서 사용할 **비공개 프로젝트를 먼저 만들고**, `.env.example`을 참고해 로컬 `.env`에 키를 설정합니다. 키 값은 명령 인수에 쓰지 않습니다. 코드가 기존 프로젝트의 공개 범위를 바꾸거나 프로젝트를 자동으로 만들지 않습니다.

`collect` 또는 `score` 명령 끝에 다음 옵션을 추가하면 온라인 전송을 요청합니다.

```console
--online --entity YOUR_ENTITY --project YOUR_PRIVATE_PROJECT
```

키가 없으면 local-only로 끝납니다. 연결이나 전송이 실패하면 `online_failed`를 기록하고 원본·로컬 mirror를 보존합니다. 동일 명령을 재실행하면 저장한 run ID와 event cursor를 사용합니다. cursor는 SDK가 수락한 위치이며 서버 저장을 독립적으로 증명하는 receipt는 아닙니다. SDK의 비동기 전송 중 장애가 있었다면 W&B history와 로컬 기록을 대조하세요. 로컬 기록이 기준입니다.

console capture, 코드 저장, 시스템 metadata·metrics·machine info, 패키지 목록 자동 수집을 끕니다. 모델·코드 artifact 업로드 호출은 없습니다. 원문 데이터, 답변, 가중치, 전체 config, 환경변수를 보내지 않습니다. `--metadata`의 가설 문자열도 사용자가 공개 범위를 확인한 내용만 사용하세요.
