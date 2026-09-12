# 한국어 Unsloth autoresearch

Qwen3.5-4B를 KLUE-MRC 독해 데이터로 30 step씩 학습하고, 코딩 에이전트가 `train.py`를 고쳐 가며 공식 EM을 비교하는 실습 저장소입니다. 에이전트는 가설을 하나 세우고, 코드를 고치고, 같은 조건으로 학습·평가한 뒤 변경을 남기거나 되돌립니다. 미리 정한 설정 목록을 돌리는 grid search가 아닙니다.

구조는 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)의 `prepare.py` / `train.py` / `program.md` 구분을 따릅니다. 해당 코드를 복사하지 않았습니다.

## 파일

| 파일 | 하는 일 | 연구 중 수정 |
| --- | --- | --- |
| `program.md` | 에이전트 지침. 목표, 후보 수, 기록 형식, 중단 규칙 | 안 함 |
| `prepare.py` | 공개 데이터를 받아 원문·hash를 확인하고 학습 1,024행과 search/final 분할을 고정 | 안 함 |
| `train.py` | Unsloth SDK QLoRA 학습 레시피 | 에이전트가 수정 |
| `evaluate.py` | 답변 생성과 공식 EM·ROUGE-W 채점 | 안 함 |
| `run.py` | GPU 작업을 하나씩 실행하고 timeout·GPU 오류를 기록 | 안 함 |
| `klue_scorer/` | KLUE-baseline 공식 채점 코드 원본 사본 | 안 함 |

## 1. 설치와 데이터 준비

Python과 [uv](https://docs.astral.sh/uv/)가 필요합니다. 데이터 준비에는 GPU 패키지가 필요 없습니다.

```console
uv sync --locked
uv run python prepare.py
```

`prepare.py`는 공개 HF 데이터와 KLUE 공식 dev를 익명으로 내려받고 train 17,554행, validation 5,841행을 모두 원본과 대조합니다. 모델을 받거나 GPU를 쓰지 않습니다. 결과는 `data/generated/klue-autoresearch-v1/manifest.json`에 기록합니다. 이 manifest는 데이터 hash, 분할, 그리고 `prepare.py`·`evaluate.py`·`run.py`·`program.md`·`pyproject.toml`·`uv.lock`의 hash를 고정합니다. `train.py`는 에이전트가 고치므로 제외합니다.

학습 입력은 seed 3407로 섞은 train 1,024행입니다. 30 step × batch 2 × accumulation 4 = 샘플 240개를 봅니다. 30 epoch나 실험 30번이 아닙니다.

## 2. GPU 환경

GPU 패키지는 `gpu` extra에 고정했습니다. Windows에서는 PyTorch CUDA 13.0 wheel을 씁니다. 드라이버·시스템 CUDA는 바꾸지 않고 프로젝트 `.venv` 안에만 설치합니다.

```console
uv sync --locked --extra gpu
```

GPU 패키지를 설치한 뒤에는 모든 명령을 `uv run --extra gpu ...`로 실행하세요. `--extra gpu` 없이 `uv run`을 쓰면 uv가 `.venv`를 기본 의존성에 맞추며 GPU 패키지를 지웁니다.

처음에는 1 step 진단으로 GPU 경로와 adapter 저장을 확인합니다. 이 진단은 성능을 측정하지 않습니다.

```console
uv run --extra gpu python run.py --gpu --state-dir outputs/state --run-dir outputs/smoke/process --timeout-seconds 1800 --frozen-manifest data/generated/klue-autoresearch-v1/manifest.json -- python train.py --manifest data/generated/klue-autoresearch-v1/manifest.json --output outputs/smoke/train --diagnostic-steps 1
```

## 3. 에이전트에게 맡기기

저장소 폴더에서 Claude Code 같은 코딩 에이전트를 열고 이렇게 요청합니다.

> program.md를 읽고 autoresearch를 시작하세요. 후보는 1개만 실행하세요.

에이전트는 `program.md`의 명령으로 후보마다 30 step 학습과 search 평가를 실행하고, 모든 후보를 `results.tsv`에 기록합니다. search EM이 현재 best보다 엄격히 높을 때만 변경을 남깁니다. GPU·CUDA·kernel·OOM 오류가 나면 즉시 멈춥니다.

## 평가 조건

- 모든 비교는 같은 모델 revision, system prompt, NF4 4bit와 bfloat16 계산, SDPA, greedy, max_length 4096, max_new_tokens 128을 씁니다.
- validation을 지문·기사 연결 관계로 묶어 search 약 20%, final 약 80%로 나눕니다. 에이전트는 search만 씁니다.
- final은 후보 선택이 끝난 뒤 사람이 평가합니다. `{"selection_complete": true, "split_manifest_sha256": "..."}` receipt가 없으면 `evaluate.py`가 final을 거부합니다.
- 공식 채점 함수는 수정하지 않습니다. 모델이 답 없음 고정 문구를 쓴 경우만 빈 문자열로 바꿔 채점합니다.
- KLUE public dev는 공개 데이터이므로 비공개 test 점수로 표현하지 않습니다.

## 출처와 라이선스

코드는 [Apache-2.0](LICENSE)입니다. KLUE 데이터와 messages 변환본은 CC-BY-SA-4.0이 따로 적용됩니다. 데이터 저자, 고정 revision, 공식 채점 코드 귀속은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.
