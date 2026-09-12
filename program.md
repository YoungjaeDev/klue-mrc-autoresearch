# autoresearch 지침

에이전트는 이 파일을 읽고 KLUE-MRC 후보 연구를 진행한다. 목표는 30 step 학습의 탐색(search) 공식 EM을 reference보다 높이는 것이다.

## 파일 역할

- `prepare.py`: 데이터 다운로드, 원문 검증, 학습 1,024행과 search/final 분할 고정. 수정하지 않는다.
- `evaluate.py`: 생성과 공식 EM·ROUGE-W 채점. 수정하지 않는다.
- `run.py`: GPU 작업을 하나씩 실행하고 timeout·GPU 오류를 기록하는 supervisor. 수정하지 않는다.
- `train.py`: 에이전트가 수정하는 유일한 파일이다.

## 준비

1. 사용자와 실행 태그(예: `sep13`)를 정하고 `autoresearch/<태그>` 브랜치를 만든다.
2. `uv run --extra gpu python prepare.py`를 실행한다. 이미 준비했다면 다시 만들지 않는다. 아래에서 `$M`은 `data/generated/klue-autoresearch-v1/manifest.json`이다.
3. `$M`의 `split_manifest_path`와 `split_manifest_sha256`을 읽는다. `frozen_files`에 `prepare.py`·`evaluate.py`·`run.py`·`program.md`·`pyproject.toml`·`uv.lock`이 있는지 확인한다. 연구 중에는 manifest를 다시 만들지 않는다.
4. 수정하지 않은 `train.py`로 reference(`c00`, `--run-kind reference`)를 학습하고 search로 평가한다. 사용자가 reference 결과를 이미 제공했다면 그 값을 쓴다.
5. `results.tsv`를 만들고 헤더 한 줄을 쓴다.

## 명령

학습과 평가는 반드시 `run.py`를 거쳐 하나씩 실행한다. `c01`은 후보 번호로 바꾼다.

```console
uv run --extra gpu python run.py --gpu --state-dir outputs/state --run-dir outputs/c01/train-process --timeout-seconds 1800 --frozen-manifest $M -- python train.py --manifest $M --output outputs/c01/train --run-kind candidate
uv run --extra gpu python run.py --gpu --state-dir outputs/state --run-dir outputs/c01/eval-process --timeout-seconds 3600 --frozen-manifest $M -- python evaluate.py run --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --tokenizer Qwen/Qwen3.5-4B --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --adapter outputs/c01/train/adapter --device cuda --dtype bfloat16 --load-in-4bit --max-length 4096 --max-new-tokens 128 --split-manifest <split_manifest_path> --split-manifest-sha256 <split_manifest_sha256> --split search --output outputs/c01/eval
```

출력 폴더는 매번 새 이름을 쓴다. 기존 산출물을 덮어쓰지 않는다. 학습 전 GPU 경로만 점검할 때는 reference 설정으로 별도 폴더에 `--diagnostic-steps 1`을 쓴다. 이 결과는 후보로 세지 않는다.

## 반복

1. 끝난 후보의 search 결과를 보고 가설 하나를 세운다. 어떤 문제를 어떤 변경으로 고치는지 적는다.
2. `train.py`만 수정하고 커밋한다. 학습률 일정, optimizer, LoRA rank·alpha·dropout, target 선택 등을 바꿀 수 있다.
3. 위 명령으로 학습한 뒤 평가한다. 각 후보는 30 optimizer step(batch 2, accumulation 4, seed 3407, 같은 1,024행)으로 학습한다.
4. search EM이 현재 best보다 엄격히 높으면 keep한다. 같거나 낮으면 discard하고 `git reset --hard <현재 best 커밋>`으로 `train.py`만 되돌린다. `outputs/`와 `results.tsv`는 Git 밖에 있으므로 남는다. ROUGE-W는 기록만 하고 판정에 쓰지 않는다.
5. `results.tsv`에 한 줄을 추가한다.

후보는 사용자가 정한 수만큼 실행하고, 정하지 않았으면 최대 10개다. 실패한 후보도 개수에 포함한다. 한도에 도달하면 멈추고 best 후보의 번호, 커밋, adapter hash를 보고한다.

## results.tsv

탭으로 구분한다. 쉼표가 들어가도 되도록 CSV를 쓰지 않는다. 모든 후보를 기록하고, 실패한 후보도 남긴다. Git에 커밋하지 않는다.

```
candidate	hypothesis	started_at	ended_at	train_code_sha256	adapter_sha256	optimizer_steps	search_em	search_rouge_w	train_seconds	peak_vram_gb	status	reason
```

- `candidate`: `c00`(reference), `c01`, `c02` 순서.
- `hypothesis`: 한 줄 가설. 탭을 쓰지 않는다.
- `started_at`, `ended_at`: 학습 시작과 평가 종료 시각(ISO 8601, 시간대 포함).
- `train_code_sha256`: `train/run.json`의 값.
- `adapter_sha256`: `train/artifacts.json` 파일의 SHA-256.
- `optimizer_steps`, `train_seconds`: `train/status.json`의 값.
- `search_em`, `search_rouge_w`: `eval/scores.json`의 `overall` 값(0–100). 실패하면 빈칸.
- `peak_vram_gb`: `train/status.json`의 `peak_gpu_allocated_bytes`를 GiB로 바꾼 값.
- `status`: `keep`, `discard`, `crash` 중 하나.
- `reason`: 판정 근거나 오류 요약.

## 즉시 중단

GPU·CUDA·kernel·OOM 오류, native crash, GPU 작업 timeout이 나거나 `outputs/state/STOP_GPU.json`이 생기면 연구 전체를 즉시 멈춘다. 재시도하지 않고, 모델·batch를 바꿔 우회하지 않는다. 드라이버·CUDA·패키지를 설치하거나 고치지 않고, 다른 프로세스를 종료하지 않는다. 오류 로그 위치와 마지막 명령을 사용자에게 보고한다.

## 금지

- final 평가를 실행하지 않는다. `--split final`과 `--final-release`를 쓰지 않고, final 분할의 정답·예측·점수를 열지 않는다. 최종 비교는 사람이 한다.
- 데이터·행 순서·분할·seed, base model과 revision, 30 step·batch 예산, 생성 조건, 채점 코드와 정답 변환을 바꾸지 않는다.
- `train.py`의 예산 검사, 응답 label 검사, callback을 지워 점수를 얻지 않는다.
- 동시에 GPU 작업을 두 개 실행하지 않는다.
- `.env`, 토큰, adapter, 데이터를 커밋하지 않는다.
