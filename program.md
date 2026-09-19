# KLUE-MRC autoresearch

Qwen3.5-4B를 KLUE-MRC 한국어 독해로 QLoRA 파인튜닝하면서, 에이전트가 스스로 가설을 세우고 `train.py`를 고쳐 search EM을 올린다. 이 문서가 루프 중에 따르는 유일한 규칙이다. 원래 스펙은 `SPEC.md`에 있다.

## 파일

- `prepare.py`: 고정 상수, 데이터 다운로드, 학습 풀과 search/final 분할, 런타임 헬퍼. 수정 금지.
- `evaluate.py`: adapter를 새 프로세스에서 불러와 search 전체를 생성하고 공식 채점한다. 수정 금지.
- `klue_mrc_utils.py`: KLUE-baseline 채점 함수 사본. 수정 금지.
- `data/*_ids.txt`: 학습 풀 1,024행, search, final ID. 수정 금지.
- `pyproject.toml`: 의존성. 수정 금지. 새 패키지를 설치하지 않는다.
- `train.py`: 루프에서 고치는 유일한 코드 파일.
- `results.tsv`, `summary.md`: 기록 파일(git 제외).

## 고정값 (바꾸면 안 되는 것)

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3.5-4B`, Unsloth NF4 4bit 로드, LoRA adapter 학습(QLoRA) |
| 데이터 | `YoungjaeDev/klue-mrc-messages` @ `ae6c27ba…`, 학습 풀 1,024행(`data/train_pool_ids.txt`) |
| 예산 | 실험 1회 = 정확히 128 optimizer steps(학습 풀 1 epoch). batch 4 × accumulation 2 = 유효 배치 8 |
| 시퀀스 | 최대 4,096, packing off, assistant 답변 token만 loss |
| seed | 3407 |
| 생성 | 예시 0개, greedy, thinking off, max_new_tokens 128, NF4. 출력이 `지문에서 답을 찾을 수 없습니다.`이면 빈 문자열로 채점 |
| 채점 | KLUE 공식 dev JSON과 `evaluate_for_klue_mrc` |

바꾸면 안 되는 것: 모델, 데이터와 학습 풀, seed, step 수와 유효 배치, sequence 길이, 생성 조건, `prepare.py`, 평가 코드, 분할 파일, 의존성.

## 바꿔도 되는 것

`train.py`의 learning rate, scheduler, warmup, optimizer 설정, weight decay, gradient clipping, LoRA rank·alpha·dropout·대상 모듈. 출발값은 lr 2e-4, linear, warmup 3, adamw_8bit, weight decay 0.001, max_grad_norm 1.0, rank 16, alpha 16, dropout 0, q/k/v/o/gate/up/down이다.

모델 구조 참고: Qwen3.5-4B 언어 모델은 32층이다. full-attention 8층에는 `q_proj`, `k_proj`, `v_proj`, `o_proj`가 있고, linear-attention(Gated DeltaNet) 24층에는 `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`가 있다. `gate_proj`, `up_proj`, `down_proj`는 32층 모두에 있다. 그래서 출발값의 q/k/v/o는 8층에만 붙는다. vision tower 모듈(`qkv`, `proj`, `linear_fc1`, `linear_fc2`)은 텍스트 과제와 관계없다.

## 목표와 채택 기준

- 채택 지표는 **search EM 하나**다. 지금까지의 최고값보다 **엄격히 높을 때만 keep**한다. 같거나 낮으면 discard한다.
- ROUGE-W, 유형별 EM, 빈 응답 수는 모니터링만 한다. ROUGE-W만 오른 후보는 discard한다.
- 결과를 부풀리지 않는다. 개선이 없으면 없다고 쓴다.

## 실험 1회

모든 명령은 저장소 루트에서 실행한다. GPU 작업은 한 번에 하나만 돈다. 출력은 `run.log`로 보내고 context에 쏟지 않는다.

```bash
C=$(git rev-parse --short HEAD)
uv run --env-file .env train.py --out runs/$C > run.log 2>&1
grep -A6 "^---" run.log                       # train_seconds, peak_vram_gb, num_steps
uv run --env-file .env evaluate.py --adapter runs/$C --split search >> run.log 2>&1
grep -A9 "^---" run.log | tail -9             # search_em, search_rouge_w, em_type1..3, empty_count, eval_seconds
```

- 매 실험은 원 모델에서 새 LoRA로 시작한다. 이전 adapter를 이어서 학습하지 않는다.
- `---` 요약이 없으면 crash다. `tail -n 50 run.log`로 원인을 본다. 오타 같은 단순 실수는 고쳐서 다시 돌리고, 아이디어 자체가 안 되면 crash로 기록하고 넘어간다.
- 실행 1회(학습 또는 평가)가 40분을 넘으면 멈춤으로 보고 crash로 기록하고 되돌린다. 이것은 예산이 아니라 멈춤 방지 규칙이다.
- CUDA, GPU, kernel, OOM 오류가 나면 즉시 멈추고 오류 원문을 보고한다. 우회, 작은 모델, 드라이버 변경을 하지 않는다.
- 가설을 세울 때 `runs/<commit>/search_predictions.jsonl`의 원답변(raw, pred, ground_truth, em)을 읽는다.

## 기록

`results.tsv` (탭 구분, git에 넣지 않음). 헤더:

```
commit	search_em	search_rouge_w	em_type1	em_type2	em_type3	empty_count	peak_vram_gb	train_seconds	eval_seconds	status	description
```

- `commit`은 7자 short hash. `status`는 `keep`, `discard`, `crash`. `description`은 가설 한 줄.
- crash 행은 점수를 0으로 채운다.

`summary.md`: 맨 위에 현재 최고 search EM과 그 commit을 두고, 후보마다 가설·결과·keep 여부를 한 줄씩 덧붙인다.

W&B에는 `train/loss`, `eval/loss`, `search/em`, `search/rouge_w`만 기록된다(코드가 이미 그렇게 한다). peak VRAM과 학습 시간은 run summary에만 남는다. 다른 차트·테이블·아티팩트를 만들지 않는다.

## 루프

브랜치는 `klue-mrc`다. 첫 실험은 `train.py`를 고치지 않은 baseline이다.

후보마다 반복한다.

1. `results.tsv`, `summary.md`, 최근 후보의 search 원답변을 읽고 가설 1개를 세운다. 미리 정한 값 목록을 순서대로 돌리지 않는다. 이전 결과에서 근거를 찾는다.
2. `train.py`만 수정하고 git commit한다.
3. 128 step 학습을 완주하고 search 전체를 채점한다.
4. `results.tsv`에 1행을 추가한다.
5. search EM이 최고값보다 엄격히 높으면 keep(commit 유지). 아니면 discard하고 `git reset --hard <마지막 keep commit>`으로 돌아간다.
6. `summary.md`를 갱신한다.

사람에게 계속할지 묻지 않는다. 사람이 `/goal clear`로 멈출 때까지 후보를 계속 돌린다. 아이디어가 떨어지면 원답변의 오류 유형(빈 응답, 긴 답, 조사 포함 등)과 near-miss 조합을 다시 본다.

## 금지

- `train.py`, `results.tsv`, `summary.md` 외의 파일을 편집하지 않는다(hook이 막는다). Bash로 보호 파일을 고치는 우회를 하지 않는다.
- final 분할의 ID·정답·응답·점수를 열지 않는다. `final_ids`나 `--split final`을 쓰지 않는다(hook이 막는다).
- 학습 코드에 정답을 넣거나 평가 데이터(validation)를 학습에 쓰지 않는다.
- `.env`와 키 값을 출력하지 않는다.
