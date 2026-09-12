# 에이전트 연구 지침

## 목표와 예산

KLUE-MRC에서 30-step QLoRA reference보다 탐색 공식 EM을 높인다. 현재 best보다 EM이 엄격히 높을 때만 채택한다. 동률·하락이면 현재 best를 유지하고 후보 코드를 되돌린다. ROUGE-W는 보고만 하며 동률 판정에 쓰지 않는다. zero-shot은 학습하지 않은 같은 base model이다.

후보는 최대 10개, 탐색은 첫 후보 작업 시작부터 6시간이다. 둘 중 먼저 도달하면 종료한다. 10은 설정 목록이 아니라 실제 실행하는 후보 수의 상한이며 실패한 후보도 센다. reference 준비와 마지막 비교는 이 시간 밖에 별도로 기록한다. 무인 예약 실행은 이 파일만으로 시작되지 않는다. 사용자가 실행을 맡긴 에이전트가 시각과 횟수를 관리한다.

각 후보는 30 optimizer step, batch 2, accumulation 4, seed/data_seed 3407, 동일한 1,024행을 사용한다. 고정 시간이 아니라 같은 update 수를 비교한다. 각 학습의 시간·VRAM도 보고해 계산 비용을 숨기지 않는다. 1-step은 진단에만 허용하고 다른 output과 `--diagnostic-steps 1`을 쓴다.

## 실행 전

1. AGENTS.md를 읽고 CPU 검사와 데이터 준비를 실행한다. manifest와 split manifest hash를 기록한다.
2. 기존 GPU 환경의 metadata를 확인한다. 임의 설치·자동 수리 없이 1-step 진단부터 진행한다.
3. Git 연구 branch를 만들고 최초 `autoresearch_lab/train.py`를 reference snapshot으로 보관한다.
4. SDK reference를 30 step 학습하고 탐색 분할로 평가한다. 모델·데이터·prompt·생성 옵션·평가 코드를 고정한다.
5. data manifest의 `frozen_files`에 bootstrap, run, 평가 코드와 고정 설정을 더한 controller용 manifest를 만들어 모든 실행 전후 hash를 검사한다. 수정할 train 코드는 제외하되 실행마다 snapshot과 hash를 기록한다. candidate가 자체적으로 검증을 우회하지 못하도록 에이전트가 외부에서 다시 확인한다.

## 반복

한 번에 가설 하나를 쓴다. 탐색 결과의 어떤 문제가 어떤 코드 변경으로 개선될지 설명한다. `autoresearch_lab/train.py`의 optimizer, 학습률 일정, LoRA rank·alpha·dropout·text target 선택 등을 실제로 수정할 수 있다. reference와의 고정 budget/model/data 조건, text-only 학습, 응답 label 검사는 유지한다. callback이나 검증 코드를 지워 점수를 얻지 않는다.

후보마다 고유 output과 코드 snapshot을 사용한다. Windows에서는 GPU lock을 공유하는 run supervisor를 사용하고, 학습이 끝나 GPU가 해제된 뒤 평가를 실행한다. 학습 timeout 1,800초, 탐색 평가 timeout 3,600초를 기본값으로 하되 항상 6시간 절대 deadline 안에서 실행한다. timeout이면 해당 후보를 실패로 기록한다. GPU/kernel 오류, native crash 또는 STOP_GPU가 있으면 즉시 연구를 멈추고 원인을 보고한다. 드라이버·CUDA·Studio 설치를 고치거나 다른 사용자 프로세스를 종료하지 않는다.

장부에는 후보 번호, 가설, 시작/종료, 코드 hash, adapter hash, step 수, 탐색 EM·ROUGE-W, 시간·최대 VRAM, keep/discard 이유를 적는다. 유지하지 않는 후보의 코드만 되돌리고 결과·로그는 보존한다. 다음 가설은 이미 끝난 후보의 탐색 결과에서 만든다. 고정 grid로 대체하지 않는다.

## 수정 금지

데이터 내용·행 순서·분할·seed, base model/revision, 고정 step·batch budget, 평가 생성 조건, 채점 함수, 정답 변환을 바꾸지 않는다. 최종 분할의 정답·예측·점수를 열어 가설을 만들지 않는다. final 분할을 열기 전에 선택을 종료하고 선택한 후보의 코드/adapter hash를 기록한다.

## 최종 비교

선택을 끝낸 뒤 `selection_complete: true`, `split_manifest_sha256`이 든 JSON receipt를 작성한다. zero-shot / 기존 Studio 30-step / SDK reference 30-step / 선택 모델을 같은 final IDs와 생성 조건으로 평가한다. 기존 Studio adapter가 없으면 미제공으로 표시한다. 공통 조건은 Transformers 5.3.0 overlay, NF4 4bit와 bfloat16 계산, SDPA, greedy, max_length 4096, max_new_tokens 128이다. GPU 최종 평가도 한 프로세스씩 실행하고 각 timeout을 명시한다. 선택 후보가 없으면 reference가 선택 결과다. 이전 평가 노출과 train/dev 지문 중복 수를 함께 보고한다. 수치가 없으면 pending으로 남긴다.
