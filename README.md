# 한국어 Unsloth autoresearch

Qwen3.5-4B를 KLUE-MRC 독해 데이터로 학습하고, AI 에이전트가 코드를 고쳐 성능을 비교하는 실습 저장소입니다. 에이전트가 가설을 세우고 코드를 수정한 뒤, 같은 조건으로 학습·평가하여 변경을 유지하거나 되돌립니다. 미리 정한 설정 목록을 순서대로 돌리는 grid search는 아닙니다.

현재는 공개용 초기 구현입니다. CPU 계약 검사와 데이터 준비는 아래 명령으로 재현할 수 있습니다. 이 저장소의 GPU 학습, 최종 EM·ROUGE-W, 성능 개선 여부는 **측정 대기**입니다. Windows GPU 환경은 특정 기존 Studio 설치를 재사용하는 경로이며, 새 PC에 설치하는 범용 GPU 프로필은 아직 검증하지 않았습니다.

## 파일 3개부터 읽기

| 파일 | 하는 일 | 연구 중 변경 |
| --- | --- | --- |
| `prepare.py` | 공개 데이터를 내려받고 원문·정답·hash를 확인합니다. 내부 구현은 `autoresearch_lab/prepare.py`입니다. | 분할과 데이터는 고정 |
| `program.md` | 에이전트에게 목표, 예산, 수정 범위, 판정 방법을 알려줍니다. | 시작 전에 확정 |
| `train.py` | 학습 레시피의 실제 코드입니다. 에이전트가 이 파일을 직접 수정하고 실행합니다. | 허용 범위 안에서 수정 |

`autoresearch_lab/run.py`는 학습과 평가를 별도 프로세스로 실행하고 시간 제한, GPU 직렬 실행, 오류 중단을 담당합니다. 가설을 만들거나 다음 실험을 스스로 선택하지는 않습니다. 그 일은 `program.md`를 읽은 코딩 에이전트가 합니다.

## 1. CPU에서 데이터 준비

Python 3.11과 [uv](https://docs.astral.sh/uv/)가 필요합니다. 저장소 폴더에서 실행하세요.

```console
uv sync --locked --python 3.11
uv run python -m unittest discover -s tests -v
uv run python prepare.py
```

준비 명령은 공개 HF 데이터와 KLUE 공식 dev를 익명으로 다운로드합니다. 모델 가중치를 받거나 GPU를 사용하지 않습니다. 원본 train 17,554행과 validation 5,841행을 모두 검증합니다. 이미 있는 파일은 hash가 같을 때만 재사용하며, 다른 실행 결과를 덮어쓰지 않습니다.

`data/generated/klue-autoresearch-v1/manifest.json`에 데이터 경로, SHA-256, 모델 revision, 선택한 ID, 분할을 기록합니다. 새로 준비하려면 `--output data/generated/새이름`을 지정하세요. 다운로드 파일과 준비 결과는 Git에서 제외합니다.

학습 입력은 seed 3407로 섞은 train의 1,024행입니다. 30 optimizer step × batch 2 × gradient accumulation 4 = 240개 샘플 제시 횟수입니다. **30번 실험이나 30 epoch라는 뜻이 아닙니다.** 이 정책은 기존 Studio의 짧은 학습 입력 선택을 독립적으로 재현하며, Studio 내부 코드나 가중치의 완전한 재현을 주장하지 않습니다.

## 2. Windows GPU 환경 확인

[Windows 실행 안내](docs/windows-runtime.md)에서 기존 Studio Python과 Transformers 5.3 overlay를 확인하세요. 이 경로는 설치된 패키지를 고치거나 자동 설치하지 않습니다. CPU용 `uv sync`에는 PyTorch·Unsloth가 포함되지 않습니다.

처음에는 별도 출력 폴더로 **1-step 진단**을 실행하고 로그와 adapter 저장을 확인합니다. 그 뒤 같은 고정 데이터와 조건으로 30-step SDK reference를 만듭니다. GUI에서 학습한 adapter를 가지고 있다면 이 SDK reference와 구분해 보관하세요. 데이터 순서·마스킹·구현 차이 때문에 같은 UI 숫자만으로 동일 실험이라고 볼 수 없습니다.

## 3. 에이전트에게 연구 맡기기

Codex 등 코딩 에이전트에 다음과 같이 요청합니다.

> AGENTS.md와 program.md를 읽고 실행 환경을 확인해 주세요. 데이터와 평가 코드를 고정한 뒤, 30-step SDK reference를 먼저 만들고 탐색 평가를 실행하세요. 결과를 근거로 가설을 하나씩 세워 train 코드를 수정하고, 같은 평가로 유지·폐기를 결정하세요. 한 GPU를 직렬로 쓰고 오류가 나면 중단하세요. 최대 10개 후보 또는 탐색 6시간 중 먼저 도달하는 시점에 후보 선택을 마치세요. 선택 후에만 최종 평가를 실행하세요.

탐색 6시간은 후보 편집·학습·평가를 합한 연구 예산입니다. 개별 학습은 30 step으로 고정합니다. 원조 autoresearch의 고정 학습 시간 방식과 다른 실습 설계입니다. reference 준비와 최종 비교 시간은 탐색 예산 밖에서 별도로 기록합니다. 코드는 후보 횟수를 자동으로 관리하지 않으므로 에이전트가 `outputs/`의 실행 장부에 시작 시각, 횟수, 가설, 코드 hash, 점수, 판정을 남겨야 합니다.

## 평가에서 고정하는 것

- 같은 모델 revision, system prompt, 생성 옵션과 공식 채점 함수를 사용합니다. 세부 옵션은 [평가 안내](docs/evaluation.md)를 확인하세요.
- validation을 지문·기사의 연결 관계로 묶어 탐색 약 20%, 최종 약 80%로 나눕니다. 원본의 답 없음 문제도 유지합니다.
- 후보 선택은 탐색 EM이 현재 best보다 **엄격히 높을 때만** 채택합니다. 동률·하락이면 현재 best를 유지합니다. ROUGE-W는 보고만 하며 동률 판정에 쓰지 않습니다. 답변 길이나 정답 정규화 규칙을 후보별로 바꾸지 않습니다.
- 최종 분할의 정답·예측·점수는 후보 선택에 사용하지 않습니다. 선택 완료 receipt가 있어야 최종 평가 명령이 실행됩니다.
- KLUE public dev는 공개 데이터입니다. 사전학습 노출 여부는 알 수 없으며, 이전 Studio 평가 등 이미 본 이력이 있다면 실행 장부에 적습니다. 새로운 비공개 test 점수로 표현하지 않습니다.

최종 표에는 **zero-shot / 기존 Studio 30-step / SDK reference 30-step / 선택한 모델**의 공식 EM·ROUGE-W를 같은 생성 조건으로 비교합니다. 공개 사용자는 기존 Studio adapter가 없으면 해당 칸을 미제공으로 남길 수 있습니다. 측정하지 않은 숫자는 비워 둡니다. 탐색 EM을 높인 후보가 없으면 reference를 유지합니다. 공통 생성 조건은 Transformers 5.3.0 overlay, NF4 4bit와 bfloat16 계산, SDPA, greedy, max_length 4096, max_new_tokens 128입니다.

## 출처와 라이선스

이 프로젝트의 코드는 [Apache-2.0](LICENSE)입니다. 다운로드하는 KLUE 데이터와 messages 변환본은 **CC-BY-SA-4.0**으로 별도 적용됩니다. 데이터 저자, 고정 revision, 변환 내역과 공식 채점 코드의 귀속은 [THIRD_PARTY.md](THIRD_PARTY.md)에 정리했습니다.

연구 흐름은 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)의 `prepare.py` / `train.py` / `program.md` 구분에서 참고했습니다. 해당 학습 코드를 복사하지 않았습니다. Studio 구현도 복사하지 않고 공개 Unsloth SDK를 호출합니다.
