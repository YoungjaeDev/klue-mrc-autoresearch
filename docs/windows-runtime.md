# Windows에서 기존 Studio 환경으로 실행

이 경로는 이미 설치되어 정상 사용 중인 Studio Python과 Transformers overlay를 별도 프로세스에서 재사용합니다. Studio 서비스나 설치 파일을 고치지 않습니다. 새 PC에 같은 환경을 자동으로 설치하는 명령은 아직 검증하지 않았습니다.

## 확인 범위

초기 개발 환경은 Windows, NVIDIA RTX 5070 Ti 16GB였습니다. bootstrap은 다음 **metadata 일치**를 요구합니다. 이는 이 공개 저장소에서 GPU 학습이 성공했다는 뜻이 아닙니다. 다른 버전으로 바꾸려면 별도 환경에서 먼저 검증하고 새 reference를 측정하세요.

| 패키지 | 확인할 버전 |
| --- | --- |
| transformers | 5.3.0 |
| unsloth / unsloth_zoo | 2026.9.4 / 2026.9.3 |
| torch | 2.11.0+cu130 |
| trl / peft | 0.23.1 / 0.18.1 |
| bitsandbytes | 0.50.2 |
| triton-windows | 3.6.0.post26 |

아래는 PowerShell 명령입니다. 저장소 루트에서 실행하며, 설치 위치가 다르면 실제 경로를 지정합니다. bootstrap은 모델 import 전에 metadata와 overlay 경로를 확인합니다. `--cache-dir`는 실행마다 새 폴더여야 합니다.

```powershell
$studioPython = Join-Path $env:USERPROFILE '.unsloth/studio/unsloth_studio/Scripts/python.exe'
$overlay = Join-Path $env:USERPROFILE '.unsloth/studio/.venv_t5_530'
$manifest = (Resolve-Path 'data/generated/klue-autoresearch-v1/manifest.json').Path
& $studioPython -B -m autoresearch_lab.bootstrap --overlay $overlay --cache-dir outputs/check-01/cache --check-only
```

`bootstrap.json`에는 실제 버전과 경로가 기록됩니다. 값이 다르거나 overlay가 없으면 중단합니다. 함수가 설치나 패키지 수리를 대신하지 않습니다. 로그에는 개인 경로가 포함되므로 검토 없이 업로드하지 마세요.

기존 전용 overlay가 없다면 **정상적으로 동작하는 위 Studio 환경 안에서** 프로젝트 내부에 다음 4개 패키지만 준비할 수도 있습니다. 새 target 폴더일 때 실행하고, 이미 준비했다면 설치 대신 아래 check부터 실행하세요.

```powershell
$overlay = Join-Path (Get-Location) '.runtime/tf530-overlay'
uv pip install --python $studioPython --target $overlay --no-deps transformers==5.3.0 huggingface_hub==1.8.0 hf_xet==1.4.2 tiktoken==0.14.0
& $studioPython -B -m autoresearch_lab.bootstrap --overlay $overlay --cache-dir outputs/project-overlay-check-01/cache --check-only
```

공개 저장소에서 이 target 설치와 metadata check를 확인했습니다. check 당시 heavy module은 로드하지 않았고, 기존 성공 환경 overlay와 비-pyc 모듈 파일이 바이트 단위로 일치했습니다. Studio 설치 패키지와 기존 overlay는 바꾸지 않습니다. **새 PC 전체 GPU 환경 설치나 공개 코드 1-step GPU 학습을 검증한 결과는 아닙니다.** PyTorch·Unsloth 등 나머지 의존성은 위 표에 맞는 기존 Studio 설치에 있어야 합니다.

`$manifest`는 현재 코드로 새로 준비한 generated manifest여야 합니다. prepare가 데이터·평가 코드뿐 아니라 bootstrap, supervisor, `program.md`, `pyproject.toml`, `uv.lock`을 고정하므로 아래 `--frozen-manifest $manifest` 명령이 운영 조건도 검사합니다. 준비 뒤 조건을 바꿨다면 기존 baseline과 분리해 새 output으로 준비하고 다시 검증하세요. 후보별로 manifest hash를 갱신해서는 안 됩니다.

## 1-step 진단

모델 가중치가 로컬에 있어야 무다운로드로 진행됩니다. CPU tokenizer preflight는 `local_files_only=True`로 캐시만 읽습니다. 학습 loader 자체는 캐시가 없으면 HF에서 가중치를 받을 수 있으므로 모델 다운로드가 필요한 새 환경에서는 먼저 해당 다운로드·저장 공간을 확인해야 합니다. 이 저장소의 CPU 준비 명령은 모델 캐시를 만들지 않습니다.

```powershell
uv run python -m autoresearch_lab.run --state-dir outputs/state --run-dir outputs/diagnostic-01/process --timeout-seconds 1800 --gpu --frozen-manifest $manifest -- $studioPython -B -m autoresearch_lab.bootstrap --overlay $overlay --cache-dir outputs/diagnostic-01/cache --module train -- --manifest $manifest --output outputs/diagnostic-01/train --diagnostic-steps 1 --run-kind reference
```

이 명령은 GPU를 사용합니다. `process/process.json`, `train/status.json`, `train/events.jsonl`, `train/artifacts.json`, `train/adapter/`를 확인합니다. 성공한 프로세스 종료뿐 아니라 `status: completed`, 실제 step 수, 저장된 adapter hash까지 확인하세요. 진단 결과를 정식 비교에 섞지 않습니다.

## SDK reference와 후보

```powershell
uv run python -m autoresearch_lab.run --state-dir outputs/state --run-dir outputs/reference-01/process --timeout-seconds 1800 --gpu --frozen-manifest $manifest -- $studioPython -B -m autoresearch_lab.bootstrap --overlay $overlay --cache-dir outputs/reference-01/cache --module train -- --manifest $manifest --output outputs/reference-01/train --run-kind reference
```

초기 설정은 30 step, max sequence 4096, batch 2 × accumulation 4, learning rate 2e-4, linear schedule, warmup 3, weight decay 0.001, AdamW 8bit, LoRA rank/alpha 16, dropout 0, seed/data_seed 3407, QLoRA 4bit, packing off, 응답 토큰만 학습입니다. text language layers의 q/k/v/o/gate/up/down projection만 대상으로 삼습니다. reference는 trainable tensor 256개와 parameter 21,233,664개도 확인합니다.

후보는 program.md의 수정 범위 안에서 코드를 바꾸고 `--run-kind candidate`를 지정합니다. 후보마다 출력 폴더를 바꾸며, 에이전트가 첫 후보 시작 때 만든 절대 시각을 `--deadline`에 동일하게 전달합니다. 예를 들어 `$deadline = (Get-Date).ToUniversalTime().AddHours(6).ToString('o')`로 한 번만 계산합니다. 후보마다 다시 6시간을 부여하지 않습니다.

`outputs/state/STOP_GPU.json`이 생기면 실행을 멈춥니다. 이 latch를 지워 자동 재시도하지 말고 오류 원인을 먼저 확인하세요. supervisor는 자신이 만든 Windows Job Object의 자식 프로세스만 종료합니다. 동일 state directory를 사용하지 않는 외부 Studio 작업까지 잠그는 도구는 아니므로, 사용자가 다른 GPU 작업을 실행하지 않는지도 확인해야 합니다.

개별 timeout은 대기나 컴파일도 포함합니다. 시간 부족으로 실패한 진단과 정상 30-step 학습을 구분합니다. 매 step scalar·시간·VRAM을 JSONL에 남기며 W&B를 import하거나 전송하지 않습니다.
