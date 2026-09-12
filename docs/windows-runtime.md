# Windows에서 기존 Studio 환경으로 실행

이 경로는 이미 설치되어 정상 사용 중인 Studio Python과 Transformers overlay를 별도 프로세스에서 재사용합니다. Studio 서비스나 설치 파일을 고치지 않습니다. 새 PC에 같은 환경을 자동으로 설치하는 명령은 아직 검증하지 않았습니다.

## 확인 범위

초기 개발 환경은 Windows, NVIDIA RTX 5070 Ti 16GB였습니다. bootstrap은 다음 **metadata 일치**를 요구합니다. 이 공개 helper로 같은 기존 Studio 환경의 GPU 1-step 학습과 별도 3문항 adapter 재로드까지 확인했습니다. 정식 30-step 학습이나 새 PC 설치가 검증됐다는 뜻은 아닙니다. 다른 버전으로 바꾸려면 별도 환경에서 먼저 검증하고 새 reference를 측정하세요. [실측 기록](verification-2026-09-13.md)

| 패키지 | 확인할 버전 |
| --- | --- |
| transformers | 5.3.0 |
| unsloth / unsloth_zoo | 2026.9.4 / 2026.9.3 |
| torch | 2.11.0+cu130 |
| trl / peft | 0.23.1 / 0.18.1 |
| bitsandbytes | 0.50.2 |
| triton-windows | 3.6.0.post26 |

아래는 PowerShell 명령입니다. 저장소 루트에서 실행하며, 설치 위치가 다르면 실제 경로를 지정합니다. Windows Store 기반 venv의 `Scripts/python.exe`는 실제 CPython으로 연결하는 redirector일 수 있습니다. 이 redirector를 supervisor의 학습 자식으로 실행하면 실제 CPython이 Job Object 밖에서 시작될 수 있으므로, 준비 단계에서 물리적 실행 파일·venv prefix·로드된 Python DLL을 확인하고 manifest로 묶습니다. `--cache-dir`와 출력 폴더는 실행마다 새 경로여야 합니다.

```powershell
$studioPython = Join-Path $env:USERPROFILE '.unsloth/studio/unsloth_studio/Scripts/python.exe'
$overlay = Join-Path $env:USERPROFILE '.unsloth/studio/.venv_t5_530'
uv run python prepare.py --output data/generated/klue-autoresearch-runtime-v1
$manifest = (Resolve-Path 'data/generated/klue-autoresearch-runtime-v1/manifest.json').Path
$runtimeManifest = Join-Path (Get-Location) 'outputs/runtime-manifest-01.json'
uv run python -m autoresearch_lab.windows_runtime prepare `
  --studio-python $studioPython `
  --overlay $overlay `
  --data-manifest $manifest `
  --output $runtimeManifest
$runtimeManifestSha = (Get-FileHash -Algorithm SHA256 $runtimeManifest).Hash.ToLowerInvariant()
uv run python runtime/windows_store/dispatch.py `
  --runtime-manifest $runtimeManifest `
  --runtime-manifest-sha256 $runtimeManifestSha `
  --state-dir outputs/state `
  --run-dir outputs/runtime-check-01/process `
  --timeout-seconds 300 `
  --cache-dir outputs/runtime-check-01/cache `
  --module train `
  --check-only
```

manifest 준비 명령은 Store redirector를 짧은 격리 CPU probe에만 사용합니다. 이후 dispatch는 manifest의 물리적 CPython을 Job Object에 직접 넣습니다. 시작할 때마다 `sitecustomize.py`가 실제 process image, venv prefix, Python DLL 경로·SHA-256, Python 버전을 다시 확인하고, 자식 프로세스에는 같은 venv launcher 설정을 전달합니다. 어느 값이든 다르면 종료 코드 121로 중단합니다.

`bootstrap.json`과 `launch-provenance.json`에는 실제 버전과 경로가 기록됩니다. 값이 다르거나 overlay가 없으면 중단합니다. helper는 설치나 패키지 수리를 하지 않습니다. 로그와 manifest에는 개인 경로가 포함되므로 검토 없이 업로드하지 마세요. manifest의 `status: cpu_prepared_gpu_unverified`는 CPU 검사가 끝났다는 뜻이며 GPU 학습 성공 표시는 아닙니다.

기존 전용 overlay가 없다면 **정상적으로 동작하는 위 Studio 환경 안에서** 프로젝트 내부에 다음 4개 패키지만 준비할 수도 있습니다. 새 target 폴더일 때 실행하고, 이미 준비했다면 설치 대신 아래 check부터 실행하세요.

```powershell
$overlay = Join-Path (Get-Location) '.runtime/tf530-overlay'
uv pip install --python $studioPython --target $overlay --no-deps transformers==5.3.0 huggingface_hub==1.8.0 hf_xet==1.4.2 tiktoken==0.14.0
# 새 overlay를 만든 뒤 위 runtime manifest 준비와 dispatch check를 새 output 이름으로 다시 실행합니다.
```

공개 저장소에서 이 target 설치와 metadata check를 확인했습니다. check 당시 heavy module은 로드하지 않았고, 기존 성공 환경 overlay와 비-pyc 모듈 파일이 바이트 단위로 일치했습니다. Studio 설치 패키지와 기존 overlay는 바꾸지 않습니다. 이후 같은 기존 Studio 환경에서 공개 helper의 GPU 1-step과 별도 재로드를 확인했습니다. **새 PC 전체 GPU 환경 설치를 검증한 결과는 아닙니다.** PyTorch·Unsloth 등 나머지 의존성은 위 표에 맞는 기존 Studio 설치에 있어야 합니다.

`$manifest`는 현재 코드로 새로 준비한 generated manifest여야 합니다. prepare가 데이터·평가 코드뿐 아니라 bootstrap, supervisor, Windows direct-runtime helper, `program.md`, `pyproject.toml`, `uv.lock`을 고정합니다. runtime manifest는 이 목록에 물리적 CPython, Python DLL, `pyvenv.cfg`, overlay 경로를 더하고 실행 전후에 검사합니다. 준비 뒤 조건을 바꿨다면 기존 baseline과 분리해 새 data/runtime output으로 준비하고 다시 검증하세요. 후보별로 manifest hash를 갱신해서는 안 됩니다.

## 1-step 진단

모델 가중치가 로컬에 있어야 무다운로드로 진행됩니다. CPU tokenizer preflight는 `local_files_only=True`로 캐시만 읽습니다. 학습 loader 자체는 캐시가 없으면 HF에서 가중치를 받을 수 있으므로 모델 다운로드가 필요한 새 환경에서는 먼저 해당 다운로드·저장 공간을 확인해야 합니다. 이 저장소의 CPU 준비 명령은 모델 캐시를 만들지 않습니다.

```powershell
uv run python runtime/windows_store/dispatch.py `
  --runtime-manifest $runtimeManifest `
  --runtime-manifest-sha256 $runtimeManifestSha `
  --state-dir outputs/state `
  --run-dir outputs/diagnostic-01/process `
  --timeout-seconds 1800 `
  --gpu `
  --cache-dir outputs/diagnostic-01/cache `
  --module train `
  -- `
  --manifest $manifest `
  --output outputs/diagnostic-01/train `
  --diagnostic-steps 1 `
  --run-kind reference
```

이 명령은 GPU를 사용합니다. `process/process.json`, `process/launch-provenance.json`, `train/status.json`, `train/events.jsonl`, `train/artifacts.json`, `train/adapter/`를 확인합니다. 성공한 프로세스 종료뿐 아니라 `status: completed`, 실제 step 수, 저장된 adapter hash까지 확인하세요. 진단 결과를 정식 비교에 섞지 않습니다.

2026-09-13에는 추적되는 공개 helper로 1 optimizer step·8 sample presentations를 완료하고 adapter를 저장했습니다. 별도 GPU 프로세스에서 그 adapter를 불러와 search의 첫 3문항 생성·공식 채점도 완료했습니다. 첫 step의 학습률은 0이므로 성능 개선 증거가 아니며, 3문항 점수도 정식 비교에 쓰지 않습니다. 수치와 hash, 검증 한계는 [실측 기록](verification-2026-09-13.md)에 있습니다.

## SDK reference와 후보

```powershell
uv run python runtime/windows_store/dispatch.py `
  --runtime-manifest $runtimeManifest `
  --runtime-manifest-sha256 $runtimeManifestSha `
  --state-dir outputs/state `
  --run-dir outputs/reference-01/process `
  --timeout-seconds 1800 `
  --gpu `
  --cache-dir outputs/reference-01/cache `
  --module train `
  -- `
  --manifest $manifest `
  --output outputs/reference-01/train `
  --run-kind reference
```

초기 설정은 30 step, max sequence 4096, batch 2 × accumulation 4, learning rate 2e-4, linear schedule, warmup 3, weight decay 0.001, AdamW 8bit, LoRA rank/alpha 16, dropout 0, seed/data_seed 3407, QLoRA 4bit, packing off, 응답 토큰만 학습입니다. text language layers의 q/k/v/o/gate/up/down projection만 대상으로 삼습니다. reference는 trainable tensor 256개와 parameter 21,233,664개도 확인합니다.

후보는 program.md의 수정 범위 안에서 코드를 바꾸고 `--run-kind candidate`를 지정합니다. 후보마다 출력 폴더를 바꾸며, 에이전트가 첫 후보 시작 때 만든 절대 시각을 `--deadline`에 동일하게 전달합니다. 예를 들어 `$deadline = (Get-Date).ToUniversalTime().AddHours(6).ToString('o')`로 한 번만 계산합니다. 후보마다 다시 6시간을 부여하지 않습니다.

`outputs/state/STOP_GPU.json`이 생기면 실행을 멈춥니다. 이 latch를 지워 자동 재시도하지 말고 오류 원인을 먼저 확인하세요. supervisor는 자신이 만든 Windows Job Object의 물리적 CPython과 그 자손만 종료합니다. CPU 계약 검사는 timeout 뒤 손자 프로세스가 계속 실행되지 않는지 확인합니다. 동일 state directory를 사용하지 않는 외부 Studio 작업까지 잠그는 도구는 아니므로, 사용자가 다른 GPU 작업을 실행하지 않는지도 확인해야 합니다.

개별 timeout은 대기나 컴파일도 포함합니다. 시간 부족으로 실패한 진단과 정상 30-step 학습을 구분합니다. 매 step scalar·시간·VRAM을 JSONL에 남기며 W&B를 import하거나 전송하지 않습니다.
