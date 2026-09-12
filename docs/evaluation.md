# 공식 EM·ROUGE-W 비교

평가는 고정 KLUE-MRC public dev와 수정하지 않은 공식 채점 함수를 사용합니다. 점수는 0–100으로 표시합니다. 답 없음 고정 문구만 빈 문자열로 연결하고, 나머지 모델 답변을 정답에 맞게 고치지 않습니다. ROUGE-W는 문자 bag F1과 다릅니다.

## 탐색 분할

Windows 실행 안내의 `$manifest`, `$runtimeManifest`, `$runtimeManifestSha`를 설정한 뒤 다음 변수를 읽습니다. `$manifest`에는 새 prepare가 기록한 bootstrap·supervisor·Windows runtime helper·`program.md`·의존성 및 데이터·평가 hash가 포함되어야 합니다. 학습과 평가에서 같은 data/runtime manifest를 사용하고 탐색 중 다시 만들지 않습니다.

```powershell
$contract = Get-Content -Raw $manifest | ConvertFrom-Json
$splitPath = $contract.split_manifest_path
$splitHash = $contract.split_manifest_sha256
$model = 'Qwen/Qwen3.5-4B'
$revision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
```

아래 명령은 base model의 탐색 평가이며 GPU를 사용합니다. 같은 옵션으로 reference와 각 후보를 평가하되 `--adapter outputs/reference-01/train/adapter`처럼 해당 adapter를 추가하고 모든 output/cache/run 경로를 새 이름으로 바꿉니다.

```powershell
uv run python runtime/windows_store/dispatch.py `
  --runtime-manifest $runtimeManifest `
  --runtime-manifest-sha256 $runtimeManifestSha `
  --state-dir outputs/state `
  --run-dir outputs/base-search/process `
  --timeout-seconds 3600 `
  --gpu `
  --cache-dir outputs/base-search/cache `
  --module rehearsal.official_mrc_runner `
  -- run `
  --model $model --revision $revision `
  --tokenizer $model --tokenizer-revision $revision `
  --device cuda --dtype bfloat16 --load-in-4bit `
  --max-length 4096 --max-new-tokens 128 `
  --split-manifest $splitPath --split-manifest-sha256 $splitHash `
  --split search --output outputs/base-search/eval
```

첫 1–2개 질문으로 생성 경로를 점검하려면 별도 output에 `--limit 2`를 지정합니다. 이것은 진단 점수이며 전체 탐색 결과로 쓰지 않습니다. 후보 실행에는 학습과 같은 6시간 절대 `--deadline`도 supervisor에 전달합니다. base/reference 준비는 탐색 시작 전 별도 기록합니다.

데이터 hash 외에 모델 revision, tokenizer revision, 4bit 여부, dtype, 최대 입력/출력 길이, prompt와 생성 옵션이 모두 같아야 비교합니다. 공통 조건은 Transformers 5.3.0 overlay, NF4 4bit와 bfloat16 계산, SDPA, greedy(`do_sample=False`), max_length 4096, max_new_tokens 128입니다. 입력과 출력 예산의 합이 4096토큰을 초과하면 조용히 자르지 않고 실패합니다. 탐색 EM이 현재 best보다 엄격히 높을 때만 후보를 채택하며, 동률·하락이면 best를 유지합니다. ROUGE-W는 보고 지표입니다.

## 최종 분할 열기

후보 선택을 종료하고 선택한 adapter hash를 장부에 기록한 뒤에만 다음 receipt를 작성합니다.

```powershell
@{selection_complete = $true; split_manifest_sha256 = $splitHash} | ConvertTo-Json | Set-Content -Encoding utf8NoBOM outputs/final-release.json
```

PowerShell 7의 UTF-8 without BOM 출력을 사용합니다. Windows PowerShell 5라면 Python으로 BOM 없는 JSON을 작성하세요. 평가 명령에서 `--split final --final-release outputs/final-release.json`을 지정하고 base/studio/reference/selected 각각 고유 경로로 실행합니다. 기존 Studio arm에는 사용자가 보관한 30-step adapter 경로를 `--adapter`로 전달합니다. 최종 평가는 질문이 더 많으므로 각 실행에 명시적인 timeout을 부여합니다(예: 14,400초). receipt는 연구 절차를 확인하는 장치이며 사용자 파일 접근을 차단하는 OS 보안 경계는 아닙니다.

4개 arm의 실행이 완료되면 CPU에서 비교합니다.

```console
uv run python -m rehearsal.official_mrc_runner compare --base outputs/base-final/eval --studio outputs/studio-final/eval --reference outputs/reference-final/eval --selected outputs/selected-final/eval --output outputs/final-comparison.json
```

Studio adapter가 없는 공개 사용자는 `--studio`를 생략하고 결과 표에 미제공으로 표시합니다. 비교 명령은 완료 기록, 생성 조건, 예측과 점수의 hash를 확인합니다. 생성 조건이 다르면 새로 맞춰 실행하세요. 공개용 결과 표에는 공식 EM·ROUGE-W, 학습 시간·VRAM, model/data/code revision, 이전 dev 노출 여부와 원본 train/dev 중복 수를 함께 적습니다. 성능 개선이 없더라도 결과를 그대로 남깁니다.
