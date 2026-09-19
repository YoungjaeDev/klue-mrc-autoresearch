# KLUE-MRC autoresearch 스펙

`karpathy/autoresearch`를 fork해서 과제를 KLUE-MRC 한국어 독해 파인튜닝으로 바꾼다. 이 문서 하나가 전체 입력이다. 에이전트는 autoresearch 원본을 이해한 뒤 이 문서대로 `program.md`, 데이터 준비 코드, 평가 코드, 학습 코드, hooks 설정을 만든다. 사람은 이 문서를 고치고, 에이전트는 루프 안에서 `train.py`만 고친다.

이 문서를 fork 루트에 `SPEC.md`로 둔다. 아래 프롬프트와 규칙은 그 이름을 기준으로 쓴다.

## 0. 왜 이렇게 하나

autoresearch는 사람이 데이터·평가·예산·규칙을 고정하고, 에이전트가 결과를 읽고 가설을 세워 학습 코드를 고치는 방식이다. 평가는 고정된 문항으로 하고, 점수가 엄격히 오를 때만 그 수정을 남긴다. 나머지는 되돌린다.

바꿀 값 목록을 미리 채워 두고 순서대로 돌리면 그리드 서치이지 autoresearch가 아니다. 에이전트가 이전 결과와 원답변을 읽고 다음 가설을 세우는 부분이 핵심이다.

## 1. 준비물

| 항목 | 필요한 것 |
|---|---|
| GPU | NVIDIA GPU 1장 권장. Apple Silicon도 된다: `bitsandbytes>=0.50`, `torch>=2.9`, macOS 26 이상이면 `kernels` 패키지를 설치한다(없으면 느린 폴백). NF4는 유지하고 optimizer만 `adamw_torch`로 바꾼다(MPS는 8bit optimizer 미지원). 결과를 CUDA와 직접 비교하지 않는다. CPU만 있는 장비에서는 진행하지 않는다. |
| 도구 | `uv`, `gh`, `git`, Claude Code |
| 네트워크 | Hugging Face 익명 다운로드(모델 약 9GB, 데이터셋), GitHub raw 파일 |
| MCP | fork 루트 `.mcp.json`에 deepwiki(저장소 질의)와 mcpdoc(Unsloth 문서 llms.txt). 에이전트가 autoresearch 원본과 Unsloth API를 문서로 확인할 때 쓴다 |
| W&B | 계정과 `WANDB_API_KEY`. 환경 변수 `WANDB_ENTITY=<내 entity>`, `WANDB_PROJECT=klue-mrc-autoresearch` |

장비별 하드웨어 조건은 적지 않는다. 환경 파악은 4절 첫 단계에서 에이전트가 직접 한다.

- OS, GPU 이름·VRAM·장수, 드라이버·CUDA 버전, Python·uv 버전, 디스크 여유를 확인해 보고한다.
- GPU가 여러 장이면 `CUDA_VISIBLE_DEVICES`로 1장만 고정한다.
- Apple Silicon이면 MPS로 진행한다. 모델이 실제로 4bit로 올라갔는지와 1 step 시간을 보고한다. NVIDIA GPU도 Apple Silicon도 없으면 진행하지 않고 그 사실만 보고한다. 다른 모델로 우회하지 않는다.
- VRAM이 부족해 batch 4가 안 들어가면 batch 2 × accumulation 4로 바꾼다. 유효 배치 8과 128 step의 뜻은 그대로다.
- 예상 시간은 적지 않는다. 진단과 baseline에서 실측한 값으로 사람이 예산을 가늠한다.
- Windows 네이티브면 `triton-windows`를 의존성에 넣는다. Claude Code hooks는 Git Bash로 실행되고 `jq`가 필요하다. hook이 받는 `file_path`는 `C:\...\train.py`처럼 역슬래시이므로 6절처럼 정규화한다. GPU가 화면 출력도 맡으면 데스크톱 앱이 VRAM 1GB 이상을 쓰므로 루프 중에는 GPU를 쓰는 앱을 줄인다.
- `git config user.name`/`user.email`이 비어 있으면 루프의 commit이 실패한다. 환경 파악 때 확인한다.
- `.env`는 자동으로 읽히지 않는다. 모든 학습·평가 명령은 `uv run --env-file .env …`로 실행하고, `.env`는 `.gitignore`에 넣는다.

fork와 스펙 배치:

```bash
gh repo fork karpathy/autoresearch --clone && cd autoresearch
git checkout -b klue-mrc
curl -fsSL https://raw.githubusercontent.com/YoungjaeDev/klue-mrc-autoresearch/main/SPEC.md -o SPEC.md
git add SPEC.md && git commit -m "docs: add KLUE-MRC autoresearch spec"
```

`.mcp.json`도 같은 자리에 둔다. Claude Code를 시작할 때 이 서버 2개를 허용한다. llms.txt는 `--urls`에 `이름:URL` 쌍으로 더 넣을 수 있다.

```json
{
  "mcpServers": {
    "deepwiki": { "type": "http", "url": "https://mcp.deepwiki.com/mcp" },
    "mcpdoc": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "mcpdoc", "--with", "mcp<2", "mcpdoc",
               "--urls", "Unsloth:https://docs.unsloth.ai/llms.txt",
               "--transport", "stdio"]
    }
  }
}
```

## 2. 고정값

에이전트가 만드는 모든 코드는 이 표를 따른다. 표에 없는 값은 Unsloth 기본값을 쓴다.

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3.5-4B`. Unsloth로 NF4 4bit 로드, LoRA adapter 학습(QLoRA). VLM(`Qwen3_5ForConditionalGeneration`)이라 transformers 5와 torchvision·pillow가 필요하고, Unsloth가 돌려주는 tokenizer는 processor일 수 있다 |
| 데이터 | Hugging Face 데이터셋 `YoungjaeDev/klue-mrc-messages`, revision `ae6c27baff54df9d0a63ed85451badd6aefc131c`. `train.jsonl`·`validation.jsonl`, 행 = `id` + `messages(system, user, assistant)` |
| 학습 풀 | train을 seed 3407로 섞은 뒤 앞 1,024행. 질문 유형 비율은 맞추지 않는다. ID 목록 `data/train_pool_ids.txt` |
| 분할 | validation을 search / final로 나눈다. 지문(NFKC 정규화 후 공백 축약)이 같거나 출처와 제목이 같은 문항은 한 그룹으로 묶어 같은 쪽에 둔다. 그룹 단위 배정이 우선이고, KLUE dev JSON의 `question_type` 3개 비율은 근사로 맞춘다. seed 3407, search 약 20%, 나머지 final. `data/search_ids.txt`, `data/final_ids.txt` |
| 예산 | 실험 1회 = 정확히 128 optimizer steps(학습 풀 1,024행 1 epoch). batch 4 × accumulation 2 = 유효 배치 8. 최대 sequence 4,096, packing off, assistant 답변 token만 loss |
| seed | 학습 풀 선택·분할·학습 모두 3407 |
| 출발값 | learning rate 2e-4, linear scheduler, warmup 3 steps, adamw_8bit, weight decay 0.001, max_grad_norm 1.0, LoRA rank 16, alpha 16, dropout 0, 대상 q/k/v/o/gate/up/down projection |
| eval/loss | batch 1로 계산하고, eval이 끝나면 `torch.cuda.empty_cache()`를 호출한다. Unsloth fused CE는 드라이버 기준 여유 메모리(`mem_get_info`)로 chunk를 정하는데, eval이 남긴 PyTorch 캐시를 여유로 보지 않아 "No or negligible GPU memory" 오류를 낸다 |
| 생성 | 예시 문답 0개, greedy decoding, thinking off, max_new_tokens 128, NF4 4bit. 앞뒤 공백을 뺀 출력이 정확히 `지문에서 답을 찾을 수 없습니다.`이면 빈 문자열로 바꿔 채점한다 |
| 채점 | KLUE 공식 dev JSON을 정답으로, KLUE-baseline의 `evaluate_for_klue_mrc`를 수정 없이 쓴다. 정답 JSON: `https://github.com/KLUE-benchmark/KLUE/blob/3efd98708a40ff49251fddde35453f8fbb11f536/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json`, 채점 함수: `https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/metrics/utils.py`. 채점 전에 validation.jsonl과 dev JSON의 ID가 모두 일치하는지 확인한다 |
| 평가 방법 | 저장한 adapter를 새 프로세스에서 `PeftModel.from_pretrained`로 불러오고, 저장된 LoRA tensor가 모두 모델에 들어갔는지 assert한다. transformers의 `load_adapter`는 key가 맞지 않으면 조용히 빼먹는다 |

출발값의 출처는 Unsloth Studio 기본값과 Unsloth 노트북 관례다. 근거가 약한 값(warmup, weight decay)은 그래서 "바꿔도 되는 항목"에 있다. Unsloth 문서는 Qwen3.5에 QLoRA(4bit)를 권장하지 않지만(양자화 오차), 이 스펙은 16GB급 GPU를 위해 NF4를 고정한다.

모델 구조: 언어 모델 32층 중 full-attention 8층에만 `q/k/v/o_proj`가 있고, linear-attention(Gated DeltaNet) 24층에는 `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`가 있다. `gate/up/down_proj`는 32층 모두에 있다. 출발값의 LoRA는 128개 모듈(tensor 256개)에 붙는다. 이 사실을 `program.md`에 적어 루프가 대상 모듈을 가설로 쓸 수 있게 한다.

루프에서 바꿔도 되는 것: learning rate, scheduler, warmup, optimizer 설정, weight decay, gradient clipping, LoRA rank·alpha·dropout·대상 모듈.

바꾸면 안 되는 것: 모델, 데이터와 학습 풀, seed, step 수와 유효 배치, sequence 길이, 생성 조건, `prepare.py`, 평가 코드, 분할 파일, 의존성.

## 3. 지표와 기록

- 채택 지표는 search EM 하나다. 지금까지의 최고값보다 엄격히 높을 때만 keep한다. 같거나 낮으면 discard. KLUE-MRC의 1차 지표가 EM이고, ROUGE-W는 긴 답에 유리해 채택 기준으로 쓰지 않는다.
- ROUGE-W, 유형별 EM, 빈 응답 수는 모니터링만 한다. ROUGE-W만 오른 후보는 discard다.
- W&B 그래프는 4개만 만든다.

| 키 | 시점 | 내용 |
|---|---|---|
| `train/loss` | 매 step | 학습 loss |
| `eval/loss` | 32 step마다 | search에서 seed 3407로 뽑은 고정 128행의 loss(assistant token만, batch 1) |
| `search/em` | 후보 끝 1점 | search 전체 공식 EM |
| `search/rouge_w` | 후보 끝 1점 | search 전체 공식 ROUGE-W |

peak VRAM(GB)과 학습 시간(초)은 run summary 값으로만 남긴다. 다른 차트·테이블·아티팩트는 만들지 않는다.

`results.tsv` 열(탭 구분, git 제외): `commit`, `search_em`, `search_rouge_w`, `em_type1`, `em_type2`, `em_type3`, `empty_count`, `peak_vram_gb`, `train_seconds`, `eval_seconds`, `status`(keep, discard, crash), `description`(가설 한 줄).

`summary.md`: 후보마다 가설·결과·keep 여부를 한 줄씩 덧붙이고, 맨 위에 현재 최고 EM과 그 commit을 둔다.

## 4. 실행

fork 루트에서 `claude --permission-mode auto`를 시작하고 아래 프롬프트를 그대로 넣는다. 에이전트는 autoresearch 원본을 먼저 이해한 뒤 이 스펙대로 파일을 만들고 진단까지 끝낸다. 끝나면 다음에 실행할 명령을 출력하고 멈춘다.

```text
이 저장소는 karpathy/autoresearch를 fork한 것이다. 먼저 README.md, program.md, prepare.py, train.py를 읽고 deepwiki로 원본 저장소를 조회해서, 각 파일의 역할과 실험 루프(브랜치, commit과 reset, results.tsv)를 5줄로 요약한다. 그 다음 SPEC.md를 끝까지 읽고 아래 순서대로 진행한다. 사람에게 묻지 않고, 모르는 값은 SPEC.md 2절의 값을 쓴다. Unsloth API는 mcpdoc의 Unsloth 문서로 확인한다.

1. 환경 파악. OS, GPU 이름·VRAM·장수, 드라이버·CUDA 버전, Python·uv 버전, 디스크 여유를 확인해 표로 보고한다. GPU가 여러 장이면 CUDA_VISIBLE_DEVICES로 1장만 고정한다. Apple Silicon이면 MPS로 진행한다. bitsandbytes>=0.50, torch>=2.9를 쓰고 macOS 26 이상이면 kernels를 설치한다. optimizer는 adamw_torch로 바꾸고, 모델이 실제로 4bit로 올라갔는지 보고한다. NVIDIA GPU도 Apple Silicon도 없으면 여기서 멈추고 그 사실만 보고한다. WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT가 설정돼 있는지만 확인하고 값은 출력하지 않는다. git user.name/user.email이 설정돼 있는지 확인한다. Windows면 jq와 Git Bash가 있는지 확인한다.

2. 생성. SPEC.md의 2·3·5·6절을 따라 아래 파일을 만든다.
   - program.md: 원본 내용을 지우고 SPEC.md 2·3·5·6절의 규칙을 옮긴다. 루프에서 읽을 유일한 규칙 문서다.
   - pyproject.toml: uv, Python 3.11. Unsloth와 Qwen3.5-4B 학습·평가에 필요한 의존성을 넣는다.
   - prepare.py: 데이터셋 다운로드, 학습 풀 1,024행 선택, validation의 search/final 분할, KLUE dev JSON 다운로드와 ID 일치 확인, data/ 아래 ID 파일 저장.
   - klue_mrc_utils.py: KLUE-baseline 채점 함수 사본. 수정하지 않는다.
   - evaluate.py: --adapter <dir> --split search [--limit N]. --split은 search만 받는다. 새 프로세스에서 PeftModel로 adapter를 로드하고 tensor 일치를 assert한다. 전체·유형별 EM과 ROUGE-W, 빈 응답 수, 원답변 파일을 남기고 W&B run에 search/em, search/rouge_w를 기록한다.
   - train.py: QLoRA 학습. 상수 MAX_STEPS=128, BATCH_SIZE=4, GRAD_ACCUM=2, eval/loss batch 1, eval 뒤 empty_cache. 인자 --out <dir>, --max-steps N(진단용). W&B에 train/loss, eval/loss만 기록하고 peak VRAM과 학습 시간은 summary에 넣는다.
   - .claude/settings.json: SPEC.md 6절의 hooks와 env.
   - .gitignore: results.tsv, summary.md, runs/, wandb/, run.log, .loop-active, .env, unsloth_compiled_cache/. data/*.txt는 커밋한다.

3. 실행. uv sync, prepare.py. 그 다음 CPU 검사 3개: 학습 풀 전체가 4,096 token 이하인지, assistant token만 loss에 들어가는지, 답 없음 변환과 공식 채점이 예시 몇 개에서 기대대로 동작하는지. 그 다음 WANDB_MODE=disabled로 --max-steps 33 학습(step 32 eval을 지나 다시 학습하는 구간까지)과 search 3문항 평가로 실행 경로를 확인한다. 이어서 search에서 지문이 가장 긴 16문항을 같은 생성 함수로 돌려 peak VRAM과 시간을 잰다(진단용 임시 스크립트, 커밋하지 않음). 이 진단은 실험으로 세지 않고 W&B에도 남기지 않는다. 6절의 hook 검증 명령을 실행해 exit 코드를 확인한다.

4. 마무리. 저장소 루트에 빈 파일 .loop-active를 만든다. 생성한 파일을 git commit한다. 환경 표, 진단 결과(학습 시간, 평가 시간, peak VRAM, 긴 문항 16개 생성 시간으로 추정한 search 전체 평가 시간), 만든 파일 목록을 보고한다. 마지막으로 SPEC.md 5절의 재시작 명령과 /goal 본문을 코드 블록으로 출력하고 멈춘다. hooks는 세션 시작 때 읽히므로 이 세션에서는 루프를 시작하지 않는다.

제약. CUDA, GPU, kernel, OOM 오류가 나면 즉시 멈추고 오류 원문을 보고한다. 드라이버, 시스템 CUDA, 전역 Python을 바꾸지 않는다. 모든 설치는 uv 가상환경 안에서만 한다. .env와 키 값을 출력하지 않는다.
```

에이전트가 멈추면 사람이 아래 5개를 확인한다.

1. `train.py` 상수와 하이퍼파라미터가 2절 표와 같다.
2. `evaluate.py`의 `--split` 선택지에 `final`이 없다.
3. `data/` 아래 ID 파일 3개의 행 수가 보고와 같고, search ID와 final ID가 겹치지 않는다.
4. `.claude/settings.json`에 hook 2개와 env 2개가 있고, 6절의 검증 명령이 통과했다.
5. `.loop-active`가 있고 commit이 됐다.

## 5. 재시작과 /goal

Claude Code hooks는 세션 시작 때 읽힌다. 방금 만든 `.claude/settings.json`을 켜려면 Claude Code를 종료하고 같은 폴더에서 다시 시작해야 한다. 에이전트가 4절 끝에 출력하는 명령이 이것이다.

```bash
# Claude Code 종료 후 같은 폴더에서
claude --permission-mode auto
# 시작되면 /hooks 로 PreToolUse 2개를 확인하고 아래 /goal 본문을 직접 타이핑한다.
```

`/goal`은 터미널에 직접 타이핑하거나 `claude -p "/goal …"`로 넣는다. 클립보드 붙여 넣기는 일반 메시지가 되어 goal이 설정되지 않는다.

```text
/goal program.md를 먼저 끝까지 읽고 그 규칙대로 실험 루프를 돈다. 사람에게 묻지 않는다. 사람이 /goal clear로 멈출 때까지 후보를 계속 돌린다. 후보 수 상한은 없다.
첫 실험은 train.py를 수정하지 않은 baseline이다. 128 step 학습, 저장 adapter를 새 프로세스에서 search 전체 채점, results.tsv 첫 행, summary.md 작성.
이후 후보마다: 이전 결과와 search 원답변을 읽고 가설 1개를 세운다. train.py만 수정하고 git commit한다. 128 step 학습을 완주하고 search 전체를 채점한다. results.tsv에 1행을 추가한다. search EM이 지금까지의 최고값보다 엄격히 높을 때만 keep하고, 아니면 discard하고 git reset으로 마지막 keep commit에 돌아간다. summary.md를 갱신한다.
제약: baseline에서 CUDA, GPU, kernel, OOM 오류가 나거나 후보에서 OOM이 아닌 CUDA, GPU, kernel 오류가 나면 즉시 멈추고 오류 원문을 보고한 뒤 이 목표를 불가능으로 선언한다. 후보가 train.py 변경(예: rank·대상 모듈 증가)으로 OOM을 내면 crash로 기록하고 되돌린 뒤 다음 후보로 간다. 우회, 작은 모델, 드라이버 변경, 고정값 변경으로 오류를 피하지 않는다. 실행 1회가 40분을 넘으면 멈춤으로 보고 crash로 기록하고 되돌린다. final 분할의 정답·응답·점수를 열지 않는다. .env와 키 값을 출력하지 않는다. W&B에는 train/loss, eval/loss, search/em, search/rouge_w만 기록한다. prepare.py, 평가 코드, 분할 파일, 의존성을 바꾸지 않는다. 미리 정한 값 목록을 순서대로 돌리지 않는다. 결과를 부풀리지 않는다. 개선이 없으면 없다고 쓴다.
```

에이전트가 루프 중에 지켜야 하는 규칙은 `program.md`에 있다. 요지는 다음과 같다.

- GPU 작업은 한 번에 하나만 실행한다. 학습 출력은 `run.log`로 보낸다.
- 매 실험은 원 모델에서 새 LoRA로 시작한다. 이전 adapter를 이어서 학습하지 않는다.
- 학습 코드에 정답을 넣거나 평가 데이터를 학습에 쓰지 않는다.
- Bash로 보호 파일을 고치는 우회를 하지 않는다.
- 후보 설정 때문에 난 OOM은 crash로 기록하고 되돌린다. baseline OOM과 OOM이 아닌 GPU 오류만 루프를 멈춘다. 그렇지 않으면 rank를 올리는 후보 하나가 루프 전체를 끝낸다.

실행 1회 40분 제한은 예산이 아니라 멈춤 방지 규칙이다. 128 step 학습은 이보다 훨씬 짧지만, search 전체 평가(1,172문항 생성)는 16GB급 GPU에서 수십 분이 걸릴 수 있으니 진단에서 추정한 값과 비교한다.

## 6. hooks 요구사항

`.claude/settings.json`에 PreToolUse hook 2개와 env 2개를 둔다. `.loop-active`가 없는 준비 단계에서는 편집 hook이 아무것도 막지 않는다.

- 편집 차단: `.loop-active`가 있을 때 Edit/Write/MultiEdit/NotebookEdit 대상이 `train.py`, `results.tsv`, `summary.md`가 아니면 exit 2. `program.md`, `prepare.py`, `evaluate.py`, `klue_mrc_utils.py`, `data/`, `pyproject.toml`이 보호된다. 경로의 `\`는 `/`로 바꿔 비교한다(Windows).
- final 차단: Read·Grep·Glob의 경로나 패턴, Bash·PowerShell 명령에 `final_ids` 또는 `--split final`이 들어 있으면 exit 2.
- env: `BASH_DEFAULT_TIMEOUT_MS`와 `BASH_MAX_TIMEOUT_MS`를 2,400,000(40분)으로 둔다.

에이전트가 생성할 때 기준으로 삼는 예시다.

```json
{
  "env": { "BASH_DEFAULT_TIMEOUT_MS": "2400000", "BASH_MAX_TIMEOUT_MS": "2400000" },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "type": "command", "command": "[ -f .loop-active ] || exit 0; f=$(jq -r '(.tool_input.file_path // .tool_input.notebook_path // \"\") | explode | map(if . == 92 then 47 else . end) | implode'); case \"$f\" in */train.py|*/results.tsv|*/summary.md) exit 0;; *) echo \"루프 중에는 train.py, results.tsv, summary.md만 수정한다: $f\" >&2; exit 2;; esac" }]
      },
      {
        "matcher": "Read|Bash|PowerShell|Grep|Glob",
        "hooks": [{ "type": "command", "command": "jq -r '(.tool_input.file_path // \"\") + \" \" + (.tool_input.command // \"\") + \" \" + (.tool_input.path // \"\") + \" \" + (.tool_input.pattern // \"\")' | grep -qE 'final_ids|--split final' && { echo 'final 분할은 탐색 중에 열지 않는다' >&2; exit 2; } || exit 0" }]
      }
    ]
  }
}
```

검증은 hook 명령에 샘플 JSON을 stdin으로 넣고 exit 코드를 본다. 4절 3단계에서 에이전트가 실행하고, 사람도 같은 명령으로 확인할 수 있다. Claude Code는 세션 중에 만든 settings.json도 바로 반영할 수 있으므로, 명령 문자열에 `final_ids`를 그대로 쓰면 final 차단 hook이 검증 명령 자체를 막는다. 아래처럼 문자열을 나눠 쓴다.

```bash
H1=$(jq -r '.hooks.PreToolUse[0].hooks[0].command' .claude/settings.json)
H2=$(jq -r '.hooks.PreToolUse[1].hooks[0].command' .claude/settings.json)
F=final
touch .loop-active
echo '{"tool_input":{"file_path":"/w/prepare.py"}}' | sh -c "$H1"; echo "prepare.py -> $?"           # 2
echo '{"tool_input":{"file_path":"/w/train.py"}}'   | sh -c "$H1"; echo "train.py -> $?"             # 0
echo '{"tool_input":{"file_path":"C:\\w\\train.py"}}' | sh -c "$H1"; echo "win train.py -> $?"       # 0
echo '{"tool_input":{"file_path":"C:\\w\\evaluate.py"}}' | sh -c "$H1"; echo "win evaluate.py -> $?" # 2
echo "{\"tool_input\":{\"command\":\"cat data/${F}_ids.txt\"}}" | sh -c "$H2"; echo "final -> $?"     # 2
echo '{"tool_input":{"command":"uv run evaluate.py --split search"}}' | sh -c "$H2"; echo "search -> $?" # 0
rm .loop-active
```

## 7. 루프가 끝난 뒤

- `results.tsv`와 `summary.md`를 읽는다. keep 행이 없으면 최종 선택은 baseline이다.
- `git log`로 keep한 commit만 이어지는지 본다. discard한 후보는 reset으로 사라져 있어야 한다.
- 대화에서 hook이 막은 기록을 찾는다. 루프 중 `prepare.py` 편집 시도나 final 열람 시도가 있었으면 stderr 메시지가 남는다.
- final 분할은 `/goal` 범위 밖이다. `prepare.py`가 분할을 만들고 hooks가 열람을 막는다. 루프가 끝난 뒤 사람이 승인하면 base 모델과 선택한 adapter를 final에서 같은 조건으로 1회만 비교한다. 이때만 `evaluate.py`의 final 차단을 풀고, `.loop-active`를 지운다. final 결과를 보고 후보를 바꾸지 않는다.

## 8. 시간

장비별 수치는 적지 않는다. 진단과 baseline에서 학습 시간과 평가 시간을 재고, `results.tsv`의 `train_seconds`·`eval_seconds`로 남긴다. 그 값으로 하루에 몇 후보가 도는지 사람이 가늠한다.

수업 중에는 baseline과 후보 1~2개까지 함께 보고, 나머지는 돌려 둔다. 사람이 멈추고 싶으면 `/goal clear`를 입력한다.
