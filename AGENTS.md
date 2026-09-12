# 작업 규칙

이 저장소는 한국어 Unsloth SDK autoresearch 실습 코드다. 먼저 README.md, program.md, docs/windows-runtime.md, docs/evaluation.md를 읽는다.

- CPU 환경은 `uv sync --locked --python 3.11`로 준비한다. GPU 환경은 별도이며 검증된 기존 Studio Python을 사용한다.
- 실행 승인이 있으면 program.md 안에서 가설·수정·학습·탐색 평가·판정을 이어 간다. 새로운 설치·환경 수리를 승인된 연구로 간주하지 않는다.
- GPU 학습과 평가는 하나씩 실행한다. `autoresearch_lab.run`의 동일 state directory/lock을 쓴다. CUDA/kernel 오류 또는 STOP_GPU가 있으면 즉시 멈춘다.
- 실행 중 frozen 파일을 변경하지 않는다. 모델·데이터·평가 hash와 후보별 코드 snapshot을 보존한다. 기존 산출물을 덮어쓰지 않는다.
- final은 선택 종료 후에만 평가한다. 평가 metric 구현이나 답변 정규화로 후보 점수를 조정하지 않는다.
- `.env`, 인증 파일, 실제 데이터·adapter·로그는 커밋하지 않는다. 비공개 파일을 공개 저장소로 자동 복사하지 않는다.
- 조사·CPU 검사·학습 성공·공개 상태를 구분한다. 로그와 artifact가 없는 성공을 쓰지 않는다.

## 검증

`uv run python -m unittest discover -s tests -v`는 CPU 계약 검사다. tracking extra가 있으면 PNG/SVG 통합 검사도 실행한다(`uv run --extra tracking python -m unittest discover -s tests -v`). 모델 통합 검사는 `RUN_MODEL_INTEGRATION=1`일 때만 실행되며 별도 ML 환경이 필요하다. CPU 검사 통과는 GPU 성공을 뜻하지 않는다. `uv run python prepare.py`는 익명 데이터 다운로드와 원문 무결성 검사다.

W&B·그래프는 docs/tracking.md를 따른다. 기본 local-only를 유지하고 `--online`을 요청한 명령만 사용자가 지정한 비공개 프로젝트에 전송한다. 원문·가중치·코드·시스템 정보 업로드 설정을 켜지 않는다. tracker가 임의로 새로운 연구를 시작하거나 final 분할을 열지 않는다.

## Code Review Rules

리뷰는 변경된 동작과 실제 회귀를 우선한다. 데이터 누출, hash 검증 누락, frozen 코드 변경, 프로세스 소유권 없는 종료, 예산 우회, 다른 생성 조건 비교, 응답 label 손실을 점검한다. 해당 파일·조건·영향을 설명하고 근거 없는 성능 개선을 승인하지 않는다. vendored 공식 scorer는 원문과 hash를 유지한다.
