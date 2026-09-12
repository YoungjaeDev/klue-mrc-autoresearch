# 작업 규칙

한국어 Unsloth SDK autoresearch 실습 저장소다. 먼저 README.md와 program.md를 읽는다.

- 연구 규칙은 program.md를 따른다. 에이전트는 `train.py`만 수정한다.
- GPU 작업은 `run.py`로 하나씩 실행한다. GPU·CUDA·kernel·OOM 오류나 `STOP_GPU.json`이 있으면 즉시 멈춘다.
- 드라이버·시스템 CUDA·전역 Python을 바꾸지 않는다. 설치는 프로젝트 `.venv` 안에서만 한다.
- final 분할은 열지 않는다. 채점 코드와 정답 변환을 바꾸지 않는다.
- `.env`, 토큰, 데이터, adapter, 로그는 커밋하지 않는다.
- GPU 패키지를 설치한 `.venv`에서는 모든 명령을 `uv run --extra gpu`로 실행한다. `--extra gpu` 없이 `uv run`을 쓰면 uv가 GPU 패키지를 제거한다.
