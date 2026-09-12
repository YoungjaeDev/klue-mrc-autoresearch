# 변경 기록

## [Unreleased]

- 공개 HF revision에서 KLUE messages와 원본 데이터를 받아 검증하는 CPU 준비 명령.
- 고정 30-step SDK reference와 에이전트가 직접 고치는 root `train.py`.
- GPU 직렬 lock, 소유 프로세스 종료, timeout 및 GPU 오류 중단 supervisor.
- 공식 EM·ROUGE-W와 탐색·최종 분할 계약 검사.
- 한국어 연구 지침과 기존 Windows Studio 환경 안내.
- GPU 실험 및 최종 성능 비교는 측정 대기.
- 선택적 tracking extra와 기본 local-only 수집기, 봉인된 공식 점수 연결.
- SDK reference부터 시작하는 best curve와 범위를 구분한 시간·VRAM PNG/SVG.
- 사용자 지정 비공개 W&B 프로젝트의 명시적 온라인 전송 경로(새 run 검증 대기).
