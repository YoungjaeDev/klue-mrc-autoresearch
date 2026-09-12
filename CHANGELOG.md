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
- 공식 점수와 canonical SDK 학습 artifact·명시적 Studio adapter의 파일별 hash 연결 검사.
- 준비 manifest에 운영 코드·program·의존성 freeze를 포함하고 generated 출력 재사용 조건을 명확화.
- 기존 정상 Studio 환경에서 프로젝트 내부 Transformers 5.3 overlay를 만드는 검증된 선택 명령.
