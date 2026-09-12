# 기여 안내

작은 변경을 먼저 제안하고 문제, 변경 동작, CPU 검증 결과를 PR에 적어 주세요. `uv sync --locked --python 3.11` 후 `uv run python -m unittest discover -s tests -v`로 검사합니다. GPU 결과를 주장하려면 모델 revision, 데이터·코드 hash, 명령, hardware, 생성 조건, 성공 artifact를 함께 기록하되 비밀값과 원본 로그를 무심코 공개하지 마세요.

공식 scorer와 고정 데이터는 임의로 고치지 않습니다. 새로운 데이터·평가 설계는 새 실험으로 분리하고 reference를 다시 측정합니다. 데이터 재배포에는 CC-BY-SA-4.0이 적용되며, 프로젝트 코드의 Apache-2.0과 구분합니다. 모델 가중치에는 해당 모델 라이선스를 확인하세요.

기여한 프로젝트 코드는 Apache-2.0으로 제공됩니다. 다른 프로젝트 코드를 가져올 때는 원래 라이선스와 귀속을 유지하세요. Studio 구현은 이 저장소에 복사하지 않습니다.
