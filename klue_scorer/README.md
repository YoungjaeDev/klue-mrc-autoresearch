# KLUE 공식 채점 코드

`klue_baseline_utils.py`는 공식 [KLUE-baseline](https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/metrics/utils.py)의 `klue_baseline/metrics/utils.py`를 수정 없이 복사했다.

- Commit: `8a03c9447e4c225e806877a84242aea11258c790`
- LF 기준 SHA-256: `6ec7e78e0e9687f1601058d23f3e270c281aaa3f3cf9db686705e0ebb723221c`
- License: [Apache-2.0 원문](KLUE_BASELINE_LICENSE.md)
- 의존성: NumPy. 과거 BERT 학습기·PyTorch Lightning 전체를 설치할 필요는 없다.

`evaluate_for_klue_mrc`를 직접 호출한다. 모델 출력 연결, 누락 ID 검사, 0–100 점수 표시, 유형별 집계는 프로젝트의 별도 연결 코드에서 담당한다. 원본 채점 함수의 정규화·EM·ROUGE-W 계산은 변경하지 않는다.
