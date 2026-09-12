# 출처와 라이선스

## 프로젝트 코드

Copyright 2026 YoungjaeDev. 별도로 표시한 파일을 제외한 프로젝트 코드는 Apache License 2.0을 따른다. [LICENSE](LICENSE)를 참고한다.

## KLUE 데이터와 messages 변환본

- 저자: Sungjoon Park et al., [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/abs/2105.09680), 2021.
- 원본: [klue/klue의 고정 revision](https://huggingface.co/datasets/klue/klue/tree/349481ec73fff722f88e0453ca05c77a447d967c/mrc).
- 변환본: [YoungjaeDev/klue-mrc-messages](https://huggingface.co/datasets/YoungjaeDev/klue-mrc-messages/tree/ae6c27baff54df9d0a63ed85451badd6aefc131c).
- 라이선스: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/). 이 저장소의 코드 라이선스로 바뀌지 않는다.
- 변환: 원본 train·validation의 모든 행과 순서를 유지하고 지문·질문을 chat messages로 구성했다. 답은 첫 원문 정답, 답 없는 문제는 고정 문구를 사용한다. 원문을 요약하거나 윤문하지 않았다. autoresearch 준비 단계는 train 일부 ID와 평가 분할 ID를 별도로 기록한다.
- 데이터 파일은 Git에 포함하지 않고 실행 시 내려받는다. 재배포할 때 귀속, 라이선스, 변경 내역과 동일조건변경허락 요건을 유지한다.

## 공식 평가 코드

`klue_scorer/klue_baseline_utils.py`는 [KLUE-baseline](https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/metrics/utils.py)의 수정 없는 사본이다. 상류 Apache-2.0 전문은 [vendored LICENSE](klue_scorer/KLUE_BASELINE_LICENSE.md)에 보존했다. 원본의 귀속과 소스 hash는 [vendor README](klue_scorer/README.md)에 있다.

## 실행 중 사용하는 모델·도구

Qwen/Qwen3.5-4B revision은 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`로 고정한다. 모델은 이 저장소에 포함하지 않으며 [모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B)를 따른다. Unsloth·Transformers·PyTorch 등 외부 패키지의 라이선스는 각 배포본에 적용된다.

[karpathy/autoresearch](https://github.com/karpathy/autoresearch)의 에이전트 연구 흐름을 참고했다. 해당 프로젝트 소스는 복사하지 않았으며, 여기서는 고정 시간 대신 고정 30-step 비교를 사용한다.
