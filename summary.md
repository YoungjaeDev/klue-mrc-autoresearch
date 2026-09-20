# KLUE-MRC autoresearch summary

루프 종료(사람이 /goal clear). 2026-09-19 13:09 ~ 09-20 19:50, 약 30.7시간, 후보 132회(keep 8, crash 1, 나머지 discard). 최종 상태는 commit 2d674bf.

> **핵심 결론**: 사실상 무변화여야 할 섭동 3회가 모두 최고값보다 3~5 낮았다(-4.01, -2.90, -4.95). 79.69는 이 설정의 대표값이 아니라 잡음이 위로 튄 한 번의 값이고, 같은 설정의 기대 EM은 대략 75~78이다. baseline 76.02와 비교해 **이 루프에서 실질적 개선은 없었다**.

> **잡음 측정 3회**: (1) 9429ea5 — 모든 step이 이미 clip되는 상태에서 max_grad_norm 1.0→0.5(Adam에선 이론상 거의 무변화): 79.69→75.68(-4.01), 빈 응답 340→434, 원답변 179/1172개 변경. (2) 5410eab — weight_decay 0.001→0(전체 학습 동안 가중치 0.002% 변화): 79.69→76.79(-2.90), train loss도 0.2373→0.2433. (3) ade8d55 — weight_decay 0.001→0.0005: 79.69→74.74(-4.95), 빈 응답 452. 즉 수치 수준 섭동만으로 search EM이 3~5 흔들린다. baseline 이후 keep된 모든 향상(+0.17~+0.94)은 이 잡음 폭 안이며, 현재 최고값은 잡음의 위쪽 끝일 가능성이 크다. **baseline 대비 실질적 개선이 있었다고 말할 근거는 없다.**

**현재 최고 search EM: 79.69 (commit 2d674bf)** — lr 2.5e-4 linear, warmup 3, adamw_8bit, clip 1.0, dropout 0.1, LoRA attention(q/k/v/o+in_proj_qkv/in_proj_z/out_proj) r16 a16 + MLP(gate/up/down) r64 a64

| commit | 가설 | search EM | ROUGE-W | 결과 |
|---|---|---|---|---|
| fb0e7a3 | baseline (train.py 수정 없음: lr 2e-4 linear, r16 a16, q/k/v/o/gate/up/down) | 76.02 | 79.67 | keep |
| 18c33e7 | 실패의 60%가 답함/답없음 판단 오류(빈 답 96, 답없음에 답 73). linear-attn 24층에 attn LoRA 없음 → in_proj_qkv/in_proj_z/out_proj 추가 | 76.19 | 80.01 | keep (+0.17 = 2문항, 잡음 수준일 수 있음; 문항 flip +39/-37) |
| a741aaf | train loss 0.18 평탄, eval loss step 96 최저 후 상승 → 과적합. lora_dropout 0 → 0.1 | 76.54 | 80.12 | keep (+0.35 = 4문항; eval loss 끝까지 단조 감소, 빈 응답 393→374, flip +28/-24) |
| a763dc5 | dropout으로 과적합이 사라졌으니 용량 여유 → r/alpha 16→32 | 77.47 | 81.20 | keep (+0.93 = 11문항; eval loss 0.246 최저, 빈 답 오류 82→70, 답없음에 답 77→84, flip +61/-50) |
| 9e1abf0 | a763dc5에서 eval loss가 step 128까지 가파르게 하강(0.286→0.246) → 미수렴으로 보고 lr 2e-4→4e-4 | 76.62 | 80.17 | discard (step 32 eval loss 0.876로 초반 불안정, 최종 eval loss 0.291로 악화) |
| 763bbf7 | r16→32가 +0.93, 과적합 신호 없음 → r/alpha 64 | 76.62 | 79.96 | discard (step 32 eval loss 0.517로 초반 불안정, 최종 0.260) |
| 6906272 | step32 eval loss가 업데이트 크기와 함께 커짐(r16 .356, r32 .387, r64 .517, lr4e-4 .876) → warmup 3→16 | 77.05 | 80.61 | discard (최종 eval loss 0.250, 빈 응답 355→398로 과잉 기권, type3↑ type1↓) |
| 94bd5e7 | 최다 오류가 답함/기권 판단(FP 84, FN 70) → DeltaNet 기억 게이트 in_proj_a/in_proj_b에도 LoRA | 76.19 | 80.02 | discard (eval loss 0.243로 최저였지만 빈 응답 408로 과잉 기권; eval loss와 EM이 따로 움직임) |
| 0303d44 | 2e-4에서 미수렴, 4e-4는 불안정 → 같은 최고 lr로 누적 학습량을 늘리는 WSD(유지 후 마지막 40 step 선형 감쇠) | 74.15 | 77.47 | discard (eval loss 0.242 최저였지만 빈 응답 453으로 과잉 기권 심화) |
| 4066034 | grad norm 2~5가 128 step 중 126 step에서 1.0으로 잘림 → 답 span 배치 가중치가 눌려 과잉 기권? max_grad_norm 5.0 | 76.19 | 80.05 | discard (빈 응답 355→336으로 방향은 맞았으나 과도: type2 79.4↑, type3 73.4↓) |
| 1d2c092 | clip 1.0은 과잉 기권, 5.0은 과소 기권 → 중간 3.0 | 77.05 | 80.85 | discard (빈 응답 362로 예상과 달리 사이에 오지 않음 → clip과 기권 수는 단조 관계 아님; eval loss 0.241 최저였으나 EM 개선 없음) |
| 104bbff | 128 step은 짧아 Adam 모멘트가 덜 안정 + 8-bit 상태 양자화 잡음 → adamw_torch(fp32 상태) | 77.30 | 80.42 | discard (최고 77.47에 2문항 못 미침, 빈 응답 393) |
| fd08d4d | 업데이트를 키운 변경(lr 4e-4, r64, clip 해제)이 모두 손해 → lr 1e-4 | 76.71 | 80.78 | discard (eval loss 0.278로 미학습; lr은 2e-4가 1e-4·4e-4보다 나음) |
| 776ed27 | WSD(후반 lr 높음)가 -3.3 → 반대로 후반 lr이 linear보다 낮은 cosine | 77.90 | 82.18 | keep (+0.43 = 5문항, 잡음 범위 안; ROUGE-W 최고, 빈 응답 351 최저; eval loss 0.270으로 오히려 높음) |
| 3ff18d6 | 후반 lr이 낮을수록 좋았음(WSD<linear<cosine) → 더 가파른 polynomial power 2 감쇠 | 76.11 | 79.71 | discard (lr 총량이 cosine의 2/3로 줄어 손해; lr 1e-4와 같은 방향 → 후반 lr보다 총량이 중요, cosine이 현재 최적) |
| 2df7232 | 최고 run이 과반 run이 맞힌 52문항을 놓침(고분산) → lora_dropout 0.2 | 77.13 | 80.70 | discard (분산 감소 효과 없음) |
| b46617b | MLP LoRA가 답 형식을 외워 긴 답(39)·고분산 유발? → attention 계열만 LoRA | 75.94 | 80.04 | discard (-1.96; MLP LoRA가 성능의 큰 몫) |
| 1fcdc23 | MLP LoRA가 중요, r64 전체는 불안정 → MLP만 rank/alpha 64(rank_pattern, 1-step 진단으로 적용 확인) | 77.56 | 81.36 | discard (빈 응답 336, type2 82.0↑ type3 74.3↓; 기권/응답 균형 이동에 그침) |
| bc5b22f | attention LoRA 추가 효과가 잡음/손해 → attention LoRA가 검색을 흔들어 FP 유발? MLP만 LoRA | 75.77 | 79.93 | discard (-2.13; attention-only -1.96과 함께 보면 두 쪽 모두 필요) |
| 36a01a6 | WSD 실패는 후반 lr 탓 → cosine 유지하고 peak 2.5e-4로 총량만 증가 | 76.37 | 80.58 | discard (lr 1e-4/2.5e-4/4e-4 모두 2e-4보다 낮음) |
| b3fe033 | MLP-64는 응답 쪽으로 균형 이동 → 반대로 attention(q/k/v/o+DeltaNet) rank/alpha 64, MLP 32 | 74.23 | 78.15 | discard (-3.67; r64 전체의 불안정은 attention 쪽 용량 증가 탓) |
| d4697c3 | attention 용량이 해로움 → attention rank/alpha 16, MLP 32 | 77.47 | 81.21 | discard (attention rank 16/32/64 = 77.47/77.90/74.23) |
| 92d297a | attention 용량이 해로움 → 정밀 검색 담당 full-attn 8층의 q/k/v/o LoRA 제거 | 77.47 | 81.42 | discard (빈 응답 372, type3↑ type1↓) |
| 2aec72f | lr·attention 용량 증가는 해롭고 MLP 용량은 견딤 → MLP alpha 64(r32, scale 2) | 76.02 | 80.71 | discard (빈 응답 311, type1 77.8↑ type3 71.0↓: MLP 업데이트 크기가 응답 쪽 기준선을 민다) |
| 5ec6836 | run별 분해: cosine은 기권 판별 최고(293/369)지만 답 정밀도 낮음(0.832), MLP-64는 정밀도 최고(0.857) → attention 16 + MLP 64로 용량 이동 | 78.07 | 81.92 | keep (+0.17 = 2문항, 잡음 수준; 정밀도 0.856·기권 정답 296으로 예측 방향은 맞음, 대신 답한 문항 745→723) |
| 1a914ae | 빈 응답 376 > 답없음 369 → MLP alpha 96(scale 1.5)으로 응답 쪽 이동 | 76.11 | 80.67 | discard (빈 응답 349로 이동은 예측대로, 그러나 정밀도 0.856→0.831·기권 정답 296→281로 질 저하) |
| 04ba3fc | MLP r64가 정밀도를 올림 → MLP rank/alpha 128 | 74.83 | 79.00 | discard (정밀도 0.825, train loss 0.254; MLP 용량은 64가 적정) |
| dff94ba | MLP scale 1→1.5가 정밀도를 낮춤 → scale 0.75(alpha 48)로 정밀도↑ 기대 | 77.56 | 81.26 | discard (예측과 반대: 빈 응답 356↓, 정밀도 0.842↓ → scale 효과는 단조가 아니고 잡음과 구분 안 됨) |
| 356bc85 | batch 8·grad norm 2~5로 기울기 잡음이 큼 → adam beta1 0.95로 평활 | 76.79 | 80.98 | discard (답있음 정답 642로 전 run 최고, type1 79.5, 그러나 빈 응답 293·type3 69.9로 과소 기권) |
| 8b7e8a7 | beta1 0.9(빈 응답 376)와 0.95(293) 사이 → 0.92로 균형 | 77.47 | 81.08 | discard (빈 응답 388로 두 점 밖 → 기권 기준선이 작은 변경에 ±40 수준으로 무작위 이동) |
| 8a9b0d6 | 기권 기준선의 무작위 이동은 기울기 크기 잡음 탓 → 부호 기반 lion_8bit, lr 5e-5 | 76.37 | 80.49 | discard (빈 응답 344, 전반적 정확도 하락) |
| 9baf968 | MLP-64 설정 train loss 0.2325 > r32의 0.2214 → 과소적합, dropout 0.05 | 76.02 | 80.00 | discard (train loss 0.2346으로 그대로, 빈 응답 303·type3 70.5) |
| b185dbc | attention 용량 64가 크게 손해 → attention rank/alpha 8, MLP 64 | 77.47 | 81.56 | discard (attention 8/16 차이 잡음 수준) |
| d5b3353 | 게이트 모듈(in_proj_a/b) LoRA가 손해 → DeltaNet 출력 게이트 in_proj_z도 제외 | 77.05 | 81.37 | discard (빈 응답 330, type3 74.5↓) |
| d4f1b7a | warmup 3→16이 빈 응답을 늘림, 현재 376으로 과잉 기권 → warmup 0 | 77.82 | 81.87 | discard (빈 응답 359로 예측대로 이동, train loss 0.214↓, EM은 개선 없음) |
| 666bf43 | 후반 grad norm ~2.3이 1.0에서 잘림 → clip 2.0으로 저 lr 정밀 구간 기울기 복원 | 77.39 | 81.57 | discard (빈 응답 387로 오히려 증가) |
| acaf86d | attention 교란이 해로움 → 어휘 매칭 담당 하위 8층(0~7) LoRA 제외(regex target, 1-step 진단으로 8~31층만 확인) | 75.94 | 80.43 | discard (-2.13; 하위층 LoRA 필요) |
| b4bb6fd | 상위 8층(24~31)이 기권/응답 첫 token을 정해 기준선이 흔들림? → 0~23층만 LoRA(regex, 진단 확인) | 77.30 | 81.80 | discard (-0.77, 잡음 수준; 상위층 제외 영향은 하위층보다 작음) |
| 7b219df | 하위층 LoRA가 더 중요 → 0~7층 MLP만 rank/alpha 128 | 76.79 | 80.93 | discard (빈 응답 331·type3 74.5; 첫 시도 cc45c1d는 rank_pattern 키 겹침으로 평가 로드 실패 → 키를 배타적으로 고쳐 재실행) |
| 800ee2f | MLP에선 scale 효과가 rank보다 큼 → attention alpha 8(r16, scale 0.5)로 attention 교란 축소 | 76.62 | 81.03 | discard (빈 응답 343, type2 75.6↓) |
| a37b80f | 128 step에서 beta2 0.999면 초반 큰 기울기가 v를 지배해 후반 업데이트가 숨은 감쇠 → beta2 0.99 | 77.56 | 81.05 | discard (빈 응답 360, 잡음 범위) |
| d42ccfb | cosine 이득(기권 판별)은 MLP32에서 측정됨. MLP64에서는 후반 lr이 높은 linear가 정밀도에 유리할 수 있음 → linear로 복귀 | 79.01 | 82.40 | keep (+0.94 = 11문항; 정밀도 0.867 전 run 최고, 답있음 정답 619→630, 기권 정답 296 유지; flip +66/-55라 잡음과 완전히 구분되지는 않음) |
| 20531e7 | cosine→linear(후반 lr↑)가 정밀도를 올림 → 한 단계 더 높은 후반 lr(sqrt 감쇠) | 75.17 | 78.38 | discard (빈 응답 481로 과잉 기권 폭증; WSD 453과 같은 패턴 → 후반 lr이 높으면 기권 과다, linear가 적정) |
| 18975c0 | cosine에서 warmup 0이 빈 응답을 줄임, 최고는 372 > 369 → linear/MLP64에서도 warmup 0 | 76.96 | 80.43 | discard (예측과 반대로 빈 응답 412↑) |
| c7fc148 | 최고 run 오류 최다가 type1 빈 답(51/76) → 매칭 담당 attention rank/alpha 32로 복원 | 74.83 | 78.14 | discard (빈 응답 432↑ → attention 용량↑은 기권↑, MLP 용량·scale↑은 응답↑) |
| d4bcd5f | MLP 32→64가 정밀도↑, 128은 과함 → MLP rank/alpha 96 | 76.02 | 79.22 | discard (빈 응답 415↑로 예측과 반대; 'MLP↑=응답↑' 경향도 예측력이 없음) |
| 65e0fbb | dropout은 r16 시절 +4문항 이득뿐, attention 16·MLP 64에선 MLP span 학습을 약화할 수 있음 → dropout 0 | 77.30 | 81.05 | discard (빈 응답 401) |
| f477167 | MLP64로 LoRA가 커짐, 0 초기화 B의 작은 초기 기울기가 8-bit 블록 양자화에 뭉개짐 → adamw_torch(fp32) | 78.58 | 82.16 | discard (최고 79.01에 5문항 부족, 빈 응답 380) |
| e776bee | type1 빈 답은 매칭 문제, DeltaNet까지 키우면 기권↑ → full-attn q/k/v/o만 32, DeltaNet 16 | 76.96 | 80.37 | discard (빈 응답 408↑; attention 용량 증가는 3번 모두 기권↑(371/432/408) — 일관된 몇 안 되는 경향) |
| 2d674bf | MLP64에선 후반 lr↑(cosine→linear)가 정밀도↑, sqrt 감쇠는 기권 폭증 → linear 유지, peak 2.5e-4 | 79.69 | 83.26 | keep (+0.68 = 8문항; 답있음 정답 649 전 run 최고, 정밀도 0.868, 빈 응답 340·기권 정답 296→285; flip +58/-50으로 잡음 가능성 남음) |
| e0895ab | 최고가 과소 기권(340<369), attention 용량↑=기권↑ → attention rank/alpha 24 | 77.47 | 81.23 | discard (빈 응답 392로 방향은 맞았으나 과도, type1 73.5↓) |
| 6c96240 | 2e-4→2.5e-4가 +8, 과거 4e-4 불안정은 attention 용량 탓이었고 지금은 16 → linear peak 3e-4 | 75.34 | 79.03 | discard (step32 eval loss 0.41, -4.35; linear peak은 2.5e-4 근처가 좁은 최적) |
| 2f78b5f | lr을 올릴수록 초반 불안정 → lr 2.5e-4에 warmup 8 | 78.33 | 82.22 | discard (빈 응답 386) |
| 1f78eb5 | 최고가 과소 기권(340), attention rank 24는 과도(392) → attention alpha 20(scale 1.25)로 소폭 | 77.13 | 80.45 | discard (빈 응답 371로 목표 369에 정확히 맞췄지만 EM↓ → 기권 개수 보정만으로는 EM이 오르지 않음, 답 정밀도와 판별 정확도가 관건) |
| 1ca0d18 | lr 2.5e-4로 업데이트↑ → dropout 0.15로 규제 | 76.02 | 79.51 | discard (빈 응답 447로 과잉 기권) |
| bbe142b | 2.5e-4로 기권 정답이 296→285로 감소, cosine이 기권 판별 최고 → cosine + peak 2.5e-4 | 75.60 | 79.22 | discard (train loss 0.258↑; step 32 부근 lr이 ~2.1e-4를 넘은 run은 모두 나빴음: cosine2.5e-4, linear3e-4) |
| 8af56fd | step 32 lr이 높으면 손해, 후반 lr은 정밀도에 도움 → linear 감쇠를 10% 하한(2.5e-5)에서 멈춤 | 76.28 | 79.29 | discard (빈 응답 450; WSD 453·sqrt 481과 같음 → 끝에서 lr이 0으로 가지 않으면 과잉 기권, 일관된 경향) |
| e4f76be | eval loss 미수렴, 전체 lr↑은 초반 불안정 → LoRA+(lora_B lr ×4, 파라미터 그룹 8-bit AdamW; 2-step 진단 확인) | 71.42 | 75.67 | discard (step32 eval loss 0.775, train loss 0.338; B만 키워도 초반 폭주) |
| c97c0d7 | 초반 불안정이 반복 실패 원인, 0 초기화 B의 작은 기울기 업데이트 과대 → adam epsilon 1e-6 | 76.37 | 80.03 | discard (step32 eval loss 0.414로 안정화 안 됨) |
| cbd93ba | 오류 246 중 149가 첫 token 기권/응답 판단 → lm_head에 LoRA | - | - | crash (1-step 진단에서 Unsloth가 lm_head를 modules_to_save로 옮겨 248320×2560 tied 전체 가중치를 학습함을 확인; LoRA가 아니고 evaluate.py의 LoRA 텐서 수 검사도 통과 못함 → 실행하지 않고 되돌림) |
| fa1fe75 | 2e-4→2.5e-4 이득이 attention인지 MLP인지 분리 → MLP alpha 48(MLP 실효 lr ~1.9e-4) | 76.02 | 79.51 | discard (-3.67 → 이득은 MLP 쪽 lr에서 옴) |
| b190b43 | 이득은 MLP lr에서, 3e-4 실패는 attention 탓 → MLP alpha 77(MLP 실효 lr ~3e-4) | 76.88 | 80.10 | discard (MLP 실효 lr 1.9e-4/2.5e-4/3e-4 = 76.02/79.69/76.88; 좁은 최적) |
| 3cab54e | beta1 0.95는 응답 쪽으로 기움, 최고는 과소 기권 → beta1 0.85 | 76.45 | 80.14 | discard (빈 응답 400으로 방향은 맞았으나 과도, type1 72.3↓) |
| 18f9b5b | 게이트 LoRA는 기권↑, 최고는 과소 기권 → in_proj_a/b를 rank 16으로 추가 | 77.39 | 80.88 | discard (빈 응답 415로 방향은 맞았으나 과도, type3만 84.6↑) |
| 3101d93 | 이득은 MLP lr, 3e-4 실패는 attention → attention alpha 12(scale 0.75, 실효 ~1.9e-4) | 77.22 | 80.40 | discard (빈 응답 421↑; attention scale↓인데도 기권↑) |
| 7f0891f | Adam 계열 기울기 잡음이 기권 기준선을 흔듦 → adafactor(분해 2차 모멘트, momentum 없음), lr 2.5e-4 | 78.33 | 81.91 | discard (빈 응답 370으로 균형, eval loss 0.240, train loss 0.263로 과소학습 신호) |
| c706492 | adafactor 과소학습(train loss .263), momentum 없음+update clipping → lr 4e-4 | 77.13 | 81.11 | discard (train loss 0.301로 오히려↑ → 2.5e-4도 이미 과함) |
| 6baf341 | adafactor lr↑에 train loss↑ → lr 1.5e-4 | 77.73 | 81.53 | discard (train loss 0.225↓로 예측대로, EM 개선 없음; adafactor 3회 78.33/77.13/77.73) |
| 9429ea5 | 잡음 측정: 모든 step이 clip되고 Adam은 스케일 불변 → max_grad_norm 0.5는 거의 무변화여야 함 | 75.68 | 79.09 | discard (-4.01, 원답변 179개 변경, flip +46/-93 → 잡음 폭 ±2~4 EM) |
| a4a343c | cosine 기준에서 beta1 0.95가 답있음 정답 642 → linear/2.5e-4 기준에서 재시험 | 76.28 | 80.65 | discard (빈 응답 363) |
| 5527a9e | 8-bit 상태 양자화 도약이 혼돈적 민감도를 키울 수 있음 → adamw_torch(fp32) at 2.5e-4 | 77.99 | 81.39 | discard (빈 응답 391) |
| af6195b | 최종 eval loss 최저 2개 변경(MLP48 0.2245, warmup8 0.2347) 결합, eval loss↔EM 상관 -0.56 | 75.85 | 79.62 | discard (eval loss 0.2516으로 효과가 더해지지 않음) |
| 5c7c46d | lr 3e-4가 과함 → lr 2.5e-4 유지, MLP rank/alpha 48로 업데이트 세기만 소폭 축소 | 78.24 | 81.48 | discard (빈 응답 366) |

## 2d674bf(79.69) 이후 (전부 discard, 60+ 후보)

기록은 `results.tsv`에 모두 있고, 여기에는 묶어서 남긴다. 최고값은 갱신되지 않았다.

- 모듈 구성: MLP 전용 78.84, DeltaNet 전용 attention 79.01, down_proj 전용 77.39, gate·up 전용 74.40(-4.4), 균일 rank 32 76.88. attention 계열은 빼도 되고, MLP 중에서는 down_proj가 필수.
- 층 범위: MLP 상위 16층만 70.73(-8.1), 하위 16층만 76.88(답있음 최고, 기권 부족), 0~23층 76.45, 0~11층+하한 77.22. attention은 상·하위 절반 모두 76.96으로 동일.
- 용량: MLP rank 32/48/56/64/96/128 = 78.58/78.24/77.65/79.69/74.40(lr 2e-4)/74.83, attention rank 0/4/8/16/24/32/64 = 78.84/78.24/77.47/79.69/77.47/74.83/74.23.
- 스케줄: lr 하한을 2.5%로 둬도 빈 응답 447(10% 하한 450)로 동일 → 하한 크기가 아니라 "lr이 0으로 끝나는지"가 기권을 좌우. step 112에서 0으로 만들면 기권은 유지되나 EM 74.83으로 하락(실질 학습 단축).
- optimizer/기타: adafactor 3회 77.1~78.3, fp32 AdamW 4회 77.3~78.6, lion 76.37, LoRA+ 71.42, beta1 0.85/0.92/0.95 76.5/77.5/76.3, epsilon 1e-6 76.37, clip 0.5/2/10 75.68/77.39/76.54, warmup 0/8/16 77.82/78.33/77.39, dropout 0/0.05/0.15 77.30/76.71/76.02.
- 조합 시도(낮은 eval loss 설정끼리, 답 강점+기권 보정 등) 5회 모두 실패. 효과가 더해지지 않는다.
- 분산 축소 시도도 효과 없음: LoRA 가중치 EMA(decay 0.95/0.99) 77.47/76.88, fp32 학습(bf16 off) 77.47. bf16 반올림은 흔들림의 주원인이 아니다.
- lr 곡선(linear, MLP64+attn16): 1e-4 76.71 / 2e-4 79.01 / 2.25e-4 76.79 / 2.5e-4 79.69 / 2.75e-4 77.99 / 3e-4 75.34. 3e-4만 잡음 밖이고 나머지는 구분되지 않는다.

참고: 후보 간 문항 단위 flip이 수십 개라 ±0.3 EM 안팎은 잡음과 구분이 어렵다. 12개 run 교차 집계: search 1,172문항 중 661개는 모든 run에서 정답, 124개는 모든 run에서 오답(빈 답 47, 답없음에 답 33, 틀린 답 28, 경계 16), 나머지 387개가 run마다 뒤집힌다. eval loss는 search EM과 상관이 약하다.

반복 관찰(65 run):
- 후보 차이는 대부분 기권/응답 기준선 이동으로 나타난다. 빈 응답 수는 작은 변경에도 290~480 사이로 움직인다.
- 비교적 일관된 방향: 게이트(in_proj_a/b) LoRA, 끝에서 lr이 0이 아닌 스케줄(WSD/sqrt/10% 하한) → 기권↑. attention 용량↑도 대체로 기권↑였지만 attention scale↓(3101d93)에서도 기권↑(421)이 나와 방향 규칙으로 믿기 어렵다.
- W&B 71 run 집계: 최종 eval loss와 search EM 상관 -0.56(step32 eval loss는 -0.28). 앞서 적은 "eval loss와 EM은 따로 움직인다"는 소수 run 기반 오판이었다. eval loss가 EM보다 덜 흔들리는 방향 지표다. 단 미세 구간에서는 eval loss도 단조롭지 않다(MLP rank 48/56/64 = 0.2245/0.2471/0.2386).
- optimizer는 잡음 이상 차이 없음: MLP48 동일 설정에서 8-bit AdamW 78.24 / fp32 AdamW 78.33 / adafactor 77.30. 계열 평균 차이(fp32 78.05, adafactor 77.62, 8-bit 76.7)는 8-bit 쪽에 실패한 탐색 설정이 몰려 생긴 착시.
- 용량 곡선(lr 2.5e-4, linear): MLP rank 32/48/56/64 = 78.58/78.24/77.65/79.69(모두 잡음 안), attention rank 4/8/16/24/32/64 = 78.24/77.47/79.69/77.47/74.83/74.23(32 이상만 확실히 나쁨).
- attention LoRA 손해의 원인은 업데이트 크기가 아니라 파라미터 수: rank 32에서 scale 1.0→0.5로 낮춰도 74.83→74.91로 동일(잡음보다 큰 -4.8 하락 유지).
- attention LoRA는 이 과제에서 보탬이 없다: 현재 기준(linear 2.5e-4, MLP 64)에서 attention rank 0/4/8/16/24/32/64 = 78.84/78.24/77.47/79.69/77.47/74.83/74.23. 빼도 잡음 안이고 32 이상만 확실히 손해. 층 절반만 붙여도 위/아래 모두 76.96으로 동일.
- 모듈 ablation(linear 2.5e-4, rank 64): MLP 3모듈 78.84 / down_proj만 77.39 / gate·up만 74.40. down_proj를 빼면 -4.4(잡음보다 큼), down_proj 하나만으로 충분. 학습이 실제로 일어나는 자리는 down_proj.
- MLP LoRA는 전 층에 필요: 상위 16층에만 붙이면 78.84→70.73(-8.1), eval loss 0.242→0.321. attention을 절반으로 줄였을 때(영향 없음)와 대조된다.
- 최고 설정 주변의 거의 모든 섭동이 76~78로 떨어진다. 최고값이 운 좋은 지점일 가능성이 크고, 이후 후보는 평균 회귀에 지배될 수 있다.
- 기준선을 목표(369)에 맞춰도 EM은 오르지 않았다(1f78eb5: 빈 응답 371, EM 77.13). 개선은 답 정밀도(맞힌 답/낸 답)와 기권 판별 정확도가 함께 좋을 때만 나왔다.
- 개선이 나온 축: MLP 용량(64), linear 감쇠, MLP 쪽 실효 lr 2.5e-4(1.9e-4·3e-4 모두 -3 안팎). 좁은 최적이라 우연 비중을 배제할 수 없다.

해석 주의: 작은 설정 변경에도 빈 응답 수가 ±40 수준으로 흔들리고(beta1 0.9/0.92/0.95 = 376/388/293), 후보 대부분이 77.5±0.5 안에 모인다. a763dc5 이후 keep(776ed27 +5문항, 5ec6836 +2문항)은 잡음과 구분되지 않는다. 같은 search 분할로 keep을 반복 판정했으므로 최고값에는 선택 편향이 섞여 있다.
