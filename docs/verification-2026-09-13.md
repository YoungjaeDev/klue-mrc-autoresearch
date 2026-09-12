# Windows helper GPU 진단 기록

2026-09-13에 추적되는 공개 Windows helper가 기존 Unsloth Studio 환경의 물리적 CPython을 직접 실행하는 경로를 확인했습니다. 로컬 증거는 Git에서 제외된 `outputs/publication-direct-runtime-gpu-01/`에 보존했습니다.

## 검증한 범위

| 단계 | 결과 |
| --- | --- |
| 1-step 학습 | completed, 1 optimizer step, 8 sample presentations |
| 시간 | 학습 호출 62.0175757초, 전체 학습 프로세스 103.7073666초 |
| GPU 메모리 | peak allocated 8,723,866,624 bytes |
| 학습 run SHA-256 | `a3fa0debf7a63ea79bd27d3fe9b1a8301d75e3a38ef529da68f2191d4b1363dc` |
| adapter model SHA-256 | `679c2fb61d86e0b61447bbb155d634cc268aaf2907d12bfaadd449f275f36910` |
| 별도 재로드 | completed, 저장 adapter로 search 앞 3문항 생성·공식 채점 |
| 재로드 시간·메모리 | 14.0374157초, peak allocated 8,661,578,752 bytes |
| scores SHA-256 | `4be9b254d7acd385a4ce9757d93432afd4f16b15a468c7faa72ee4c12a710020` |

학습과 평가 supervisor는 모두 return code 0과 completed 상태였습니다. `read_completed`로 평가 산출물 연결을 다시 검사했고, 재로드가 위 adapter를 사용했는지와 실행 전후 frozen hash가 같은지도 확인했습니다. 당시 전체 CPU 계약 검사 60개 중 53개가 통과했고 별도 모델 환경이 필요한 7개는 제외됐습니다. 실행 경로의 중요 검사 5개도 따로 통과했습니다.

## 확인한 환경

- Windows, NVIDIA RTX 5070 Ti 16GB
- Python 3.13.14
- torch 2.11.0+cu130, transformers 5.3.0
- Unsloth 2026.9.4, Unsloth Zoo 2026.9.3
- TRL 0.23.1, PEFT 0.18.1, bitsandbytes 0.50.2, triton-windows 3.6.0.post26
- Qwen3.5-4B 고정 revision, NF4 4bit
- 재로드 생성은 bfloat16·SDPA, greedy, thinking off, 최대 sequence 4,096, 최대 출력 128 token

## 한계

이 진단은 이미 설치된 한 Studio 환경에서 공개 helper의 프로세스 소유권, 1-step 실행, adapter 저장과 별도 재로드가 이어지는지 확인합니다. 새 PC의 Studio·CUDA·driver·모델 설치를 자동으로 재현하지 않습니다. 정식 30-step reference, 전체 search·final 평가, 성능 개선도 검증하지 않았습니다.

진단 첫 step은 warmup으로 학습률이 0이었습니다. 따라서 이 adapter나 3문항 점수로 학습 효과를 주장하지 않습니다. 3문항은 재로드·평가 연결을 확인하는 diagnostic prefix이며 공식 전체 성능 비교에 포함하지 않습니다.

## 공개 tracking 코드의 온라인 검증

GPU 진단과 tracking 검증은 범위가 다릅니다. `collect`는 위 공개 1-step 진단이 남긴 event 3개를 W&B에 전송했습니다. 첫 실행은 3개를 기록했고 재실행은 0개를 추가했습니다. 같은 원격 run의 history 3행이 로컬 event index·값과 일치했고, system metric과 artifact는 0개였습니다. remote history SHA-256은 `c52f3858607429369724377c62ecd66b43d1638c5a12b3fde77a6c2bf373a00a`입니다.

`score`는 기존에 완료된 private 30-step reference와 candidate 학습·전체 search 평가를 공개 tracking 코드로 읽는 상호운용 검사였습니다. 두 평가 모두 search 1,168문항과 같은 conditions SHA-256 `94045594601f9590f5f785632d596a21ca2e5028ba545f7dab7a71c79bf59823`을 사용했습니다.

| 입력 종류 | scores SHA-256 | 확인한 결과 |
| --- | --- | --- |
| reference | `b50056971ba279af9197690cdaf1e74ba05f5a0ba69bfa6b252bde810a1401ef` | 학습·adapter·평가 hash 연결, search baseline summary |
| candidate | `3c90467415b67edba16243f40e07aa0d25fcb8a353884858e88256ed568cc397` | 같은 조건과 hash 연결, strict EM 규칙의 `discard` summary |

두 W&B score summary는 서버 값과 일치했고 재실행은 기존 run을 재사용했습니다. score run의 history·system metric·artifact는 모두 0개입니다. tracking source SHA-256은 `4580602c8a328a13c1e1d15530c826db18b3679e7cd02966bfa54ae0ec623ab4`이며 관련 tracking CPU 검사는 19개 통과했습니다. 전체 CPU snapshot은 56개 통과, opt-in model 검사 7개 skip, 실패 0개였습니다.

첫 score 시도는 Windows와 POSIX 경로 구분자가 달라 artifact key 비교에서 네트워크 전에 중단됐습니다. 수정 뒤에는 구분자를 정규화하면서 key 충돌·경로 이탈·파일 hash 불일치를 계속 거부했고, 기존 학습·평가 manifest는 바꾸지 않았습니다.

이 온라인 검증은 collector가 GPU를 실행한 작업이 아닙니다. 공개 GPU 1-step·3문항 재로드 진단과, private 완료 산출물을 공개 tracking 코드가 처리한 검증을 합쳐 공개 30-step 성공이나 최종 benchmark로 해석하지 않습니다. 원응답, 모델 가중치, artifact, system metric은 W&B에 보내지 않았습니다.
