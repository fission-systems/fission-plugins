# G1: GGUF 변환·양자화·생성 확인 (2026-11-09)

`docs/PLAN_M2.md` 3절의 게이트. 학습 없이, 후보 모델을 llama.cpp 경로에 올릴 수 있는지만 본다.
환경: Apple M4 Pro, 24GB, 로컬 `llama-cli` 10310 (`cb26014d9`).

## 변환 스크립트

llama.cpp 태그 `b10310`(설치된 바이너리와 같은 커밋 `cb26014d965036f0eacdab7387b222d28d36b9e6`)을 sparse clone으로
`gguf-py`, `requirements`, `conversion`만 받았다(`convert_hf_to_gguf.py` 포함). 리포 밖
`/Users/sjkim1127/fission-plugins-corpus/llamacpp-b10310`에 두었고 리포에는 넣지 않는다.
변환기는 `-I` 격리 모드로 실행하면 같은 폴더의 `conversion` 패키지를 못 찾는다(`-I` 없이 실행해야 함).
Python은 이 리포의 `.venv`(torch 2.14, transformers 5.21)를 썼다.

```sh
python convert_hf_to_gguf.py <HF snapshot dir> --outtype f16 --outfile <out>.gguf
llama-quantize <f16>.gguf <out>-Q8_0.gguf Q8_0      # Q4_K_M도 같은 방식
```

## Qwen3.5-2B-Base (하이브리드 Gated DeltaNet + Attention): 통과

| 단계 | 결과 |
|---|---|
| HF → GGUF f16 | 성공. 335 텐서, 3.9GB. 비전 인코더를 제외했는지는 확인하지 않았다(로그에 mmproj/vision 언급 7줄) |
| 양자화 | Q8_0 2.08GB(5초), Q4_K_M 1.31GB(11초) |
| 의사코드 perplexity (Fission 의사코드 150개, 145KB, 문맥 1024, 파인튜닝 전 베이스 모델) | f16 2.8103 ± 0.035, Q8_0 2.8095, Q4_K_M 2.8382 (f16 대비 +1.0%) |
| `llama-completion` 생성, Q4_K_M, 의사코드 369토큰 프롬프트 | Metal: 프롬프트 1050 tok/s, 생성 115 tok/s. CPU만(`-ngl 0`): 232 tok/s, 73 tok/s |
| `llama-server` (Q4_K_M, 문맥 4096) | `/health` ok, `/completion` 응답 정상, 131 tok/s |

해석:
- 이 아키텍처도 변환, 4비트 양자화, 서버 생성이 모두 된다. 계획에서 가장 큰 미검증 위험(3절)은 이 경로에서 해소됐다.
- 양자화 손실은 베이스 모델 perplexity로 보면 Q8_0은 사실상 없고 Q4_K_M은 약 1%다. 오차 범위(±0.035)에 가깝고
  쌍체 비교는 하지 않았다. **이름 정확도 기준의 손실은 파인튜닝한 모델로 따로 재야 한다**(M2a).
- 속도는 이 Mac의 값이다. 사용자 기기의 CPU 전용 성능은 측정하지 않았다.
- 베이스 모델은 "readable version" 프롬프트에서 이름을 바꾸지 않고 원문 선언을 되풀이했다. 이름 복원은 파인튜닝의 몫이다.

산출물(리포 밖): `/Users/sjkim1127/fission-plugins-corpus/g1/` 의 `*.gguf`, 로그, `ppl-pseudocode.txt`.

## 아직 안 한 후보

Qwen2.5-Coder-1.5B(safetensors 3.09GB, Apache-2.0), Qwen3.5-0.8B-Base(1.75GB, Apache-2.0). 라이선스와 크기는 HF 모델 정보로 확인했고
가중치는 내려받지 않았다.
