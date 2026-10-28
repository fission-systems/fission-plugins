# fission-plugins 설계안 (초안 v0)

상태: **초안, 미확정.** 리포 이름·공개 범위·조직은 사용자 확인 전이며 GitHub에는 아무것도 만들지 않았다.
이 문서의 "확인함"은 2026-10-29에 로컬 파일을 읽거나 실제로 실행해 본 것, "가정"은 그렇지 않은 것이다.

## 1. 목적과 범위

Fission(`fission-systems/Fission`)이 내놓는 의사코드를 **후처리**하는 플러그인 모음.
첫 플러그인 `neural-rename`: 사전학습 소형 코드 모델(파인튜닝 + 양자화, 로컬 실행)로
변수·함수 이름을 복원하고 주석을 제안한다.

`fission-forge`의 scratch 모델 연구와는 별개다. 그쪽 결과(합성 문법에서만 부분 일반화, 사전학습 0.5B 파인튜닝이
같은 데이터에서 79/205 대 3/205)는 "작은 사전학습 모델 + 파인튜닝" 선택의 근거로만 쓴다.
단, 그 실험은 합성 한/두 연산 문법이므로 **실제 디컴파일러 출력에서의 이름 복원 성능은 아직 아무 근거도 없다.**

## 2. Fission의 확장 지점 (조사 결과)

| 지점 | 확인한 사실 | 후처리 용도 |
|---|---|---|
| `fission-plugin` 크레이트 (`FissionPlugin` 트레이트) | 훅은 `on_load`/`on_unload`/`on_binary_loaded`/`on_function_decompiled(&self, ctx, addr, code: &str)`. **반환값이 없는 관찰 전용**이라 출력을 바꿀 수 없다. 동적 라이브러리는 `fission_plugin_create` 심볼을 `libloading`으로 로드하며 소스 주석이 직접 "Rust 트레이트 객체 기반, 버전 일치가 필요한 ABI이고 안정 C ABI가 아니다"라고 적는다. `interactive_runtime` 피처 뒤에 있고 `PluginType`은 `Lua`/`Native`만 있다(Lua 로더는 못 찾음). | 직접 플러그인 불가. 향후 변경 제안 대상 |
| `fission-script` (Rhai) | 읽기 전용 바이너리 인벤토리용 샌드박스. 디컴파일 결과(NIR/HIR) 접근은 README에 "Deferred"로 명시. | 부적합 |
| **`fission_cli decomp --json`** | 직접 실행해 확인(0.2.3). 함수별 객체의 JSON 배열. 키: `address`, `name`, `size`, `code`, `code_hir`, `code_nir`, `variables`, `line_mappings*`, `engine_used`, `fell_back`. 디컴파일 실패 행은 `error`만 있고 `code`가 없다. 로그(`[fission] Loaded config ...`)가 stdout 앞에 섞여 JSON 파싱 전에 `[`부터 잘라야 한다. | **1차 통합 지점** |

`variables` 항목은 `name`, `kind`(`arg`/`reg`/`stack`), `type`, `arg_index`, `stack_offset`, `type_shape`를 가진다.
`line_mappings.defined_variables`는 줄 ↔ 변수 정의를 연결한다. 이름 복원의 입력/검증 신호로 쓸 수 있다.

실제 출력의 특징(arm64 Mach-O, `checksum` 함수): 매개변수는 `param_1`, 레지스터 유래 지역은 `x0`, `w8`, `xVar15`,
`iVar20`처럼 나온다. 즉 복원 대상은 "`param_N`, 레지스터명, `xVarN`/`iVarN`" 패턴이다.
(참고: 이 샘플의 루프 본문은 원본 C와 의미가 어긋나 보이는데, 이는 Fission 쪽 문제로 이 작업의 범위가 아니다.
후처리기는 의미를 고치지 않으므로 이런 오류를 그대로 전달한다.)

**결론: 후처리는 Fission 프로세스 밖에서 `decomp --json`을 소비하는 방식(out-of-process)으로 시작한다.**
Fission 쪽에 "출력을 변경할 수 있는 훅" 또는 안정 ABI가 생기면 얇은 네이티브 shim을 추가한다(M4).

## 3. 핵심 설계 원칙: 모델은 "코드"가 아니라 "편집 목록"만 낸다

플러그인은 코드를 다시 쓰지 않고 `EditList`(변수 rename, 함수 이름, 줄 주석, 타입 힌트)만 반환한다. 적용기(`apply_edits`)가
다음을 강제한다.

- 식별자 토큰 단위로만 치환 (문자열/문자 리터럴, 주석, `.x`/`->x` 멤버는 건드리지 않음)
- 새 이름이 C 식별자이고, 키워드/내장 타입이 아니며, 기존 식별자·다른 새 이름과 충돌하지 않을 때만 적용
- Fission이 보고한 변수(`variables`)만 rename 가능
- 타입 힌트는 v0에서 **적용하지 않고 보고만** 한다 (타입 변경은 의미를 바꿀 수 있음)

이렇게 하면 "의미 보존"이 모델 품질이 아니라 구조로 보장된다(rename의 알파 동치는 테스트로 확인함: `tests/test_apply.py`).
구조화 출력은 디코딩 비용도 작고 GBNF 문법으로 JSON 형식 강제가 가능하다. 대가는 모델이 줄 수 있는 표현력의 제한이다
(예: 루프를 `for`로 재구성하는 정리는 v0 범위 밖).

## 4. 리포 구조

```
fission-plugins/
  README.md  LICENSE  NOTICE(예정)  pyproject.toml
  docs/DESIGN.md                  # 이 문서
  schemas/edit-list.schema.json   # 플러그인 출력 계약
  src/fission_plugins/            # 공용 코어: schema, apply, plugin(Protocol), backends/(예정)
  plugins/<id>/                   # 플러그인 1개 = 디렉터리 1개
    plugin.toml                   #   id, version, 모델 repo+revision+sha256, 라이선스, 지원 Fission 버전
    README.md  eval/              #   플러그인별 평가 결과
  training/                       # 데이터 생성·파인튜닝·양자화 스크립트 (fission-forge/scripts/finetune에서 이식)
  eval/                           # 공용 평가 하네스와 벤치 설정
  tests/  tests/fixtures/         # Fission 실제 출력 일부를 고정한 픽스처
  shims/                          # (M4) Fission 네이티브 플러그인 shim
```

모델 가중치는 git에 넣지 않는다. HF 계정 `sjkim1127`(또는 조직)에 두고 `plugin.toml`에 repo, revision, sha256을 고정한다.
Fission 출력 형식이 바뀔 수 있으므로 어댑터는 `FunctionRecord.from_fission_json` 한 곳에만 둔다.

## 5. 플러그인 인터페이스 (초안)

`src/fission_plugins/plugin.py`: `PostProcessor.propose(FunctionRecord) -> EditList`, `Backend.complete(prompt, max_tokens, grammar)`.
호스트(`fission-postproc`, M1에서 구현)가 `decomp --json` 출력을 읽어 함수별로 호출하고 편집을 적용해 JSON/C 파일로 내보낸다.
플러그인은 바이너리를 보지 못하고 의사코드와 변수 표만 본다. 언어는 Python으로 시작(학습 스택이 Python)하되 계약이
JSON 데이터이므로 이후 Rust로 바꿔도 플러그인 호환이 유지된다.

## 6. 모델 로딩·양자화 경로 비교

| 후보 | 장점 | 단점 | 판단 |
|---|---|---|---|
| **GGUF + llama.cpp** (`llama-server`/`llama-cpp-python`) | macOS/Linux/Windows, CPU·Metal·CUDA 모두 지원. Q4_K_M/Q8_0. **GBNF 문법 제약 디코딩**으로 편집 목록 JSON 강제. Fission 사용자(Windows 바이너리 분석)가 Windows일 가능성이 높음 | 양자화 손실을 직접 측정해야 함 | **기본 경로** |
| MLX (`mlx_lm`) | Apple Silicon에서 빠르고 변환·4bit 간단 | Apple 전용, 문법 제약 디코딩이 약함 | 개발용 선택 백엔드 |
| ONNX Runtime | 폭넓은 배포 | 변환·디코딩 루프를 직접 관리, 소형 LLM 생태계 약함 | 보류 |
| Candle (Rust) | Fission과 같은 언어, 의존성 단순화 가능 | GGUF 지원은 있으나 속도·양자화 종류·문법 제약에서 뒤처짐 | 보류 (M4 이후 재평가) |

로컬 확인: 이 Mac(arm64)에 `llama-cli` b10310과 `llama-quantize`, `mlx-lm` 0.30.0이 이미 설치돼 있다. 아직 모델을 변환·실행해 보지는 않았다.

기본 모델 후보는 `Qwen/Qwen2.5-Coder-0.5B`(forge 실험에서 이미 사용, Apache-2.0)와 1.5B. 같은 시리즈의 3B는 다른 라이선스
(Qwen Research)로 알고 있으므로 **채택 시점에 모델 카드로 재확인**한다. 이 문서 작성 시점 이후 더 나은 소형 코드 모델이 있을 수 있으므로
후보는 고정하지 않는다.

학습은 forge의 `scripts/finetune/`(Colab `colab` CLI, 전체 파인튜닝, 완성부 손실만, NLL로 체크포인트 선택, 예약 test 미사용)의 프로토콜을
그대로 이식하되, 입력을 "P-code 레인 프롬프트"에서 "Fission 의사코드 + 변수 표"로, 출력을 "C 소스"에서 "편집 목록 JSON"으로 바꾼다.
재사용 가능한 것: `finetune_job.py`(학습 루프, 상태 파일, 시드·리비전 고정), `run_finetune.py`(Colab 세션 수명주기와 예산 제한),
`prepare_data.py`의 매니페스트·sha256 방식. 재사용 불가: 프롬프트 렌더링(`forge pcode-compile`), 합성 코퍼스.

## 7. 데이터 (가장 큰 위험)

이름 복원에는 `(Fission 의사코드, 원본 변수명)` 쌍이 필요하다. 후보 구성: 오픈소스 C 프로젝트를 `-g -O0..O2`로 빌드 → DWARF에서
원본 변수명 추출 → Fission 출력의 `variables`/`line_mappings`와 주소로 정렬. **이 정렬의 정확도와 커버리지는 아직 모른다.** 레지스터 유래
변수(`x0`, `w8`)는 원본 변수와 1:1이 아닐 수 있다. 따라서 M0에서 정렬률부터 측정하고, 낮으면 계획을 바꾼다.
분할은 프로젝트 단위로 하고 DecBench와의 중복을 점검한다(forge `docs/research/llm4decompile.md`의 방침 재사용). 학습 데이터 라이선스는
프로젝트별로 기록한다.

## 8. 평가

1. **이름 정확도**: 정확 일치, 정규화 일치(대소문자·스네이크/카멜), 서브토큰 F1. 기준선은 (a) 무변경 (b) 타입 기반 규칙 이름(`buf`, `len`, `i`)
   (c) 파인튜닝 전 같은 모델 zero-shot.
2. **의미 보존**: 적용 결과를 역 rename했을 때 원본과 바이트 동일(속성 테스트) + 가능한 경우 컴파일 확인. Fission의 `fission-dir` 실행 차분은
   rename에서는 불변이어야 하므로 회귀 점검용.
3. **안전성**: 편집 목록 파싱 실패율, 적용기 거부율(환각된 변수, 충돌). 거부 사유별 집계.
4. **비용**: 양자화별(FP16/Q8_0/Q4_K_M) 이름 정확도 손실, 함수당 지연, 최대 RSS (M-시리즈와 CPU 전용).
5. 분할은 프로젝트 단위, 시드·모델 리비전·데이터 해시를 결과 옆에 기록. 단일 시드 결과는 그렇다고 명시한다.

## 9. 라이선스

- **코드: Apache-2.0 권장.** Fission, fission-forge, fission-benchmark 모두 Apache-2.0이고 Fission은 NOTICE 파일로 서드파티를 고지한다. 호환·일관성이 가장 좋다.
- 대안: MIT(더 단순하지만 특허 조항 없음), GPL/AGPL(Fission 생태계와 맞지 않음).
- **모델 가중치는 코드와 별도 라이선스**를 모델 카드에 적는다. 베이스 모델 라이선스(Qwen2.5-Coder 0.5B/1.5B는 Apache-2.0으로 알고 있음, 채택 시 재확인)와
  학습 데이터 출처 라이선스를 따르며, 데이터가 GPL 코드 유래라면 가중치 배포 조건을 별도 검토한다.
- `LICENSE`는 fission-forge의 Apache-2.0 전문을 그대로 복사해 둔 상태다(결정 확정 전 임시).

## 10. 마일스톤

- **M0 (측정)**: Fission 출력 ↔ DWARF 변수명 정렬률 측정, 규칙 기반 기준선 구현, 평가 하네스. 모델 없음.
- **M1 (호스트)**: `fission-postproc` CLI(`decomp --json` 입력 → 적용 결과 출력), 규칙 기반 플러그인으로 end-to-end.
- **M2 (모델)**: 소형 모델 파인튜닝(Colab), GGUF 변환, Q8/Q4 평가.
- **M3 (배포)**: 모델 카드, HF 업로드, 설치 문서.
- **M4 (선택)**: Fission 쪽에 변경 가능한 훅/안정 ABI를 제안하고 네이티브 shim 추가.

## 11. 열린 결정과 위험

- 리포 이름·공개 범위·조직 (사용자 확정 필요, 아래 질문 참조)
- 학습 데이터 출처와 정렬 정확도(7절)가 성패를 가른다
- Fission 출력 형식은 0.2.3 기준으로만 확인했고 버전마다 달라질 수 있다 → `plugin.toml`에 지원 버전 범위를 둔다
- 현재 로컬에서는 다른 워크트리(`Fission-wt-abA`)의 빌드 바이너리로 확인했다. 어느 커밋의 빌드인지는 확인하지 않았다
