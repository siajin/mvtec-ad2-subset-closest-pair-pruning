# 재현 안내와 공개 범위

## 공개본에서 할 수 있는 작업

- CSV/JSON에서 8범주 평균, 대표 ID 공유율, 교차 기하 통계를 검산한다.
- 고정된 순서의 정상 FP16 특징 pool에 SCPP를 적용한다.
- 작은 합성 데이터로 pruning, native 1NN, oracle gradient·고유 FIT 치환, 양방향 교차 NN을 확인한다.

원본 이미지에서 표의 모든 수치를 자동 재생성하는 완결 benchmark 배포본은 아니다. 이미지·GT·가중치·추출 cache·개별 source manifest·대형 bank·score map은 포함하지 않는다.

## 설치와 검산

Python 3.10 이상을 사용하고 repository root에서 실행한다.

```bash
python -m venv .venv
# OS에 맞게 가상환경을 활성화한다.
python -m pip install -e .
python tools/summarize_results.py
python tools/verify_release.py
python -m unittest discover -s tests -v
```

기본 의존성은 NumPy 2.1.3과 Torch 2.10.0이다. CUDA wheel은 해당 stage의 계약에 맞게 별도로 선택한다. 이 설치가 Linux CPU 2.9.1 reference 환경까지 만드는 것은 아니다. `evaluation` extra는 원본 metric·resize 함수에 필요한 SciPy/OpenCV를 추가한다.

## 준비한 특징에서 대표 선택

```python
import numpy as np
from scpp import select_representatives

fit = np.load('normal_fit.npy', mmap_mode='r', allow_pickle=False)
# [N,1280], frozen float16, original source/patch ordering
ids, trace = select_representatives(fit, device='cpu')
np.save('selected_ids.npy', ids)
np.save('bank.npy', np.asarray(fit[ids]))
```

CLI도 같은 함수를 사용한다.

```bash
python -m scpp select --features normal_fit.npy --output selected_ids.npy --device cpu
python -m scpp score --queries query.npy --bank bank.npy --output score.npy --device cpu
```

출력 JSON에 입력·출력 SHA, seed, chunk, 실제 Torch/NumPy, device를 기록하고 기존 출력을 덮어쓰지 않는다. `--target`·`--subsets`를 줄인 toy는 별도 설정이며 100k benchmark 결과로 표기하지 않는다. FP32 특징을 조용히 재양자화하지 않도록 선택 API는 FP16 pool만 받는다.

`score`는 한 층의 global-bank RMS 1NN이다. 네 층 평균, dense 보간, GT 평가, binary 후처리는 별도다. 전체 bank를 device에 올리고 query chunk × bank 크기의 거리 행렬을 계산한다. CPU 예시는 API 확인용이며 역사적 GPU 수치나 속도의 재현을 보장하지 않는다.

## 역사적 DINO 결과의 재현에 필요한 입력

1. 공식 데이터와 동결 DINOv3 checkpoint, 원래 preprocessing·밝기 draw·overlap 처리.
2. L7/15/23/31 raw1280D FP16 cache의 원래 바이트와 split·source·patch 순서.
3. 원래 selected ID, bank, native/dense control과 SHA.
4. 아래처럼 분리된 stage 환경과 동결 resize 경로.
5. 정상 control 통과 후 수행하는 별도 GT 평가.

| 단계 | 검증된 SCPP 계보 환경 | 계약 |
|---|---|---|
| native 거리 | Linux H100, Torch 2.10.0+cu128 | FP32/highest, TF32/autocast off, query4096/full-bank100k, image batch8 |
| dense reference | Linux CPU, Torch 2.9.1+cu128 build | 실제 연산은 CPU, frozen bilinear |
| local evaluator | Windows CPU, Torch 2.10.0+cpu | portable frozen Linux2.9.1 보간, FP16 ranking |
| bank geometry | FP64 | exhaustive NN, 선택된 쌍의 거리 직접 재계산 |

관련 stage의 NumPy는 2.1.3, OpenCV는 4.12.0이다. 저장 FP16, 계산 FP32, 평가 FP16은 다른 계약이다. GEMM·min·sqrt·나눗셈·평균 순서와 chunk·batch를 지킨다. 이 환경을 KC100이나 별도 PatchCore producer에 자동 적용하지 않는다.

subset-mean 후속 실험은 같은 native A control을 공유하지만 score reduction이 다르다. 그 registry의 group 평균을 여기 main SCPP의 추론 정의로 복사하지 않는다. 본 대표 선정 연구는 can 정렬을 사용하지 않았으며, 공개 작업에서도 alignment를 재계산하지 않았다.

## 소스와 결과 추적

[SOURCE_ARTIFACTS.json](../provenance/SOURCE_ARTIFACTS.json)은 원본의 source ID·SHA256·크기·용도를 기록한다. 명칭 변경 전 원본 SHA와 공개 파일 SHA를 별도로 보존한다. `public_derivative`는 명칭·범례를 통일한 공개본이다. 원본의 위치는 비공개 출처 대장에 남기고 공개본은 source ID로 연결한다. 결과의 수치와 알고리즘은 변경하지 않았다. 원본 실험이나 부분 완료 기록을 수정하지 않았다.

`research/scpp/`은 `MVTEC_GRID_ROOT` 아래 원래 cache 구조와 manifest를 전제하는 producer snapshot이다. 일반적인 배열 입력에는 `scpp/` adapter를 사용한다. `research/oracle/`은 핵심 함수 보존본이며 전체 학습 launcher는 아니다.

[RELEASE_SHA256.json](../provenance/RELEASE_SHA256.json)은 자체 파일을 제외한 게시 payload를 고정한다. 이번 발행은 새 학습·GT 평가·원격 할당 없이 기존 결과와 짧은 CPU 제어만 사용했다.
