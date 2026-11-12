# SCPP 방법 상세

## 문제 설정

정상 FIT 특징 $X=\{x_i\}_{i=1}^N$에서 예산 $K$의 실제 표본 부분집합을 고른다. 연구의 기본 DINO 설정은 raw1280D, 범주·층별 $K=100000$이다. 학습으로 backbone이나 특징 좌표를 바꾸지 않는다. global FIT ID는 고정된 source 순서로 특징을 펼친 행 번호다. 파일 이름이 같아도 source 순서가 다르면 ID를 직접 비교할 수 없다.

## SCPP의 실제 알고리즘

```text
rng = RandomState(42)
selected = []
repeat 100 times:
    eligible = all FIT IDs except IDs already in selected
    active = rng.choice(eligible, floor(N/100), replace=False)
    while len(active) > 1000:
        find the nearest TWO non-self neighbors for every active point
        make undirected edges from each point to its FIRST neighbor
        sort edges by squared distance, then endpoint global IDs
        freeze these neighbor lists for this round
        visit unique edges whose two endpoints are still alive:
            compute each endpoint's nearest alternative excluding its partner
            remove endpoint with smaller alternative distance
            if equal, remove endpoint with larger global FIT ID
            stop after min(len(active)-1000, floor(.20*len(active))) removals
        retain alive points and recompute neighbors in the next round
    append the 1000 survivors to selected
```

원본은 최소1개를 round 제거 한도로 두지만 기본1000개 목표에서는 20% 한도가 항상 충분히 크다. 최종 선택된 ID만 후속 후보에서 제외한다. 이전 round 또는 부분집합에서 삭제된 FIT 후보는 다시 뽑힐 수 있다. 최종 bank의 행 순서는 부분집합 순서와 각 부분집합의 생존 순서다.

최근접 두 개를 알고 있으면 쌍 $(i,j)$에서 상대를 제외한 escape distance를 구할 수 있다. $j$가 $i$의 첫 이웃이면 두 번째 거리, 그렇지 않으면 첫 번째 거리다. 원본 구현은 이 값을 squared-L2로 비교한다. 전체 정상 분포에 대한 density estimator를 별도로 fitting하지 않는다.

## 직관과 비교

가까운 쌍은 두 점이 비슷한 정상 정보를 담을 가능성을 보여준다. 한 점을 없앨 때, 쌍 밖에서도 쉽게 대체되는 끝점을 지우고 상대적으로 고립된 끝점을 남기려는 규칙이다. 이것이 anomaly localization을 직접 최적화하거나 최적 coverage를 보장하는 목적함수는 아니다.

| 정책 | 주요 판단 | 특징 |
|---|---|---|
| SuperADD-style | k-NN 기반 local density score가 작은 후보 우선 | 희소 영역을 보존하는 방향 |
| k-center / PatchCore coreset | 현재 대표 집합에서 먼 후보 추가 | 최악 coverage 거리 축소를 지향 |
| SCPP | 가까운 쌍에서 escape distance가 작은 끝점 제거 | 중복을 줄이는 삭제 방식 |

같은 거리 공간과 예산을 쓰더라도 세 정책의 최종 대표는 다를 수 있다. SCPP에서 cutoff·tie·부분집합·제거 순서가 바뀌면 다른 selector다.

## 거리와 연산 순서

DINO 선택 원본의 기본 query chunk는2048이다. 특징을 FP32로 읽고 다음 squared-L2를 계산한다.

```python
d2 = (q.square().sum(1, keepdim=True)
      + active.square().sum(1)[None, :]
      - 2.0 * (q @ active.T)).clamp_min(0)
```

이웃 검색 시 자기 행은 infinity로 마스킹한다. 다른 행의 동일 좌표는 합법적인 거리0 이웃이다. 두 이웃은 `torch.topk(..., k=2, largest=False, sorted=True)`로 구하므로 부동소수점 distance tie의 이웃 순서는 device/library까지 포함해 재현해야 한다. endpoint tie에서만 global ID 규칙이 명시적으로 적용된다.

실제 native scoring은 full100k bank, query chunk4096, image batch8, FP32/highest와 TF32/autocast off를 따른다. squared distance에서 **min → sqrt → /sqrt1280** 후 네 층 FP32 평균이다. 부분집합별 min을 평균하는 다른 실험의 score로 바꾸지 않는다.

## 계산량과 근사

한 round에서 $m$개 활성 특징, 차원 $D$, query chunk $c$일 때 pair distance 연산은 $O(m^2D)$, 주요 거리 행렬 메모리는 $O(cm)$이다. 각 부분집합의 매 round마다 exhaustive 검색하지만 전역 FIT 전체에서 매 삭제마다 최근접을 갱신하지 않는다. 부분집합 sampling, round 내 stale neighbor, 20% batch 삭제가 근사·경로 의존성을 만든다.

원본 producer는 부분집합 완료별 ID, RNG state와 진단을 저장하고 완료 marker에 SHA를 남긴다. 공개 portable adapter는 단일 `.npy` pool을 받아 같은 pruning 함수를 호출한다. 전체 역사 cache의 source manifest나 중간 체크포인트를 자동 복구하는 기능은 제공하지 않는다.

## 변형의 구분

- **SCPP + SA 저거리 추가:** SCPP를 유지하고 추가 FIT 표본을 더해100k를 초과한다. 동일 예산 실험으로 보지 않는다.
- **K-means 2단계:** KMeans100 → 군집 내부 CP로 총200k → 전역 SCPP 100k. centroid100k 최적화와 다르다.
- **subset-mean 후속 실험:** 같은 representative lineage라도 score의 reduction이 다르다. 본 공개 표는 global1NN이다.
- **PatchCore 이식:** 동일128D projection 공간에서 대표를 고르고1024D 원본 특징으로 squared-L2 score를 구한다. raw1280D DINO bank와 다르다.
- **GT 학습형/치환형:** 진단용이며 정상-only SCPP selector 자체와 구분한다.
