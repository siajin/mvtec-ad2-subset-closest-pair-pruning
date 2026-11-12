# 오라클과 대표 유사도

## 왜 오라클을 만들었는가

정상 표본만 고르는 selector와, 결함 GT를 보고 prototype 좌표를 움직이는 최적화는 다른 제약 문제다. 후자의 결과를 진단 도구로 사용해 SCPP 대표가 어디에 놓이는지 확인한다. 여기서 oracle은 편의상 붙인 이름이며 수학적 global optimum 또는 검출 성능의 certified upper bound가 아니다.

## 학습형과 실제 FIT 치환형

각 범주 네 층의 SCPP bank로 초기화한다. backbone은 고정하고 prototype만 FP32로 학습한다. 매 step의 결함 component 균등 표본64개, normal FIT64개, public good32개, bad 안의 normal background32개, hard background64개를 사용한다. normalized score를 $u$, defect 표본을 $A$, 정상 표본을 $N$이라 하면 구현된 loss는 다음과 같다.

$$L=0.1\,\operatorname{mean}_{n\in N}u_n^2+
\operatorname{mean}_{a\in A}\max(0,1.5-u_a)^2+
\operatorname{mean}_{a\in A,n\in N}\max(0,0.1+u_n-u_a).$$

native forward는 FP32 hard-min을 유지하고 선택된 prototype row에 sparse gradient를 전달한다. SparseAdam, clip norm1, seed42,2000step이다. step0/250/500/1000/1500/2000에서 저장하고 같은 GT의 실제 AU-PRO@0.05로 선택했다. 모든 범주에서2000step이 선택됐다. 설정은 [oracle_diagnostic.json](../configs/oracle_diagnostic.json), 핵심은 [learning_core.py](../research/oracle/learning_core.py).

학습형의 자유좌표를 전체 정상 FIT에 FP64 exhaustive 최근접으로 제안한다. 같은 FIT ID를 제안한 prototype끼리 직접 거리와 row 순서로 승자를 정하고, 나머지는 사용하지 않은 FIT에서 다시 제안한다. 고유100k ID를 얻고 재학습하지 않는다. 이는 exhaustive 후보 검색+결정적 충돌 해소이며, 총거리 최소의 전역 assignment 해법은 아니다.

## 결과를 읽는 순서

| 8범주 평균 (%) | SCPP | 학습형 | 실제 FIT 치환형 |
|---|---:|---:|---:|
| pixel AUROC | 93.1240 | 99.3093 | 93.3164 |
| pAUROC@0.05 | 84.2729 | 96.3487 | 84.3188 |
| AU-PRO@0.05 | 51.6859 | 97.7524 | 52.0260 |

SuperADD paper-reported 참고는 별도다:8범주 pAUROC83.93%, fixed F162.61%, AU-PRO 미제공. 이 표의 oracle과 직접 순위·차이를 계산하지 않는다.

1. **성능:** 자유좌표 학습의 in-sample 성능은 높지만 실제 FIT 치환 후 SCPP에 가까워진다.
2. **ID:** SCPP와 치환형의 FIT ID 공유 비율은32bank 평균99.95078125%다. 자유좌표에는 원본 FIT ID가 없어N/A다.
3. **움직임:** prototype 이동량/출발CP의 행별 내부NN간격 비율을 bank 안에서 median한 뒤32bank median은 약0.000190이다. 이동량0인 prototype 평균은35.1386%다. 이는 다른 bank와의 최적 matching 비용이 아니다.
4. **위치:** 학습형과 치환형은 SCPP와 각각99.98800%,99.98722%의 CP방향 상호NN 비율을 보인다. 높은 상호NN이 동일 좌표·동일 score를 뜻하지 않는다.

초기값이 SCPP이고 상당수 이동량이 작기 때문에 치환형이 CP로 돌아오기 쉽다. 따라서 이 결과를 selector와 무관하게 얻은 정답 bank와의99.95% 일치로 표현하지 않는다. 다른 초기값·최적화·제약에서의 결과는 검증하지 않았다.

## 유사도의 네 정의

실제 FIT ID 집합 $I_A,I_B$에 대해 containment는 $|I_A\cap I_B|/|I_A|$, Jaccard는 $|I_A\cap I_B|/|I_A\cup I_B|$다. 증강 bank는 크기가 다르므로 방향별 containment를 각각 보고한다.

교차 위치는 각 $a\in A$에서 $d_B(a)=\min_{b\in B}\|a-b\|/\sqrt{1280}$를 구한다. 공유ID를 포함한 전체 target bank를 검색한다. 비공유 조건부 통계는 source query에서 공유ID만 제외한다. 다대일 최근접이며 최적 일대일 matching이 아니다.

상호NN은 A의 대표가 B의 최근접 대표와 서로를 최근접으로 고르는 비율이다. 내부 간격 CDF는 각 bank 안의 self-excluded 최근접 거리 분포다. 강체 이동된 두 bank는 내부 CDF가 같아도 서로 다른 위치에 있을 수 있다.

## Hausdorff와 query score의 관계

$h(A,B)=\max_{a\in A}d_B(a)$라 하면 정확한 실수 거리에서

$$d_B(x)-h(A,B)\le d_A(x)\le d_B(x)+h(B,A).$$

대칭 Hausdorff $H=\max(h(A,B),h(B,A))$로 $|d_A(x)-d_B(x)|\le H$다. 두 query의 기존 score 차이가 $2H$보다 크면 순서를 보존한다. 더 작은 차이는 역전이 가능한 범위이며 실제 순위 역전의 증거는 아니다. 동일 가중 다층 평균에는 층별 bound의 평균을 적용할 수 있다. FP32 계산과 FP16 ranking의 반올림은 별도로 다룬다.

따라서 내부 histogram, 작은 평균 이동량, 높은 ID 중복만으로 AUROC/AU-PRO의 동일성을 결론낼 수 없다. 실제 query별 score·순위·component 변화가 추가로 필요하다.

## 공개한 증거

- [32bank × 15 실제-ID 쌍](../results/geometry/id_overlap_per_bank.csv), [ID 요약](../results/geometry/id_overlap_summary.csv)
- [32bank × 21 내부CDF 쌍](../results/geometry/within_bank_spacing_cdf_per_bank.csv)
- [672교차쌍](../results/geometry/cross_pairs_672.csv), [1344방향](../results/geometry/cross_directional_1344.csv)
- [prototype 이동·projection 통계](../results/oracle/movement_summary.json)
- [L7](../assets/cross_mean_L7.png), [L15](../assets/cross_mean_L15.png), [L23](../assets/cross_mean_L23.png), [L31](../assets/cross_mean_L31.png) 교차 거리 행렬
- [원본 교차NN 구현](../research/geometry/cross_core.py)

원래 독립 검증은 회수 배열 SHA, 배열 요약, 상호NN, 비공유 통계를 다시 확인했다. 모든 원격1280D exhaustive 검색을 로컬에서 다시 계산한 것은 아니다. 이번 public 발행도 새로운 GPU 검색을 하지 않았다.
