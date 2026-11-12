# 평가 계약과 비교의 범위

## DINO main comparison

1. 범주별 공개 bad 이미지의 모든 pixel을 합친다. bad 이미지 안의 정상 배경은 음성 pixel이다.
2. 원본 $H,W$의 `H//4,W//4` 격자를 사용한다. GT는 OpenCV `INTER_LINEAR`로 resize한 뒤 `>0`으로 만든다.
3. 네 층 FP32 native score를 평균하고 동결 bilinear 경로로 dense map을 만든다. 순위 평가 직전에 FP16으로 변환한다.
4. pAUROC는 `roc_auc_score(y, score, max_fpr=.05)`의 **표준화 partial AUC**다. 단순 적분값을0.05로 나눈 값과 다르다.
5. AU-PRO는 이미지별8-connected GT component의 recall을 component마다 같은 가중치로 계산한다. curve를 FPR0.05까지 적분하고0.05로 나눈다. FP16 tie는 동일 score group으로 처리한다.
6. 범주 내 pooled metric을 계산한 후8범주 arithmetic mean을 보고한다. 이미지별 평균이나 모든 범주 pixel을 한꺼번에 합친 값이 아니다.

본 main table은 rank-only다. fixed F1, threshold 또는 morphology를 사용하지 않았다. 따라서 AUROC/AU-PRO 차이를 고정 threshold나 binary morphology의 영향으로 설명하지 않는다. [원본 rank evaluator](../research/oracle/metrics_core.py), [동결 Linux2.9.1 portable 보간](../research/oracle/bilinear_frozen_linux_torch291.py).

## SuperADD 논문과 로컬 실험

SuperADD paper-reported8범주 pAUROC83.93%, fixed F162.61%; can은51.61%,0.00%다. [원문 Table1](https://arxiv.org/html/2605.14808v1).

| 항목 | paper-reported reference | 로컬 SuperADD-style |
|---|---|---|
| 출처 | 논문 표 | 보존된 로컬 대표·score 계보 |
| 공식 pretrained bank 재실행 | 이 저장소에서 수행하지 않음 | 공식 rerun 아님 |
| 의미 | 외부 보고값 참고 | 같은 특징 연구 내 selector 비교 |
| fixed F1 | 논문 값 제공 | 이 main table에 없음 |
| AU-PRO | 논문에서 미제공 | 로컬 별도 산출 |

원래 code commit과 은행·출력 grid·GT 변환·dtype·pooling·threshold·후처리가 같은지 확인하기 전에는 두 종류의 수치를 같은 조건의 superiority 비교로 읽지 않는다. 과거 stored pilot은 공개 표·그림·결론의 기준값으로 사용하지 않았다.

## 역사적 비교가 남기는 수치 한계

DINO의 기존5방법은 검증된 같은 범주의 역사 score에서 재집계했다. 두 오라클은 독립 CPU 평가를 했으며 SCPP anchor는8범주 모두 기존 pAUROC/AU-PRO와 exact 일치했다. 이것은 모든7방법을 같은 runtime에서 새로 추론한 실험이라는 뜻이 아니다.

특히 KC100의 frozen native 산출물은 같은 source·patch·최근접 FIT ID에서도 CP와 일부 값이 달랐다. can calibration L31에서 관측한 최대 차이는 raw L2/√1280 단위0.0095100403이다. Torch/CUDA kernel 하나를 원인으로 특정하지 않았고 새 허용 오차로 사용하지 않는다. 기존 SCPP runtime을 KC producer runtime으로 추정하지 않는다. KC100의 이 범위 AU-PRO 기록은 없어 N/A다.

## PatchCore 별도 계약

PatchCore 표는 public good379+bad705장 전체, FP32 score다. 분할은 train/good 전량 FIT2528장과 공식 정상 validation302장이며, DINO의 train 내부 held-out calibration과 다르다. Quarter GT/8-connected component 정의를 사용한다. Q99 F1은 정상 validation score의99th percentile로 고정한 good+bad pooled F1이다. Paper fixed F1과 직접 subtract하지 않는다. 상세는 [PatchCore](PATCHCORE.md).

## 오라클과 통계

공개 GT는 oracle 학습·hard-background 선정·checkpoint 선택·최종 평가에 함께 사용됐다. `LEARNED`, `PROJECTED`는 supervised in-sample diagnostic이다. 생성 코드와 최종 report evaluation을 분리했어도 이 사실은 바뀌지 않는다.

최종 공개 표는 한 seed의 point estimate다. 672쌍 geometry나 수백만 대표는 독립적인 defect test sample 수가 아니다. can의 동일 scene6조명은 상관된 관측이다. 새로운 신뢰구간을 이 공개 작업에서 계산하지 않았으며, 유의성·동등성·독립 generalization 주장을 하지 않는다.
