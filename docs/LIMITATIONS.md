# 해석의 한계와 후속 검증

1. **공개 test의 반복 사용.** TESTpub은 개발에 사용됐다. 독립 private/holdout 일반화와 구분한다.
2. **작은 효과와 지표 차이.** DINO 평균 상승은 작고 일부 범주는 하락한다. PatchCore의 주 지표는 4승4패이며 다른 지표에서는 CP가 낮다.
3. **역사적 native 조건.** DINO의 기존 5방법은 새 공통 환경 재추론이 아니다. KC100에서 같은 FIT 이웃의 native 거리 차이가 관측돼 선정 정책만의 완전한 인과 효과를 주장하지 않는다.
4. **예산 차이.** SCPP+SA 저거리 추가는 100k를 넘는다. 동일 예산의 개선으로 표현하지 않는다.
5. **GT와 초기값.** 오라클은 SCPP 초기화와 같은 GT의 학습·선정·평가를 사용한다. 99.95% 공유는 독립 정답 bank와의 일치나 전역 최적성의 증거가 아니다.
6. **Projection의 제약.** 전체 FIT NN 검색은 exhaustive지만 충돌 해소는 전역 최소비용 assignment가 아니다. 치환 후 재학습하지 않았다.
7. **유사도와 검출 성능.** 내부 CDF·평균 이동량·ID 공유·상호 NN은 서로 다르며 국소 query의 순위나 component별 검출을 대신하지 않는다.
8. **반복과 통계.** 주 표는 seed42 한 번이다. 새 paired scene-bootstrap CI나 유의성 검정을 제공하지 않는다. 672개 기하 비교를 독립 실험 반복으로 세지 않는다.
9. **정확성 범위.** SCPP는 부분집합과 batch 삭제를 사용한다. 전역 매 단계 exact closest-pair 삭제가 아니다. 같은 FP32 공식도 runtime·chunk·kernel이 바뀌면 bitwise 같지 않을 수 있다.
10. **공개 범위.** 선택·점수 함수와 집계표를 제공한다. Cache·GT·가중치·전체 extractor가 없어 이 저장소만으로 전체 benchmark의 bitwise 재생성을 보장하지 않는다.

후속 검증에는 같은 runtime에서 모든 bank 재추론, 여러 seed와 예산, 독립 holdout/private 평가, SCPP 외 초기화의 oracle, query별 rank 변화와 작은 component 분석이 필요하다. 이번 공개 작업에서는 이런 새 실험을 실행하지 않았다.

외부 참고: SuperADD paper-reported 8범주 pAUROC 83.93%, fixed F1 62.61%; AU-PRO는 미제공이다. 로컬과 조건이 달라 직접 성능 차이를 계산하지 않는다.
