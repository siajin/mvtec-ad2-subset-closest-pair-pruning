# 관련 연구와 연구의 위치

## 대표 특징 bank를 사용하는 산업 이상 검출

[PatchCore, Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)는 정상 patch 특징 bank를 greedy coreset으로 축약한다. 원 논문의 MVTec AD 성능을 현재 MVTec AD 2 결과로 옮겨 쓰지 않는다. [공식 구현](https://github.com/amazon-science/patchcore-inspection)의 approximate greedy 방식은 별도 WRN 통제 실험의 대조군이다.

[SuperADD, Roming et al., 2026](https://arxiv.org/abs/2605.14808)는 DINOv3 기반 다층 최근접 특징 비교와 memory-bank subsampling을 사용한다. 여기서는 그 선정 규칙에서 출발한 로컬 SuperADD-style과 CP100을 비교한다. 전체 방법의 공식 재현과 selector 연구를 구분한다. Paper-reported 8범주 pAUROC 83.93%, fixed F1 62.61%, can 51.61%/0.00%는 외부 참고다. AU-PRO는 논문에 제공되지 않았다.

[DINOv3, Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)는 이 연구가 재사용하는 동결 특징의 backbone이다. CP100은 새로운 backbone 학습법이 아니다. [MVTec AD 2, Heckler-Kram et al.](https://arxiv.org/abs/2503.21622)는 어려운 산업 검사와 촬영 조건 변화의 benchmark를 제공한다.

## 이번 연구의 위치

연구 단위는 대표 선정 규칙이다. 희소점 보존·coverage 확장·중복 삭제를 같은 후보 공간과 예산에서 살피고, 실제 대표 ID·1280D 위치·내부 간격을 검출 성능과 분리해 분석한다. CP100의 제안은 구체적인 escape-distance 삭제 규칙과 그 비교 근거다. 모든 closest-pair·dispersion·coreset 문헌에 대한 최초성을 입증한 것은 아니다.

GT prototype은 CP 초기화의 in-sample 진단이다. 이를 unsupervised 방법의 공정한 일반화 성능 대조군이나 모든 대표 선정의 상한으로 사용하지 않는다.

## 서지

- Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.* IJCV 2026. [DOI](https://doi.org/10.1007/s11263-026-02743-0), [arXiv](https://arxiv.org/abs/2503.21622).
- Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022, 14318–14328. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).
- Lukas Roming, Felix Lehnerer, Jonas V. Funk, Andreas Michel, Georg Maier, Thomas Längle, Jürgen Beyerer. *SuperADD: Training-free Class-agnostic Anomaly Segmentation — CVPR 2026 VAND 4.0 Workshop Challenge Industrial Track.* [arXiv](https://arxiv.org/abs/2605.14808).
- Oriane Siméoni et al. *DINOv3.* 2025. [arXiv](https://arxiv.org/abs/2508.10104).

SuperADD 평가 코드의 참조 commit은 `44cf25144442fbbc1334ea59d1632327a4376d1a`다. 이는 문헌 식별자이며 이 release가 공식 bank를 재실행했다는 뜻은 아니다.
