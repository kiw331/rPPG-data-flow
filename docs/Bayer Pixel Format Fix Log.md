# Basler BayerRG12 디베이어링 버그 수정 기록

**날짜:** 2026-04-27 ~ 2026-04-28  
**관련 파일:** `modules/camera.py`, `rPPG-analysis-tool.py`, `raw_analysis.py`

---

## 1. 발견된 문제

rPPG 분석 툴 및 카메라 프리뷰에서 흰 피험체(쥐)가 **파란색/보라색**으로 표시되는 증상.  
원인: Basler 카메라의 `BayerRG12` 포맷 데이터에 잘못된 OpenCV Bayer 변환 코드 사용.

---

## 2. 핵심 원인: OpenCV Bayer 코드 대칭성

OpenCV에서 Bayer 패턴 이름(RG↔BG)과 출력 채널 순서(RGB↔BGR)를 **동시에 뒤집으면** 내부 연산 코드(번호)가 동일해지는 수학적 대칭성이 존재한다.

```python
# 아래 두 쌍은 각각 같은 내부 연산 번호를 가짐
cv2.COLOR_BayerBG2RGB == cv2.COLOR_BayerRG2BGR  # 둘 다 코드 48
cv2.COLOR_BayerRG2RGB == cv2.COLOR_BayerBG2BGR  # 둘 다 코드 46
```

이 카메라(Basler BayerRG12)에서 **올바른 내부 연산은 코드 48**이며,  
디스플레이 API의 채널 해석 방식에 따라 이름만 다르게 선택해야 한다.

---

## 3. 디스플레이 API별 올바른 변환 코드

| 표시 방법 | 채널 해석 | 사용할 코드 | 내부 번호 |
|---|---|---|---|
| `QImage.Format_RGB888` → `QLabel` | RGB 순서 기대 | `COLOR_BayerBG2RGB` | 48 ✅ |
| `pg.ImageView.setImage()` (pyqtgraph) | BGR 순서 기대 | `COLOR_BayerRG2BGR` | 48 ✅ |
| ~~`COLOR_BayerRG2RGB`~~ (잘못된 예) | RGB 기대이나 코드 46 | 코드 46 | ❌ 파란색 |

> 세 파일 모두 **동일한 내부 연산(코드 48)**을 사용하며, 이름만 API에 따라 다르다.

---

## 4. 파일별 수정 내역

### `modules/camera.py` — 카메라 실시간 프리뷰
```python
# 수정 전 (버그)
view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerBG2RGB)  # 우연히 맞았으나 이유 불명확

# 수정 후 (버그 없음, 이유 명확화 + 주석 추가)
view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerBG2RGB)  # 코드 48, QImage RGB888용
```
- 코드 자체는 원래 코드와 동일하게 복원
- `BayerRG12` 외 다른 Bayer 포맷(BayerRG8 등)에 대한 분기도 정리

---

### `rPPG-analysis-tool.py` — TIFF 기반 분석 툴

```python
# 수정 전 (버그 2종)
# HistPopup.show_frame (L1138): BayerBG2BGR (코드 46) → 색 뒤집힘
# MainWindow.show_frame (L629): BayerRG2RGB (코드 46) → 파란색

# 수정 후
img_display = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)   # 코드 48, pyqtgraph용
self.current_img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)  # 히스토그램 계산용 별도 보관
```
- 화면 표시용(BGR)과 히스토그램 계산용(RGB)을 변수 분리

---

### `raw_analysis.py` — RAW 파일 전용 분석 툴 (신규)

```python
# 처음부터 올바르게 작성
img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)  # 코드 48, pyqtgraph용
```

---

## 5. 저장 파이프라인은 영향 없음

```
GetArray() → img_queue.put(raw_data) → tifffile.imwrite()
```
디베이어링 변환 없이 **uint16 원본 Bayer 배열을 그대로 저장**하므로,  
위 수정 사항은 화면 표시에만 영향을 주며 저장된 파일 품질과는 무관하다.

---

## 6. 주의사항

- `tifffile.imwrite()`로 저장된 BayerRG12 TIFF는 `uint16` 2D 배열 (1채널 Bayer).  
  불러올 때 `len(img.shape) == 2`로 체크하여 Bayer 변환 적용.
- BGR8 포맷으로 저장된 TIFF는 `uint8` 3채널 배열. 변환 없이 그대로 사용.
- OffsetX=208, OffsetY=28 (짝수 오프셋)은 Bayer 위상에 영향을 주지 않음.
