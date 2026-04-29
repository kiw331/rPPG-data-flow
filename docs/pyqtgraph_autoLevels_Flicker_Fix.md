# pyqtgraph ImageView 플리커 이슈 수정 기록

**날짜:** 2026-04-28  
**관련 파일:** `rPPG-analysis-tool.py`, `raw_analysis.py`

---

## 1. 발견된 문제

`rPPG-analysis-tool.py` 및 `raw_analysis.py`에서 프레임을 재생할 때
**프레임마다 밝기가 출렁이는 플리커/리플 현상** 관찰.

같은 데이터를 AVI로 저장해서 미디어플레이어로 재생하면 플리커가 없음.

---

## 2. 원인: pyqtgraph `autoLevels` 기본값

```python
# pyqtgraph ImageView.setImage() 시그니처 (기본값)
setImage(img,
    autoRange  = True,   # 뷰포트 줌/패닝 자동 조절
    autoLevels = True,   # ← 이게 문제! 매 프레임 밝기 자동 정규화
    levels     = None,
    ...
)
```

기존 코드:
```python
self.image_view.setImage(img_pg, autoRange=False)
#  autoRange=False  → 뷰 고정 ✅
#  autoLevels=True  → 매 프레임 min/max 기반 정규화 → 플리커 ❌
```

`autoLevels=True`이면 pyqtgraph는 **매 프레임마다 픽셀의 min/max를 계산**하여
그 범위를 화면 밝기 0~255로 늘려서 표시한다.

프레임마다 실제 픽셀값 분포가 조금씩 달라지므로(노이즈, rPPG 신호 등)
→ 정규화 스케일이 바뀜 → 화면 밝기가 프레임마다 달라짐 → **플리커**

---

## 3. AVI에서 플리커가 없는 이유

```
TIFF(uint16) → >> 4 (고정 스케일) → MJPG 인코딩
```

`>>4` 는 고정 연산이므로 모든 프레임에서 동일한 스케일 적용.
밝기 기준이 일정 → 플리커 없음.

---

## 4. 수정 내용

### `autoLevels=False` + 고정 `levels` 적용

```python
# 수정 전
self.image_view.setImage(img_pg, autoRange=False)

# 수정 후
levels = (0, 255) if img_display.dtype == np.uint8 else (0, 4095)
self.image_view.setImage(img_pg, autoRange=False, autoLevels=False, levels=levels)
```

| dtype | levels | 이유 |
|---|---|---|
| `uint8` (BGR8) | `(0, 255)` | 8비트 전체 범위 |
| `uint16` (BayerRG12) | `(0, 4095)` | 12비트 전체 범위 |

### 수정된 위치

| 파일 | 라인 | 위치 |
|---|---|---|
| `rPPG-analysis-tool.py` | L638 | `HistPopup.show_frame()` |
| `rPPG-analysis-tool.py` | L1146 | `MainWindow.show_frame()` |
| `raw_analysis.py` | L691 | `HistPopup.show_frame()` |
| `raw_analysis.py` | L1225 | `MainWindow.show_frame()` |

---

## 5. 실제 rPPG 플리커와 인공물 구분

| 종류 | 원인 | 크기 | 수정 후 |
|---|---|---|---|
| **실제 rPPG 신호** | 혈류에 의한 피부 반사율 미세 변화 | ~0.1~1% | 여전히 존재 (정상) |
| **autoLevels 인공물** | pyqtgraph 자동 정규화 스케일 변동 | 수~수십% | 제거됨 ✅ |

> 수정 후 실제 rPPG 신호로 인한 극히 미세한 밝기 변화는 여전히 데이터에 존재하지만,
> 육안으로는 보이지 않을 수준이며 신호 분석은 픽셀 값으로 직접 수행하므로 무관.

---

## 6. 주의사항

- `levels=(0, 4095)` 고정이므로 노출 과다 이미지(max > 4095)는 클리핑될 수 있음
- 필요 시 첫 프레임 로드 후 `img.max()`로 동적으로 levels 설정 가능
- pyqtgraph 우측의 히스토그램 위젯(숨김 처리)으로 수동 레벨 조정도 가능
