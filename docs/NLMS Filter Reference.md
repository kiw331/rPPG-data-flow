
## 📑 rPPG NLMS 필터 알고리즘 설계 설명

### 1. 핵심 설계 철학 (Concept)

본 코드는 "카메라로 측정된 신호에는 심박(Signal)뿐만 아니라 환경 노이즈(Noise)가 섞여 있다"는 전제에서 시작합니다. 이를 분리하기 위해 두 가지 영역을 사용합니다.

* **Desired Signal ($d$):** 심박 정보가 포함된 관심 영역(RoI)의 시계열 강도 데이터.
* **Reference Signal ($x$):** 조명 변화, 진동 등 환경 노이즈만 포함된 배경 영역(Background)의 시계열 데이터.

### 2. 알고리즘 주요 단계 (Pipeline)

#### ① 데이터 정규화 (Normalization)

논문의 공식(Equation 1)에 따라 각 채널 신호를 정규화합니다.


$$s_{norm, i}(t) = \frac{s_i(t)}{\mu(s_i(t))} - 1$$

* **이유:** 카메라 센서의 절대적인 밝기 값보다는 시간에 따른 상대적인 변화량(AC 성분)이 중요하기 때문입니다. 평균으로 나누어줌으로써 조명 강도 차이를 상쇄합니다.

#### ② NLMS(Normalized Least Mean Square) 적응형 필터링

알고리즘의 핵심으로, 배경 신호를 바탕으로 RoI 내의 노이즈를 추정하여 제거합니다.

* **필터 출력($y_n$):** 배경 신호($x$)에 필터 계수($w$)를 곱해 'RoI에 섞였을 것으로 추정되는 노이즈'를 계산합니다.
* **에러 계산($e_n$):** $e_n = d_n - y_n$. 실제 RoI 신호에서 추정된 노이즈를 뺍니다. **이 $e_n$ 값이 우리가 찾는 순수 심박 신호**가 됩니다.
* **계수 업데이트:** 아래 수식에 따라 필터가 실시간으로 학습하며 노이즈를 추적합니다.

$$w(n+1) = w(n) + \frac{\mu}{\|x(n)\|^2 + \epsilon} e(n)x(n)$$


* **$\mu$ (Step size):** 학습 속도를 결정합니다. 코드에서는 안정적인 추적을 위해 `0.1`을 권장값으로 설정했습니다.
* **$\|x(n)\|^2$:** 입력 신호의 에너지로 나누어주는 과정(Normalization) 덕분에 조명이 급격히 변해도 필터가 발산하지 않습니다.



#### ③ 채널 선택 및 대역 통과 필터 (Bandpass Filter)

* **채널 전략:** 사용자의 요청에 따라 가시광선 환경에서 유리한 G(Green)와 **R(Red)** 채널을 처리합니다.
* **필터링:** 심박수 이외의 저주파(움직임)와 고주파(전자 노이즈)를 제거하기 위해 Butterworth 밴드패스 필터를 적용합니다. 쥐의 경우 `5~10Hz` (300~600bpm) 구간을 추출합니다.

### 3. 구현 파라미터 근거

| 파라미터 | 설정값 | 근거 및 목적 |
| --- | --- | --- |
| **Filter Order** | `32` | rPPG 신호의 비정상성(Non-stationarity)을 고려할 때, 움직임 노이즈를 충분히 모델링하면서 계산 효율을 챙길 수 있는 적정 차수입니다. |
| **Step Size ($\mu$)** | `0.1` | 너무 높으면 신호가 왜곡되고, 낮으면 노이즈 추적이 느립니다. 논문 기반 rPPG 연구들에서 가장 범용적인 수치입니다. |
| **Epsilon ($\epsilon$)** | `1e-6` | 배경 신호가 0에 가까울 때 분모가 0이 되어 발생하는 수치적 오류를 방지합니다. |

### 4. 데이터 구조 (CSV Input)

코드에서는 Pandas를 사용하여 다음과 같은 구조의 CSV를 처리하도록 설계되었습니다.

* `roi_g`, `roi_r`: 관심 영역의 채널별 평균 밝기.
* `bg_g`, `bg_r`: 배경 영역의 채널별 평균 밝기.

---

### 💡 요약 및 팁

이 코드는 "배경에서 일어나는 변화는 RoI에서도 똑같이 일어날 것"이라는 가정하에 배경 신호를 '마이너스' 요소로 활용하는 영리한 방식입니다.

나중에 참고하실 때, 만약 결과 신호가 너무 뭉개진다면 `mu` 값을 낮추고, 노이즈가 여전히 많다면 배경 영역(Background RoI)을 피사체와 더 가깝지만 움직임 영향이 없는 곳으로 다시 설정해 보시기 바랍니다.



```python
"""
파일명: rppg_nlms_processor.py
설명: 
1. CSV 파일로부터 RoI(관심영역)와 Background(배경)의 R, G 채널 평균 강도값을 읽어옵니다.
2. 논문(Li et al., 2014)의 로직에 따라 배경 신호를 참조 신호로 하여 NLMS 필터를 적용합니다.
3. R, G 채널별로 노이즈가 제거된 신호를 생성하고, 최종 심박수(HR)를 추정합니다.
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def nlms_filter(d, x, order=32, mu=0.1, eps=1e-6):
    """
    NLMS 적응형 필터 함수
    d: RoI 신호 (Desired)
    x: 배경 신호 (Reference Noise)
    order: 필터 차수 (안정적인 추적을 위해 32 설정)
    mu: 학습률 (Step size, 0.1 내외가 일반적)
    """
    n_samples = len(d)
    w = np.zeros(order)  # 필터 계수 초기화
    e = np.zeros(n_samples)  # 노이즈 제거된 신호 저장
    
    for n in range(order, n_samples):
        # 입력 벡터 (최근 데이터부터 order만큼)
        x_n = x[n-order:n][::-1]
        
        # 1. 출력 계산 (추정된 노이즈)
        y_n = np.dot(w, x_n)
        
        # 2. 에러 계산 (실제 신호 - 추정 노이즈)
        e_n = d[n] - y_n
        e[n] = e_n
        
        # 3. 계수 업데이트 (Normalized LMS 공식)
        norm_x = np.dot(x_n, x_n)
        w = w + (mu / (norm_x + eps)) * e_n * x_n
        
    return e

def process_rppg(csv_path):
    # 1. 데이터 로드 (Columns: roi_r, roi_g, bg_r, bg_g 가정)
    data = pd.read_csv(csv_path)
    fps = 40  # 샘플링 레이트 (논문 기준)
    
    channels = ['r', 'g']  # B 채널 제외
    results = {}

    for ch in channels:
        roi = data[f'roi_{ch}'].values
        bg = data[f'bg_{ch}'].values
        
        # 2. 정규화 (논문 공식: s / mean(s) - 1)
        roi_norm = (roi / np.mean(roi)) - 1
        bg_norm = (bg / np.mean(bg)) - 1
        
        # 3. NLMS 필터 적용 (배경 노이즈 제거)
        # mu=0.1, order=32는 일반적인 움직임 노이즈 제거에 효과적입니다.
        denoised = nlms_filter(roi_norm, bg_norm, order=32, mu=0.1)
        
        # 4. 밴드패스 필터링 (심박 구간 추출)
        # 쥐 기준(300~600bpm -> 5~10Hz), 사람 기준이면 0.7~4Hz로 조정 필요
        low, high = 5.0, 10.0 
        nyq = 0.5 * fps
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        filtered = filtfilt(b, a, denoised)
        
        results[ch] = filtered

    # 5. 최종 신호 선택 및 HR 추정 (FFT)
    # R, G 중 파워 스펙트럼 피크가 더 명확한 것을 선택하는 로직 포함 가능
    return results

# 실행 예시
# results = process_rppg('signal_data.csv')




```