"""
PPG 신호 처리 방법 비교 테스트
- 구간(시작초, duration)을 지정해서 5가지 방법 비교
"""

import platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.ndimage import uniform_filter1d

# 한글 폰트
if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
else:
    matplotlib.rc("font", family="NanumGothic")
matplotlib.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# ★ 설정 (여기만 수정)
# ─────────────────────────────────────────────
DATA_PATH  = r"data\mouse_an_v1\20260507_162235\ppg_sensor.csv"

VIEW_START    = 0.55    # 시작 시간 (초)
VIEW_DURATION = 0.8    # 구간 길이 (초)

BPF_LO    = 5.0        # BPF 하한 (Hz)
BPF_HI    = 15.0       # BPF 상한 (Hz)
BPF_ORDER = 4

PEAK_MAX_BPM    = 800  # 최대 BPM (피크 최소 간격 계산용)
PEAK_PROMINENCE = 0.3  # prominence = std * 이 값


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

timestamps = df["Timestamp"].values
signal_raw = df["IR_Value_Raw"].values.astype(float)
dt = np.median(np.diff(timestamps))
fs = round(1.0 / dt)
t  = timestamps - timestamps[0]

print(f"샘플링 레이트: {fs} Hz | 전체 길이: {t[-1]:.1f}s")

# 구간 추출
t_end   = VIEW_START + VIEW_DURATION
win     = (t >= VIEW_START) & (t <= t_end)
t_seg   = t[win]
sig_seg = signal_raw[win]

print(f"\n분석 구간: {VIEW_START}s ~ {t_end:.2f}s  ({len(sig_seg)} 샘플)")
print(f"Raw 범위 : {sig_seg.min():.0f} ~ {sig_seg.max():.0f}  "
      f"(range={sig_seg.max()-sig_seg.min():.0f})")


# ─────────────────────────────────────────────
# 필터 함수
# ─────────────────────────────────────────────
def apply_bpf(sig, lo=BPF_LO, hi=BPF_HI, order=BPF_ORDER):
    nyq  = fs / 2
    hi   = min(hi, nyq * 0.98)
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def apply_hpf(sig, cutoff, order=BPF_ORDER):
    nyq  = fs / 2
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, sig)


# ─────────────────────────────────────────────
# 5가지 방법 정의
# ─────────────────────────────────────────────
x_norm = np.linspace(-1, 1, len(sig_seg))   # poly 피팅용 정규화 x

# 방법 1: 선형(1차) detrend + BPF
s1   = detrend(sig_seg, type='linear')
out1 = apply_bpf(s1)

# 방법 2: 2차 다항식 detrend + BPF
c2   = np.polyfit(x_norm, sig_seg, 2)
s2   = sig_seg - np.polyval(c2, x_norm)
out2 = apply_bpf(s2)

# 방법 3: 3차 다항식 detrend + BPF  (호흡 곡률까지 제거)
c3   = np.polyfit(x_norm, sig_seg, 3)
s3   = sig_seg - np.polyval(c3, x_norm)
out3 = apply_bpf(s3)

# 방법 4: HPF(2Hz) + BPF
s4   = apply_hpf(sig_seg, 2.0)
out4 = apply_bpf(s4)

# 방법 5: 이동평균 제거 + BPF
#   이동평균 창: 심박 주기(60/BPM)보다 크게 → 0.2s (300BPM 기준)
ma_win = max(3, int(0.2 * fs))
s5     = sig_seg.astype(float) - uniform_filter1d(sig_seg.astype(float), size=ma_win)
out5   = apply_bpf(s5)

methods = [
    ("1. Linear detrend + BPF",   out1, s1, "#1f77b4"),
    ("2. Poly2 detrend + BPF",    out2, s2, "#2ca02c"),
    ("3. Poly3 detrend + BPF",    out3, s3, "#ff7f0e"),
    ("4. HPF(2Hz) + BPF",         out4, s4, "#9467bd"),
    ("5. Moving avg(0.2s) + BPF", out5, s5, "#d62728"),
]


# ─────────────────────────────────────────────
# 피크 탐지 & 결과 출력
# ─────────────────────────────────────────────
min_d = max(2, int(60 / PEAK_MAX_BPM * fs))

print(f"\n{'방법':<35} {'peaks':>6} {'HR(BPM)':>9} {'std':>8} {'amp':>8}")
print("-" * 70)

results = []
for name, sig_bpf, sig_dt, color in methods:
    p, _ = find_peaks( sig_bpf, distance=min_d,
                        prominence=sig_bpf.std() * PEAK_PROMINENCE)
    v, _ = find_peaks(-sig_bpf, distance=min_d,
                        prominence=sig_bpf.std() * PEAK_PROMINENCE)
    dur = t_seg[-1] - t_seg[0]
    hr  = len(p) / dur * 60 if len(p) > 1 else 0
    amp = (np.median(sig_bpf[p]) - np.median(sig_bpf[v])
           if len(p) > 0 and len(v) > 0 else 0)
    print(f"{name:<35} {len(p):>6} {hr:>9.0f} {sig_bpf.std():>8.2f} {amp:>8.2f}")
    results.append((name, sig_bpf, sig_dt, color, p, v, hr))


# ─────────────────────────────────────────────
# 시각화 (흰색 배경)
# ─────────────────────────────────────────────
n_methods = len(methods)
fig, axes = plt.subplots(n_methods + 2, 1,
                          figsize=(14, 3 * (n_methods + 2)),
                          sharex=True,
                          facecolor='white')
fig.suptitle(
    f"PPG 방법 비교  [{VIEW_START:.2f}s ~ {t_end:.2f}s]  "
    f"BPF {BPF_LO}~{BPF_HI}Hz",
    fontsize=12, fontweight="bold", color='black'
)

# ── ① Raw
axes[0].plot(t_seg, sig_seg, color='black', lw=1.0, label='Raw')
axes[0].set_title(f"Raw  range={sig_seg.max()-sig_seg.min():.0f} ADC", color='black')
axes[0].set_ylabel("ADC")
axes[0].legend(fontsize=8)
axes[0].set_facecolor('white')

# ── ② Detrend 비교 (BPF 전)
ax_dt = axes[1]
for name, sig_bpf, sig_dt, color, p, v, hr in results:
    ax_dt.plot(t_seg, sig_dt, lw=0.9, color=color, alpha=0.8,
               label=name.split('+')[0].strip())
ax_dt.axhline(0, color='gray', lw=0.8, ls='--')
ax_dt.set_title("Detrend 결과 비교 (BPF 적용 전)", color='black')
ax_dt.set_ylabel("Detrended")
ax_dt.legend(fontsize=7, ncol=2)
ax_dt.set_facecolor('white')

# ── ③~ 각 방법 BPF 결과
for i, (name, sig_bpf, sig_dt, color, p, v, hr) in enumerate(results):
    ax = axes[i + 2]
    ax.plot(t_seg, sig_bpf, color=color, lw=1.0)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    if len(p):
        ax.scatter(t_seg[p], sig_bpf[p],
                   color='red',   s=60, zorder=5,
                   label=f'peak ({len(p)})', marker='^')
    if len(v):
        ax.scatter(t_seg[v], sig_bpf[v],
                   color='blue',  s=50, zorder=5,
                   label=f'valley ({len(v)})', marker='v')
    ax.set_title(f"{name}   →   {len(p)} peaks  /  {hr:.0f} BPM", color='black')
    ax.set_ylabel("BPF")
    ax.legend(fontsize=8)
    ax.set_facecolor('white')

for ax in axes:
    ax.grid(True, color='#cccccc', lw=0.5)
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['top'].set_color('#aaaaaa')
    ax.spines['right'].set_color('#aaaaaa')
    ax.yaxis.label.set_color('black')

axes[-1].set_xlabel("Time (s)")

plt.tight_layout()
# plt.savefig("ppg_window_compare.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()