"""
ppg_gap_viewer.py  —  PPG 재구성 + 갭 비교 뷰어

실행: python ppg_gap_viewer.py

기능:
  - hr_segments.csv 기반 재구성 (Method 5 + 엔벨로프 정규화)
  - 갭 구간 네비게이션 (이전/다음)
  - 원신호(Raw) / 재구성 신호 2-패널 비교
  - 폴더 전환 (이전/다음 폴더)
  - 그래프 확대/축소: 마우스 우클릭 드래그, 휠
  - 품질 불량 구간 라벨링 → excluded_segments.csv 저장
    (라벨 모드 ON → 재구성 그래프에서 좌클릭 2번으로 구간 지정)
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QDoubleSpinBox,
    QSplitter,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# ── 파일명 ──────────────────────────────────────────────────────────
PPG_CSV   = "ppg_sensor.csv"
LABEL_CSV = "hr_segments.csv"
EXCL_CSV  = "excluded_segments.csv"   # 품질 불량 구간 라벨 저장 파일

# ── 처리 기본값 ─────────────────────────────────────────────────────
PAD_SEC    = 0.30
MA_WIN_SEC = 0.15
BPF_ORDER  = 4
PEAK_PROM  = 0.3


# ────────────────────────────────────────────────────────────────────
class PPGGapViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG 재구성 갭 뷰어")
        self.resize(1400, 900)

        self.current_folder = None
        self._t        = None
        self._sig_raw  = None
        self._labels   = None
        self._fs       = 200.0

        self._valid_segs = []
        self._gaps       = []
        self._sig_rp     = None   # 실신호 (NaN 마스킹)
        self._sig_sp     = None   # 합성신호 (NaN 마스킹)
        self._gap_idx    = 0

        self._raw_region_items = []
        self._rc_region_items  = []

        # ── 구간 라벨링 상태 ────────────────────────────────────────
        self._label_mode      = False   # 라벨 모드 ON/OFF
        self._label_start_t   = None    # 첫 번째 클릭 시각(초)
        self._label_start_line = None   # 시작점 InfiniteLine
        self._mark_labels     = []      # [{"start": float, "end": float}, ...]
        self._mark_items      = []      # plot_rc 의 LinearRegionItem 리스트

        self._init_ui()

    # ── UI ──────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 행1: 폴더 컨트롤
        row1 = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("ppg_sensor.csv + hr_segments.csv 폴더 경로...")
        self.path_edit.returnPressed.connect(
            lambda: self.load_folder(self.path_edit.text().strip()))
        btn_browse = QPushButton("📁 폴더 선택")
        btn_browse.clicked.connect(self._browse)
        self.btn_prev_folder = QPushButton("◀ 이전 폴더")
        self.btn_prev_folder.clicked.connect(self._prev_folder)
        self.btn_next_folder = QPushButton("다음 폴더 ▶")
        self.btn_next_folder.clicked.connect(self._next_folder)
        row1.addWidget(self.path_edit, stretch=1)
        row1.addWidget(btn_browse)
        row1.addWidget(self.btn_prev_folder)
        row1.addWidget(self.btn_next_folder)
        root.addLayout(row1)

        # 행2: 파라미터 + 갭 네비게이션
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("BPF (Hz):"))
        self.spin_lo = QDoubleSpinBox()
        self.spin_lo.setRange(0.1, 20.0); self.spin_lo.setSingleStep(0.5)
        self.spin_lo.setValue(3.0)
        row2.addWidget(self.spin_lo)
        row2.addWidget(QLabel("~"))
        self.spin_hi = QDoubleSpinBox()
        self.spin_hi.setRange(1.0, 50.0); self.spin_hi.setSingleStep(0.5)
        self.spin_hi.setValue(12.0)
        row2.addWidget(self.spin_hi)
        btn_proc = QPushButton("⚙️ 재처리")
        btn_proc.clicked.connect(self._process_and_draw)
        row2.addWidget(btn_proc)
        row2.addSpacing(30)

        self.btn_prev_gap = QPushButton("◀ 이전 갭")
        self.btn_prev_gap.clicked.connect(self._prev_gap)
        self.lbl_gap = QLabel("갭  -/-")
        self.lbl_gap.setAlignment(Qt.AlignCenter)
        self.lbl_gap.setMinimumWidth(100)
        self.btn_next_gap = QPushButton("다음 갭 ▶")
        self.btn_next_gap.clicked.connect(self._next_gap)
        btn_full = QPushButton("🔭 전체 보기")
        btn_full.clicked.connect(self._full_view)

        row2.addWidget(self.btn_prev_gap)
        row2.addWidget(self.lbl_gap)
        row2.addWidget(self.btn_next_gap)
        row2.addSpacing(20)
        row2.addWidget(btn_full)
        row2.addStretch()
        self.lbl_status = QLabel("폴더를 선택하세요")
        self.lbl_status.setStyleSheet("color:#555;")
        row2.addWidget(self.lbl_status)
        root.addLayout(row2)

        # 행3: 구간 라벨링 컨트롤
        row3 = QHBoxLayout()
        self.btn_label_mode = QPushButton("🔴 라벨 모드")
        self.btn_label_mode.setCheckable(True)
        self.btn_label_mode.setToolTip(
            "ON: 재구성 그래프에서 좌클릭 두 번으로 불량 구간 지정\n"
            "ESC: 시작점 취소")
        self.btn_label_mode.toggled.connect(self._toggle_label_mode)
        row3.addWidget(self.btn_label_mode)

        btn_undo_lbl = QPushButton("↩ 마지막 삭제")
        btn_undo_lbl.setToolTip("가장 마지막에 추가한 라벨 1개 삭제")
        btn_undo_lbl.clicked.connect(self._undo_label)
        row3.addWidget(btn_undo_lbl)

        btn_clear_lbl = QPushButton("🗑 전체 삭제")
        btn_clear_lbl.clicked.connect(self._clear_labels)
        row3.addWidget(btn_clear_lbl)

        btn_save_lbl = QPushButton("💾 라벨 저장")
        btn_save_lbl.setToolTip(f"현재 폴더/{EXCL_CSV} 에 저장")
        btn_save_lbl.clicked.connect(self._save_labels)
        row3.addWidget(btn_save_lbl)

        row3.addSpacing(20)
        self.lbl_mark_status = QLabel("라벨: 0개")
        self.lbl_mark_status.setStyleSheet("color:#333; font-weight:bold;")
        row3.addWidget(self.lbl_mark_status)
        row3.addStretch()
        root.addLayout(row3)

        # 갭 상세 정보
        self.lbl_gap_info = QLabel("")
        self.lbl_gap_info.setStyleSheet("color:#1565C0; font-weight:bold; padding:2px;")
        root.addWidget(self.lbl_gap_info)

        # 그래프
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # Raw plot
        self.plot_raw = pg.PlotWidget(
            title="<b>원신호 (Raw ADC)</b>  |  "
                  "■ 초록=피크 탐지 성공 구간 (실신호 사용)  "
                  "■ 파랑=hr_segments 라벨 원본  "
                  "■ 빨강=갭 (합성 보간 대상)")
        self.plot_raw.setBackground('w')
        self.plot_raw.showGrid(x=True, y=True, alpha=0.25)
        self.plot_raw.setLabel('left', 'ADC')
        self.plot_raw.setLabel('bottom', 'Time (s)')
        # y축 자동 스케일: 보이는 X 구간에 맞춰 y 범위 조정 → DC 오프셋 문제 해결
        self.plot_raw.getViewBox().setAutoVisible(y=True)
        self.plot_raw.enableAutoRange(axis='y', enable=True)
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('#333333', width=0.8))
        splitter.addWidget(self.plot_raw)

        # Reconstruction plot
        self.plot_rc = pg.PlotWidget(
            title="<b>재구성 신호 (정규화 [-1, +1])</b>  |  "
                  "─ 파랑=실신호 기반 정규화  "
                  "-- 주황=갭 구간 코사인 합성 보간")
        self.plot_rc.setBackground('w')
        self.plot_rc.showGrid(x=True, y=True, alpha=0.25)
        self.plot_rc.setLabel('left', '정규화 진폭')
        self.plot_rc.setLabel('bottom', 'Time (s)')
        self.curve_real  = self.plot_rc.plot(
            pen=pg.mkPen('#1f77b4', width=1.3), name='실신호')
        self.curve_synth = self.plot_rc.plot(
            pen=pg.mkPen('#ff7f0e', width=1.3, style=Qt.DashLine), name='합성')
        self.plot_rc.setYRange(-1.6, 1.6)
        splitter.addWidget(self.plot_rc)
        splitter.setSizes([480, 380])

        # X축 링크 (두 그래프가 같이 스크롤/줌)
        self.plot_rc.setXLink(self.plot_raw)

        # 라벨링 클릭 이벤트 연결 (plot_rc 에서만 동작)
        self.plot_rc.scene().sigMouseClicked.connect(self._on_rc_click)

    # ── 폴더 탐색 ───────────────────────────────────────────────────
    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path:
            self.load_folder(path)

    def _siblings(self):
        if not self.current_folder:
            return []
        parent = os.path.dirname(self.current_folder)
        try:
            return sorted([
                os.path.join(parent, d)
                for d in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, d))
                   and os.path.exists(os.path.join(parent, d, PPG_CSV))
            ])
        except Exception:
            return []

    def _prev_folder(self):
        sib = self._siblings()
        if not sib:
            return
        cur = os.path.normcase(os.path.normpath(self.current_folder))
        idx = next((i for i, s in enumerate(sib)
                    if os.path.normcase(os.path.normpath(s)) == cur), -1)
        if idx > 0:
            self.load_folder(sib[idx - 1])

    def _next_folder(self):
        sib = self._siblings()
        if not sib:
            return
        cur = os.path.normcase(os.path.normpath(self.current_folder))
        idx = next((i for i, s in enumerate(sib)
                    if os.path.normcase(os.path.normpath(s)) == cur), -1)
        if 0 <= idx < len(sib) - 1:
            self.load_folder(sib[idx + 1])

    def load_folder(self, path):
        path = os.path.normpath(path)
        ppg_path   = os.path.join(path, PPG_CSV)
        label_path = os.path.join(path, LABEL_CSV)

        if not os.path.exists(ppg_path):
            self.lbl_status.setText(f"❌ {PPG_CSV} 없음: {path}")
            return

        self.current_folder = path
        self.path_edit.setText(path)
        self.setWindowTitle(f"PPG 재구성 갭 뷰어 — {os.path.basename(path)}")

        df            = pd.read_csv(ppg_path).dropna()
        self._sig_raw = df["IR_Value_Raw"].values.astype(float)
        t_raw         = df["Timestamp"].values
        self._t       = t_raw - t_raw[0]
        self._fs      = 1.0 / np.median(np.diff(self._t))

        if os.path.exists(label_path):
            self._labels = pd.read_csv(label_path)
        else:
            self._labels = pd.DataFrame(
                columns=["segment_id", "start_time", "end_time", "length"])
            self.lbl_status.setText("⚠️ hr_segments.csv 없음 — Raw만 표시")

        # 라벨 초기화 후 기존 excluded_segments.csv 로드
        self._clear_labels()
        excl_path = os.path.join(path, EXCL_CSV)
        if os.path.exists(excl_path):
            try:
                excl_df = pd.read_csv(excl_path)
                for _, row in excl_df.iterrows():
                    self._add_label(float(row["start_time"]), float(row["end_time"]))
                self.lbl_mark_status.setText(
                    f"라벨: {len(self._mark_labels)}개  (기존 로드: {EXCL_CSV})")
            except Exception as ex:
                self.lbl_mark_status.setText(f"라벨 로드 실패: {ex}")
        else:
            self.lbl_mark_status.setText("라벨: 0개")

        self._process_and_draw()

    # ── 처리 ────────────────────────────────────────────────────────
    def _process_and_draw(self):
        if self._t is None:
            return

        bpf_lo  = self.spin_lo.value()
        bpf_hi  = self.spin_hi.value()
        fs      = self._fs
        t       = self._t
        sig_raw = self._sig_raw

        # ── 내부 함수 ──
        def bpf(sig):
            nyq  = fs / 2
            hi   = min(bpf_hi, nyq * 0.98)
            b, a = butter(BPF_ORDER, [bpf_lo / nyq, hi / nyq], btype='band')
            return filtfilt(b, a, sig)

        def proc_iso(s, e):
            m_inner = (t >= s) & (t <= e)
            t_inner = t[m_inner]
            s_inner = sig_raw[m_inner].copy()
            if len(s_inner) < 3:
                return t_inner, s_inner

            pad_len = int(PAD_SEC * fs)
            # 신호의 급격한 경계면(갭과의 경계)에서 발생하는 필터 과도 현상(ringing)을 방지하기 위해 'edge' 패딩 적용
            se_padded = np.pad(s_inner, pad_width=pad_len, mode='edge')
            
            ma_n = max(3, int(MA_WIN_SEC * fs))
            sdt_padded = se_padded - uniform_filter1d(se_padded, ma_n)
            sbf_padded = bpf(sdt_padded)
            
            sbf_inner = sbf_padded[pad_len:-pad_len]
            return t_inner, sbf_inner

        def detect(ts, ss, seg_start, seg_end):
            prom      = ss.std() * PEAK_PROM
            prom_edge = ss.std() * 0.12
            mind_boot = max(2, int(0.05 * fs))   # 부트스트랩용 느슨한 최소 간격(50ms)

            # ── Step 1: 부트스트랩 탐지 → 평균 주기 추정 ──
            pk0, _ = find_peaks(ss, distance=mind_boot, prominence=prom)
            if len(pk0) < 2:
                vl0, _ = find_peaks(-ss, distance=mind_boot, prominence=prom)
                return pk0, vl0, False, False, []
            avg_T = np.median(np.diff(ts[pk0]))

            # ── Step 2: 본 탐지 — 최소 간격 0.6×avg_T (한 파형 이중 탐지 방지) ──
            mind = max(mind_boot, round(0.6 * avg_T * fs))
            pk, _ = find_peaks( ss, distance=mind, prominence=prom)
            vl, _ = find_peaks(-ss, distance=mind, prominence=prom)
            if len(pk) < 2:
                return pk, vl, False, False, []

            avg_T = np.median(np.diff(ts[pk]))   # 재계산

            # ── Step 2.5: 내부 누락 피크/밸리 보완 ──
            # 인접 피크 간격이 1.4×avg_T 초과 → 그 사이에 피크가 누락됐을 가능성
            # find_peaks 완화 탐지 후 실패 시 argmax 강제 사용
            def _fill_interior(arr, invert=False):
                """arr: 피크 또는 밸리 인덱스 배열. invert=True이면 밸리(음수 신호)."""
                nonlocal avg_T
                changed = True
                while changed:
                    changed = False
                    if len(arr) < 2:
                        break
                    ivls = np.diff(ts[arr])
                    for i in np.where(ivls > 1.4 * avg_T)[0]:
                        t_mid = (ts[arr[i]] + ts[arr[i + 1]]) / 2
                        win   = 0.35 * avg_T
                        m_w   = (ts >= t_mid - win) & (ts <= t_mid + win)
                        if m_w.sum() < 3:
                            continue
                        sig_w = -ss[m_w] if invert else ss[m_w]
                        ep2, _ = find_peaks(sig_w, distance=mind_boot,
                                            prominence=prom * 0.15)
                        if len(ep2):
                            cands = np.where(m_w)[0][ep2]
                            new_i = cands[np.argmax(sig_w[ep2])]
                        else:
                            new_i = np.where(m_w)[0][np.argmax(sig_w)]
                        arr = np.sort(np.unique(np.append(arr, new_i)))
                        avg_T = np.median(np.diff(ts[arr]))
                        changed = True
                        break   # 재계산 후 while 재시작
                return arr

            # 내부 채우기 전 넓은 간격 기록 → env_norm 후 해당 구간 gen_synth 교체
            interior_syn = []
            if len(pk) >= 2:
                for i in np.where(np.diff(ts[pk]) > 1.4 * avg_T)[0]:
                    interior_syn.append((ts[pk[i]], ts[pk[i + 1]]))

            pk = _fill_interior(pk, invert=False)
            vl = _fill_interior(vl, invert=True)

            avg_T  = np.median(np.diff(ts[pk]))
            ref_pk = np.median(ss[pk])
            edge_r = avg_T         # 양끝단 유효 탐색 반경

            def _edge_search(t_lo, t_hi):
                """경계 범위 안에서 양수이고 진폭 일관성을 만족하는 최대 피크 1개 반환."""
                m = (ts >= t_lo) & (ts <= t_hi)
                if m.sum() <= 3:
                    return np.array([], dtype=int)
                ep, _ = find_peaks(ss[m], distance=mind_boot, prominence=prom_edge)
                if not len(ep):
                    return np.array([], dtype=int)
                abs_idx = np.where(m)[0][ep]
                # 양수 피크만 허용 — 음수 로컬맥스는 실제 심박 피크가 아님
                ok = (ss[abs_idx] > 0) & (ss[abs_idx] <= ref_pk * 1.8)
                if not ok.any():
                    return np.array([], dtype=int)
                abs_idx = abs_idx[ok]
                return np.array([abs_idx[np.argmax(ss[abs_idx])]])

            def _period_forced(ref_t, direction):
                """탐지 실패 시 인접 5개 파형 주기 중앙값으로 피크 위치 강제 결정.
                direction: +1=ref_t 이후(tail), -1=ref_t 이전(head)"""
                ivl  = np.diff(ts[pk])
                N    = min(5, len(ivl))
                m_T  = np.median(ivl[-N:] if direction == 1 else ivl[:N])
                t_tg = ref_t + direction * m_T
                return np.array([np.argmin(np.abs(ts - t_tg))])

            forced_head = forced_tail = False

            # ── Step 3: 첫 피크 위치 검증 (실피크 우선 탐색) ──
            # hr_segments.csv 는 골(valley)를 라벨링 → 이상적 첫 피크는 seg_start + 0.5*avg_T
            # 실피크가 [seg_start, seg_start + avg_T]에 있으면 추가
            if ts[pk[0]] > seg_start + edge_r:
                new = _edge_search(seg_start, seg_start + edge_r)
                if len(new):
                    pk = np.sort(np.unique(np.append(pk, new)))

            # ── Step 4: 끝 피크 위치 검증 (실피크 우선 탐색) ──
            if ts[pk[-1]] < seg_end - edge_r:
                new = _edge_search(seg_end - edge_r, seg_end)
                if len(new):
                    pk = np.sort(np.unique(np.append(pk, new)))
            # ── Step 5: 피크-골 교대성(Alternation) 검증 및 단축 ──
            # 피크와 골이 번갈아 나타나지 않는 왜곡이 발생하면, 위반 시점 이후를 잘라내고 보간으로 대체합니다.
            events = []
            for p in pk:
                events.append((ts[p], 1, p))   # 피크는 1
            for v in vl:
                events.append((ts[v], -1, v))  # 골은 -1
            events.sort(key=lambda x: x[0])

            valid_events = []
            last_type = None
            truncated = False

            for i, ev in enumerate(events):
                t_ev, type_ev, idx_ev = ev
                if last_type is not None and type_ev == last_type:
                    # 연속해서 같은 타입(피크-피크 또는 골-골)이 나타나면 왜곡이 시작된 것이므로 단축
                    truncated = True
                    break
                valid_events.append(ev)
                last_type = type_ev

            if truncated:
                pk = np.array([ev[2] for ev in valid_events if ev[1] == 1], dtype=int)
                vl = np.array([ev[2] for ev in valid_events if ev[1] == -1], dtype=int)

            # ── Step 6: 경계 강제 확장 ──
            # hr_segments.csv 는 골 라벨이므로 첫/마지막 피크는 양끝에서 ~0.5T 위치에 있어야 함.
            # 실 탐지된 피크가 양끝에서 1T 초과 떨어져 있으면 avg_T 간격으로 강제 피크 추가.
            # 추가된 구간은 gen_synth로 교체되어 자연스러운 코사인 파형 유지.
            if len(pk) >= 2:
                avg_T = np.median(np.diff(ts[pk]))

                # HEAD 확장
                first_real_t = ts[pk[0]]
                added_head   = False
                while ts[pk[0]] - seg_start > 1.0 * avg_T:
                    t_new = ts[pk[0]] - avg_T
                    if t_new < seg_start + 0.3 * avg_T:
                        t_new = max(seg_start + 0.5 * avg_T, ts[0])
                    idx = int(np.argmin(np.abs(ts - t_new)))
                    if idx >= pk[0] or idx <= 0:
                        break
                    pk = np.sort(np.unique(np.append(pk, idx)))
                    added_head = True
                if added_head:
                    forced_head = True
                    interior_syn.append((ts[pk[0]], first_real_t))

                # TAIL 확장
                last_real_t = ts[pk[-1]]
                added_tail  = False
                while seg_end - ts[pk[-1]] > 1.0 * avg_T:
                    t_new = ts[pk[-1]] + avg_T
                    if t_new > seg_end - 0.3 * avg_T:
                        t_new = min(seg_end - 0.5 * avg_T, ts[-1])
                    idx = int(np.argmin(np.abs(ts - t_new)))
                    if idx <= pk[-1] or idx >= len(ts) - 1:
                        break
                    pk = np.sort(np.unique(np.append(pk, idx)))
                    added_tail = True
                if added_tail:
                    forced_tail = True
                    interior_syn.append((last_real_t, ts[pk[-1]]))

            return pk, vl, forced_head, forced_tail, interior_syn

        def env_norm(tr, sr, tpk, spk, tvl, svl):
            # 상·하한 엔벨로프를 '선형보간'으로 계산.
            #  CubicSpline 은 진폭이 급변(정상 피크→artifact 피크)할 때
            #  제어점 사이에서 오버슈트(아래로 처짐)하여 실신호가 엔벨로프를
            #  벗어남(>+1) → 스파이크/평탄화 발생.
            #  선형보간은 두 피크를 직선으로 잇기에 그 사이 신호(밸리로 하강)는
            #  항상 직선 아래에 머무름이 수학적으로 보장됨 → 출력 [-1,+1] 보장.
            upper = np.interp(tr, tpk, spk)   # 피크 선형보간
            lower = np.interp(tr, tvl, svl)   # 밸리 선형보간 (양끝 자동 클램프)
            denom = np.where(upper - lower < 1e-9, 1e-9, upper - lower)
            return np.clip(2.0 * (sr - lower) / denom - 1.0, -1.0, 1.0)

        def gen_synth(tg, T_start, T_end=None, start_type='peak', end_type='peak', valley_ref=-1.0):
            # T_start: period at gap start (matches preceding real segment boundary)
            # T_end:   period at gap end   (matches following real segment boundary)
            # phase varies smoothly so d_phi/dt = 2pi/T(t) with T(t) linearly varying
            if T_end is None:
                T_end = T_start
            dur = tg[-1] - tg[0]
            if dur <= 0 or len(tg) < 2:
                val = 1.0 if start_type == 'peak' else valley_ref
                return np.full(len(tg), val)

            # Harmonic-mean period → integer cycle count
            T_hmean = 2 * T_start * T_end / (T_start + T_end)
            
            # 목표 총 위상 변화량 delta_phi 결정
            if start_type == end_type:
                n_cyc = max(1, round(dur / T_hmean))
                delta_phi = 2 * np.pi * n_cyc
            else:
                n_half = max(1, round(2 * (dur / T_hmean) - 1)) # 홀수 배의 반주기 개수
                if n_half % 2 == 0:
                    n_half += 1
                delta_phi = np.pi * n_half

            # Angular frequencies at each end
            w0    = 2 * np.pi / T_start
            w1    = 2 * np.pi / T_end
            alpha = (tg - tg[0]) / dur          # 0 → 1

            # Integrate linearly-varying omega: phi = dur*(w0*a + (w1-w0)*a²/2)
            phi_raw   = dur * (w0 * alpha + (w1 - w0) * alpha ** 2 / 2)
            phi_total = phi_raw[-1]              # = dur*(w0+w1)/2

            # Rescale so phi spans exactly delta_phi
            scale = delta_phi / phi_total if phi_total > 0 else 1.0
            phi   = phi_raw * scale

            if start_type == 'peak':
                y = np.cos(phi)
            else:
                y = -np.cos(phi)

            mid = (1.0 + valley_ref) / 2
            amp = (1.0 - valley_ref) / 2
            return mid + amp * y

        # ── 세그먼트 처리 ──
        segs = []
        for _, row in self._labels.iterrows():
            s, e   = row["start_time"], row["end_time"]
            ts, ss = proc_iso(s, e)
            if len(ts) < 10:
                segs.append(None); continue

            pk, vl, forced_head, forced_tail, interior_syn = detect(ts, ss, ts[0], ts[-1])
            if len(pk) < 2 or len(vl) < 1:
                segs.append(None); continue

            tlo, thi = ts[pk[0]], ts[pk[-1]]
            mpp = (ts >= tlo) & (ts <= thi)
            if mpp.sum() < 5:
                segs.append(None); continue

            tr, sr = ts[mpp], ss[mpp]
            pkr = pk[(ts[pk] >= tlo) & (ts[pk] <= thi)]
            vlr = vl[(ts[vl] >= tlo) & (ts[vl] <= thi)]
            if len(pkr) < 2 or len(vlr) < 1:
                segs.append(None); continue

            try:
                sn = env_norm(tr, sr, ts[pkr], ss[pkr], ts[vlr], ss[vlr])
            except Exception:
                segs.append(None); continue

            # ── 아티팩트 구간 gen_synth 교체 ──
            # 공통 헬퍼: pkr 로컬 주기 중앙값으로 구간 [t0,t1]을 코사인으로 덮어씀
            def _replace_syn(t0, t1):
                m_s = (tr >= t0) & (tr <= t1)
                if m_s.sum() < 2:
                    return
                ivl = np.diff(ts[pkr])
                m_T = float(np.median(ivl)) if len(ivl) else (t1 - t0)
                sn[m_s] = gen_synth(tr[m_s], m_T)

            # 내부 넓은 간격: fill_interior 삽입 피크가 아티팩트 → env_norm 왜곡 → 교체
            for t0, t1 in interior_syn:
                _replace_syn(t0, t1)

            # 인접 피크 진폭 비율 > 1.8 → 상단 엔벨로프 경사 왜곡 → gen_synth 교체
            if len(pkr) >= 2:
                pk_amp = ss[pkr]
                for i in range(len(pkr) - 1):
                    lo = min(abs(pk_amp[i]), abs(pk_amp[i + 1]))
                    hi = max(abs(pk_amp[i]), abs(pk_amp[i + 1]))
                    if lo > 0 and hi / lo > 1.8:
                        _replace_syn(ts[pkr[i]], ts[pkr[i + 1]])

            # 사이클별 코사인 적합도(RMSE) 검사 → 노이즈로 왜곡된 사이클 교체
            # 이상 사이클: 시작/끝=+1, 중간=-1 인 cos(2πα) 형태.
            # RMSE > 0.25 이면 노이즈/왜곡이 심하므로 코사인으로 대체.
            for i in range(len(pkr) - 1):
                t0, t1 = ts[pkr[i]], ts[pkr[i + 1]]
                mc = (tr >= t0) & (tr <= t1)
                if mc.sum() < 4:
                    continue
                cycle = sn[mc]
                tc    = tr[mc]
                alpha = (tc - tc[0]) / (tc[-1] - tc[0] + 1e-12)
                ideal = np.cos(2 * np.pi * alpha)
                rmse  = float(np.sqrt(np.mean((cycle - ideal) ** 2)))
                if rmse > 0.25:
                    _replace_syn(t0, t1)

            pk_ivl      = np.diff(ts[pkr])
            period_mid  = float(np.median(pk_ivl)) if len(pk_ivl) else 0.15
            N_loc       = min(5, len(pk_ivl))
            raw_head    = float(np.median(pk_ivl[:N_loc])) if len(pk_ivl) else period_mid
            raw_tail    = float(np.median(pk_ivl[-N_loc:])) if len(pk_ivl) else period_mid
            
            # 국소 노이즈로 인해 주기가 늘어나거나 줄어드는 왜곡을 막기 위해 대표 주기(중앙값)의 85%~115% 범위로 클램핑
            period_head = np.clip(raw_head, 0.85 * period_mid, 1.15 * period_mid)
            period_tail = np.clip(raw_tail, 0.85 * period_mid, 1.15 * period_mid)

            segs.append({
                "t":           tr,
                "sig":         sn,
                "period":      float(np.median(pk_ivl)),
                "period_head": period_head,   # local period near segment start
                "period_tail": period_tail,   # local period near segment end
                "t_s":         tlo,
                "t_e":         thi,
                "label_start": s,
                "label_end":   e,
            })

        valid = [s for s in segs if s is not None]

        # ── 재구성 배열 ──
        n      = len(t)
        sig_rc = np.full(n, np.nan)
        filled = np.zeros(n, bool)
        is_syn = np.zeros(n, bool)

        # 1. 세그먼트 내부 영역 채우기 (실신호 및 피크 감지 실패로 인한 앞/뒤 보간)
        for seg in valid:
            # 실신호 구간 (첫 피크 ~ 마지막 피크)
            m_real = (t >= seg["t_s"]) & (t <= seg["t_e"])
            sig_rc[m_real] = np.interp(t[m_real], seg["t"], seg["sig"])
            filled[m_real] = True

            # 앞단 보간 (원래 시작 골 ~ 첫 피크)
            if seg["t_s"] > seg["label_start"]:
                m_head = (t >= seg["label_start"]) & (t < seg["t_s"])
                if m_head.sum() >= 2:
                    sig_rc[m_head] = gen_synth(t[m_head], seg["period_head"], start_type='valley', end_type='peak')
                    filled[m_head] = True
                    is_syn[m_head] = True

            # 뒷단 보간 (마지막 피크 ~ 원래 끝 골)
            if seg["label_end"] > seg["t_e"]:
                m_tail = (t > seg["t_e"]) & (t <= seg["label_end"])
                if m_tail.sum() >= 2:
                    sig_rc[m_tail] = gen_synth(t[m_tail], seg["period_tail"], start_type='peak', end_type='valley')
                    filled[m_tail] = True
                    is_syn[m_tail] = True

        # 2. 세그먼트 간 순수 갭 영역 채우기 (이전 끝 골 ~ 다음 시작 골)
        for i in range(len(valid) - 1):
            a, b    = valid[i], valid[i + 1]
            gs, ge  = a["label_end"], b["label_start"]
            if ge <= gs:
                continue
            T_start = a["period_tail"]
            T_end   = b["period_head"]
            m_gap   = (t > gs) & (t < ge)
            if m_gap.sum() < 2:
                continue
            sig_rc[m_gap] = gen_synth(t[m_gap], T_start, T_end, start_type='valley', end_type='valley')
            filled[m_gap] = True
            is_syn[m_gap] = True

        self._valid_segs = valid
        # sig_rp: 끊기지 않는 연속 재구성 (실신호 + 합성 모두 포함)
        # sig_sp: 합성 구간만 (오버레이용) — NaN이 실신호 구간을 나누지 않으므로 파란선이 끊기지 않음
        self._sig_rp = np.where(filled,  sig_rc, np.nan)
        self._sig_sp = np.where(is_syn,  sig_rc, np.nan)

        self._gaps = [
            (valid[i]["t_e"], valid[i + 1]["t_s"], valid[i], valid[i + 1])
            for i in range(len(valid) - 1)
            if valid[i + 1]["t_s"] > valid[i]["t_e"]
        ]
        self._gap_idx = 0

        self.lbl_status.setText(
            f"fs={fs:.0f} Hz  |  세그먼트: {len(valid)}/{len(self._labels)}"
            f"  |  갭: {len(self._gaps)}")

        self._draw_all()
        self._zoom_to_gap()

    def _draw_all(self):
        """곡선 + 배경 영역 전체 갱신"""
        t = self._t

        # 곡선
        self.curve_raw.setData(t, self._sig_raw)
        self.curve_real.setData( t, self._sig_rp, connect='finite')
        self.curve_synth.setData(t, self._sig_sp, connect='finite')

        # 기존 영역 아이템 제거
        for item in self._raw_region_items:
            self.plot_raw.removeItem(item)
        for item in self._rc_region_items:
            self.plot_rc.removeItem(item)
        self._raw_region_items.clear()
        self._rc_region_items.clear()

        def add_region(x0, x1, r, g, b, a):
            brush = pg.mkBrush(r, g, b, a)
            pen   = pg.mkPen(None)
            for plot, lst in ((self.plot_raw, self._raw_region_items),
                              (self.plot_rc,  self._rc_region_items)):
                item = pg.LinearRegionItem(values=[x0, x1],
                                           movable=False, brush=brush, pen=pen)
                plot.addItem(item)
                lst.append(item)

        # 라벨 구간 (파랑)
        for _, row in self._labels.iterrows():
            add_region(row["start_time"], row["end_time"], 30, 100, 220, 40)

        # peak-to-peak 실구간 (초록)
        for seg in self._valid_segs:
            add_region(seg["t_s"], seg["t_e"], 30, 180, 60, 55)

        # 갭 구간 (빨강)
        for gs, ge, *_ in self._gaps:
            add_region(gs, ge, 220, 30, 30, 35)

    def _zoom_to_gap(self):
        if not self._gaps:
            self.lbl_gap.setText("갭 없음")
            self.lbl_gap_info.setText("")
            return

        idx          = self._gap_idx
        gs, ge, a, b = self._gaps[idx]
        pad          = max(0.3, ge - gs)

        self.plot_raw.setXRange(gs - pad, ge + pad, padding=0)
        # plot_rc는 XLink로 자동 동기화

        T_s   = a["period_tail"]
        T_e   = b["period_head"]
        bpm_s = 60.0 / T_s if T_s > 0 else 0
        bpm_e = 60.0 / T_e if T_e > 0 else 0
        self.lbl_gap.setText(f"갭  {idx + 1} / {len(self._gaps)}")
        self.lbl_gap_info.setText(
            f"갭 #{idx + 1}  |  {gs:.3f}s ~ {ge:.3f}s  |  "
            f"길이 {ge - gs:.3f}s  |  "
            f"전 {T_s*1000:.1f}ms ({bpm_s:.0f}BPM) → 후 {T_e*1000:.1f}ms ({bpm_e:.0f}BPM)"
        )

    # ── 갭 네비게이션 ───────────────────────────────────────────────
    def _prev_gap(self):
        if self._gap_idx > 0:
            self._gap_idx -= 1
            self._zoom_to_gap()

    def _next_gap(self):
        if self._gap_idx < len(self._gaps) - 1:
            self._gap_idx += 1
            self._zoom_to_gap()

    def _full_view(self):
        if self._t is not None:
            self.plot_raw.setXRange(self._t[0], self._t[-1], padding=0.01)

    # ── 구간 라벨링 ─────────────────────────────────────────────────
    def _toggle_label_mode(self, checked: bool):
        self._label_mode = checked
        if checked:
            self.btn_label_mode.setStyleSheet(
                "QPushButton { background:#ffdddd; color:#cc0000; font-weight:bold; }")
            self.lbl_mark_status.setText(
                "🔴 라벨 모드 ON — 재구성 그래프에서 시작점 클릭  |  ESC=취소")
        else:
            self.btn_label_mode.setStyleSheet("")
            self._cancel_label_start()
            self.lbl_mark_status.setText(
                f"라벨: {len(self._mark_labels)}개")

    def _cancel_label_start(self):
        """시작점 대기 상태 취소."""
        if self._label_start_line is not None:
            try:
                self.plot_rc.removeItem(self._label_start_line)
            except Exception:
                pass
            self._label_start_line = None
        self._label_start_t = None

    def _on_rc_click(self, event):
        """재구성 그래프 좌클릭 → 라벨 시작/끝 지정."""
        if not self._label_mode:
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        vb  = self.plot_rc.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return

        t_click = vb.mapSceneToView(pos).x()

        if self._label_start_t is None:
            # ── 첫 번째 클릭: 시작점 지정 ──
            self._label_start_t = t_click
            line = pg.InfiniteLine(
                pos=t_click, angle=90,
                pen=pg.mkPen('#cc0000', width=2, style=Qt.DashLine),
                label=f"{t_click:.3f}s",
                labelOpts={"color": "#cc0000", "position": 0.9},
            )
            self.plot_rc.addItem(line)
            self._label_start_line = line
            self.lbl_mark_status.setText(
                f"🔴 끝점 클릭  |  시작: {t_click:.3f}s  |  ESC=취소")
        else:
            # ── 두 번째 클릭: 끝점 지정 → 구간 생성 ──
            t_start = min(self._label_start_t, t_click)
            t_end   = max(self._label_start_t, t_click)
            self._cancel_label_start()

            if t_end - t_start < 0.001:   # 너무 짧으면 무시
                self.lbl_mark_status.setText(
                    f"라벨: {len(self._mark_labels)}개  (너무 짧음 — 무시)")
                return

            self._add_label(t_start, t_end)
            event.accept()

    def _add_label(self, t_start: float, t_end: float):
        """라벨 구간 추가 + plot_rc 에 시각화."""
        self._mark_labels.append({"start": t_start, "end": t_end})
        item = pg.LinearRegionItem(
            values=[t_start, t_end],
            movable=False,
            brush=pg.mkBrush(220, 30, 30, 75),
            pen=pg.mkPen('#cc0000', width=1),
        )
        self.plot_rc.addItem(item)
        self._mark_items.append(item)
        self.lbl_mark_status.setText(
            f"라벨: {len(self._mark_labels)}개  |  "
            f"마지막: {t_start:.3f}~{t_end:.3f}s  "
            f"(길이 {t_end - t_start:.3f}s)")

    def _undo_label(self):
        """가장 마지막에 추가한 라벨 1개 제거."""
        if not self._mark_labels:
            return
        self._mark_labels.pop()
        item = self._mark_items.pop()
        try:
            self.plot_rc.removeItem(item)
        except Exception:
            pass
        self.lbl_mark_status.setText(f"라벨: {len(self._mark_labels)}개")

    def _clear_labels(self):
        """모든 라벨 제거."""
        for item in self._mark_items:
            try:
                self.plot_rc.removeItem(item)
            except Exception:
                pass
        self._mark_items.clear()
        self._mark_labels.clear()
        self._cancel_label_start()
        self.lbl_mark_status.setText("라벨: 0개")

    def _save_labels(self):
        """현재 라벨을 excluded_segments.csv 로 저장."""
        if not self.current_folder:
            self.lbl_mark_status.setText("❌ 폴더가 열려있지 않습니다")
            return
        path = os.path.join(self.current_folder, EXCL_CSV)
        rows = [
            {
                "segment_id": i + 1,
                "start_time": round(m["start"], 6),
                "end_time":   round(m["end"],   6),
                "length":     round(m["end"] - m["start"], 6),
            }
            for i, m in enumerate(self._mark_labels)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        self.lbl_mark_status.setText(
            f"✅ 저장 완료: {len(rows)}개 → {EXCL_CSV}")

    # ── 키보드 단축키 ────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._prev_gap()
        elif event.key() == Qt.Key_Right:
            self._next_gap()
        elif event.key() == Qt.Key_Home:
            self._full_view()
        elif event.key() == Qt.Key_Escape:
            # 라벨 시작점 대기 중이면 취소
            if self._label_start_t is not None:
                self._cancel_label_start()
                self.lbl_mark_status.setText(
                    f"라벨: {len(self._mark_labels)}개  (시작점 취소됨)")
        else:
            super().keyPressEvent(event)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = PPGGapViewer()
    win.show()
    sys.exit(app.exec_())
