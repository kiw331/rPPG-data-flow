# 파일명: segment_labeler.py
#
# [실행 방법]
#   python segment_labeler.py
#
# 목적: ppg_sensor.csv 에서 심박 성분이 우세한 구간을 시각적으로 라벨링
# 입력: <folder>/ppg_sensor.csv  (컬럼: Timestamp, Frame_Index, IR_Value_Raw, RED_Value_Raw)
# 출력: <folder>/hr_segments.csv (컬럼: segment_id, start_time, end_time, length)
#
# 조작 요약
#   • 마우스 좌클릭 + 드래그(영역 경계)   : 구간 양끝 이동
#   • 우클릭 / 휠                         : 그래프 축 확대/축소 (pyqtgraph 기본)
#   • SpinBox / 키보드 ↑↓                : 0.01초 단위 미세 조정
#   • 표에서 행 선택                       : 그 구간이 노란색으로 활성화
#   • 그래프 좌클릭(빈 곳)                : 선택된 구간의 가까운 경계를 클릭 위치로 스냅

import sys
import os
import csv
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QDoubleSpinBox, QComboBox, QMessageBox,
    QGroupBox, QHeaderView, QAbstractItemView, QSplitter
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

SAMPLE_RATE_DEFAULT = 200  # Hz
LABEL_FILENAME = "hr_segments.csv"
CSV_FILENAME = "ppg_sensor.csv"

BRUSH_NORMAL = pg.mkBrush(50, 200, 50, 60)
BRUSH_ACTIVE = pg.mkBrush(255, 200, 0, 110)
PEN_NORMAL = pg.mkPen(color=(30, 150, 30), width=2)
PEN_ACTIVE = pg.mkPen(color=(220, 140, 0), width=2)


class SegmentLabelerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG 심박 구간 라벨러")
        self.resize(1400, 880)

        self.current_folder = None
        self.timestamps = None       # 원본 타임스탬프 (sec)
        self.t_rel = None            # 0부터 시작하는 상대시간
        self.ir_signal = None
        self.red_signal = None
        self.fs = SAMPLE_RATE_DEFAULT
        self.segments = []           # [{'start':float,'end':float,'region':LinearRegionItem}, ...]
        self.selected_seg = None
        self._unsaved = False        # 미저장 변경사항 추적

        self.init_ui()
        self._connect_plot_click()

    # ---------------- UI ----------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # 상단: 폴더 컨트롤
        top = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("ppg_sensor.csv 가 있는 폴더 경로...")
        self.path_edit.returnPressed.connect(lambda: self.load_folder(self.path_edit.text().strip()))
        self.btn_browse = QPushButton("📁 폴더 선택")
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_reload = QPushButton("🔄 다시 불러오기")
        self.btn_reload.clicked.connect(self.reload)
        self.btn_prev = QPushButton("◀ 이전 폴더")
        self.btn_prev.clicked.connect(self.prev_folder)
        self.btn_next = QPushButton("다음 폴더 ▶")
        self.btn_next.clicked.connect(self.next_folder)
        top.addWidget(self.path_edit, stretch=1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_reload)
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        outer.addLayout(top)

        # 상태/보기채널/저장지정채널/샘플레이트
        info = QHBoxLayout()
        info.addWidget(QLabel("보기 채널:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(["IR", "RED"])
        self.combo_channel.currentTextChanged.connect(self.redraw_plot)
        info.addWidget(self.combo_channel)
        info.addSpacing(20)
        
        info.addWidget(QLabel("저장 지정 채널:"))
        self.combo_save_channel = QComboBox()
        self.combo_save_channel.addItems(["IR", "RED"])
        self.combo_save_channel.setCurrentIndex(0) # 기본 IR
        info.addWidget(self.combo_save_channel)
        info.addSpacing(20)
        
        info.addWidget(QLabel("Sampling Rate (Hz):"))
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1, 5000); self.spin_fs.setDecimals(0); self.spin_fs.setValue(SAMPLE_RATE_DEFAULT)
        self.spin_fs.valueChanged.connect(self._on_fs_changed)
        info.addWidget(self.spin_fs)
        info.addStretch()
        self.lbl_status = QLabel("폴더를 선택하세요")
        self.lbl_status.setStyleSheet("color:#555;")
        info.addWidget(self.lbl_status)
        outer.addLayout(info)

        # 그래프 영역 (Splitter: 위=원신호, 아래=미리보기)
        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter, stretch=1)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'PPG Raw')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='#1f77b4', width=1))
        splitter.addWidget(self.plot_widget)

        self.preview_widget = pg.PlotWidget()
        self.preview_widget.setBackground('w')
        self.preview_widget.showGrid(x=True, y=True, alpha=0.3)
        self.preview_widget.setLabel('left', 'Processed (z-score)')
        self.preview_widget.setLabel('bottom', 'Time (s)')
        self.preview_curve = self.preview_widget.plot(pen=pg.mkPen(color='#d62728', width=1.4))
        splitter.addWidget(self.preview_widget)
        splitter.setSizes([520, 220])

        # 미리보기 처리 옵션
        proc = QHBoxLayout()
        proc.addWidget(QLabel("BPF Low (Hz):"))
        self.spin_bpf_low = QDoubleSpinBox()
        self.spin_bpf_low.setRange(0.05, 20.0); self.spin_bpf_low.setSingleStep(0.1); self.spin_bpf_low.setValue(3.0)
        proc.addWidget(self.spin_bpf_low)
        proc.addWidget(QLabel("BPF High (Hz):"))
        self.spin_bpf_high = QDoubleSpinBox()
        self.spin_bpf_high.setRange(0.5, 50.0); self.spin_bpf_high.setSingleStep(0.5); self.spin_bpf_high.setValue(10.0)
        proc.addWidget(self.spin_bpf_high)
        proc.addWidget(QLabel("Filter Order:"))
        self.spin_order = QDoubleSpinBox()
        self.spin_order.setRange(1, 8); self.spin_order.setDecimals(0); self.spin_order.setValue(4)
        proc.addWidget(self.spin_order)
        proc.addStretch()
        self.btn_preview = QPushButton("🔍 미리보기 (선택구간)")
        self.btn_preview.clicked.connect(self.preview_segment)
        proc.addWidget(self.btn_preview)
        self.lbl_preview_stats = QLabel("")
        self.lbl_preview_stats.setStyleSheet("color:#1565C0; font-weight:bold;")
        outer.addLayout(proc)
        outer.addWidget(self.lbl_preview_stats)

        # 구간 테이블 + 컨트롤
        seg_group = QGroupBox("심박 구간 라벨")
        sv = QVBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Start (s)", "End (s)", "Length (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.table.setMaximumHeight(180)
        sv.addWidget(self.table)

        # 선택구간 미세조정
        adj = QHBoxLayout()
        adj.addWidget(QLabel("Start (s):"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 1_000_000.0); self.spin_start.setDecimals(3); self.spin_start.setSingleStep(0.01)
        self.spin_start.valueChanged.connect(self.on_spin_changed)
        adj.addWidget(self.spin_start)
        adj.addWidget(QLabel("End (s):"))
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0.0, 1_000_000.0); self.spin_end.setDecimals(3); self.spin_end.setSingleStep(0.01)
        self.spin_end.valueChanged.connect(self.on_spin_changed)
        adj.addWidget(self.spin_end)
        adj.addSpacing(20)
        adj.addWidget(QLabel("Step (s):"))
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.001, 5.0); self.spin_step.setDecimals(3); self.spin_step.setValue(0.01)
        self.spin_step.valueChanged.connect(self._on_step_changed)
        adj.addWidget(self.spin_step)
        adj.addStretch()
        sv.addLayout(adj)

        # 액션 버튼들
        btns = QHBoxLayout()
        self.btn_add = QPushButton("➕ 구간 추가")
        self.btn_add.clicked.connect(self.add_segment)
        self.btn_del = QPushButton("➖ 선택구간 삭제")
        self.btn_del.clicked.connect(self.delete_segment)
        self.btn_save = QPushButton("💾 저장")
        self.btn_save.setStyleSheet("font-weight: bold; background-color:#10b981; color:white;")
        self.btn_save.clicked.connect(self.save_segments)
        self.btn_reset = QPushButton("🗑️ 초기화 (현재 라벨 지우기)")
        self.btn_reset.setStyleSheet("color:white; background-color:#dc2626; font-weight:bold;")
        self.btn_reset.clicked.connect(self.reset_labels)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addStretch()
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_reset)
        sv.addLayout(btns)

        seg_group.setLayout(sv)
        outer.addWidget(seg_group)

    def _on_fs_changed(self, v):
        self.fs = float(v)

    def _on_step_changed(self, v):
        self.spin_start.setSingleStep(float(v))
        self.spin_end.setSingleStep(float(v))

    # ---------------- 폴더 / 데이터 ----------------
    def browse_folder(self):
        start = self.current_folder or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "ppg_sensor.csv가 있는 폴더 선택", start)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder_path):
        folder_path = os.path.normpath(folder_path)
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "오류", f"폴더 경로가 잘못되었습니다:\n{folder_path}")
            return
        csv_path = os.path.join(folder_path, CSV_FILENAME)
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "오류", f"폴더 안에 {CSV_FILENAME} 가 없습니다:\n{csv_path}")
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.warning(self, "오류", f"CSV 읽기 실패:\n{e}")
            return

        if 'Timestamp' not in df.columns or 'IR_Value_Raw' not in df.columns:
            QMessageBox.warning(self, "오류", "CSV에 'Timestamp' 또는 'IR_Value_Raw' 컬럼이 없습니다.")
            return

        self.timestamps = df['Timestamp'].to_numpy(dtype=np.float64)
        self.ir_signal = df['IR_Value_Raw'].to_numpy(dtype=np.float64)
        self.red_signal = df['RED_Value_Raw'].to_numpy(dtype=np.float64) if 'RED_Value_Raw' in df.columns else None

        t = self.timestamps - self.timestamps[0]
        self.t_rel = t

        # 추정 sampling rate
        if len(t) > 10:
            dt = np.median(np.diff(t))
            if dt > 0:
                est = round(1.0 / dt)
                self.spin_fs.blockSignals(True)
                self.spin_fs.setValue(est)
                self.spin_fs.blockSignals(False)
                self.fs = float(est)

        # 기존 구간 제거
        for s in self.segments:
            self.plot_widget.removeItem(s['region'])
        self.segments = []
        self.selected_seg = None
        self.preview_curve.setData([], [])

        self.current_folder = folder_path
        self.path_edit.setText(folder_path)

        # 라벨 자동 로드
        self.load_labels()

        # SpinBox 범위 갱신
        max_t = float(t[-1])
        for s in (self.spin_start, self.spin_end):
            s.setRange(0.0, max_t)

        self._unsaved = False
        self.lbl_preview_stats.setText("")
        self.redraw_plot()
        self.refresh_table()
        self._update_status()

    def _update_status(self):
        n = 0 if self.timestamps is None else len(self.timestamps)
        dur = 0.0 if self.t_rel is None or len(self.t_rel) == 0 else float(self.t_rel[-1])
        self.lbl_status.setText(
            f"샘플: {n} | 길이: {dur:.2f}s | fs≈{self.fs:.0f}Hz | 구간: {len(self.segments)}"
        )

    def _active_signal(self):
        ch = self.combo_channel.currentText()
        if ch == "IR":
            return self.ir_signal
        return self.red_signal if self.red_signal is not None else self.ir_signal

    def redraw_plot(self):
        if self.t_rel is None:
            return
        y = self._active_signal()
        self.curve.setData(self.t_rel, y)
        self.plot_widget.enableAutoRange()

    def reload(self):
        if self.current_folder:
            self.load_folder(self.current_folder)

    # ---------------- 형제 폴더 탐색 ----------------
    def get_sibling_folders(self):
        if not self.current_folder:
            return []
        parent = os.path.dirname(self.current_folder)
        try:
            out = []
            for name in sorted(os.listdir(parent)):
                full = os.path.normpath(os.path.join(parent, name))
                if os.path.isdir(full) and os.path.exists(os.path.join(full, CSV_FILENAME)):
                    out.append(full)
            return out
        except Exception:
            return []

    def _find_sibling_index(self, siblings):
        cur_nc = os.path.normcase(self.current_folder)
        for i, s in enumerate(siblings):
            if os.path.normcase(s) == cur_nc:
                return i
        return -1

    def _confirm_unsaved_navigate(self):
        if not self._unsaved:
            return True
        reply = QMessageBox.question(
            self, "미저장 변경사항",
            "저장하지 않은 변경사항이 있습니다. 저장하지 않고 이동할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def prev_folder(self):
        siblings = self.get_sibling_folders()
        if not siblings or not self.current_folder:
            return
        i = self._find_sibling_index(siblings)
        if i < 0:
            QMessageBox.warning(self, "오류", "현재 폴더를 형제 목록에서 찾을 수 없습니다.")
            return
        if i > 0:
            if not self._confirm_unsaved_navigate():
                return
            self.load_folder(siblings[i - 1])
        else:
            QMessageBox.information(self, "안내", "첫 폴더입니다.")

    def next_folder(self):
        siblings = self.get_sibling_folders()
        if not siblings or not self.current_folder:
            return
        i = self._find_sibling_index(siblings)
        if i < 0:
            QMessageBox.warning(self, "오류", "현재 폴더를 형제 목록에서 찾을 수 없습니다.")
            return
        if i < len(siblings) - 1:
            if not self._confirm_unsaved_navigate():
                return
            self.load_folder(siblings[i + 1])
        else:
            QMessageBox.information(self, "안내", "마지막 폴더입니다.")

    # ---------------- 라벨 I/O ----------------
    def load_labels(self):
        path = os.path.join(self.current_folder, LABEL_FILENAME)
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"라벨 읽기 실패: {e}")
            return
        if 'start_time' not in df.columns or 'end_time' not in df.columns:
            return
            
        # 저장 채널값 복원 (있을 경우)
        if 'channel' in df.columns and len(df) > 0:
            saved_ch = str(df['channel'].iloc[0]).upper()
            if saved_ch in ["IR", "RED"]:
                self.combo_save_channel.blockSignals(True)
                self.combo_save_channel.setCurrentText(saved_ch)
                self.combo_save_channel.blockSignals(False)
        else:
            # 기존 파일에 채널 정보가 없으면 기본값인 IR로 설정
            self.combo_save_channel.blockSignals(True)
            self.combo_save_channel.setCurrentIndex(0)
            self.combo_save_channel.blockSignals(False)

        for _, row in df.iterrows():
            try:
                self._add_segment_internal(float(row['start_time']), float(row['end_time']))
            except Exception:
                pass

    def save_segments(self):
        if not self.current_folder:
            return
            
        view_ch = self.combo_channel.currentText()
        save_ch = self.combo_save_channel.currentText()
        
        # 경고 1: 보고 있는 채널과 저장 지정 채널이 다를 경우
        if view_ch != save_ch:
            reply = QMessageBox.question(
                self, "보기 채널과 저장 채널 불일치",
                f"현재 화면에서 보고 있는 채널({view_ch})과 저장 지정 채널({save_ch})이 다릅니다.\n"
                f"정말 {save_ch} 채널로 저장하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
        # 경고 2: 기존에 저장된 채널과 현재 선택한 저장 지정 채널이 다를 경우
        path = os.path.join(self.current_folder, LABEL_FILENAME)
        if os.path.exists(path):
            try:
                old_df = pd.read_csv(path)
                if 'channel' in old_df.columns and len(old_df) > 0:
                    old_ch = str(old_df['channel'].iloc[0]).upper()
                    if old_ch != save_ch:
                        reply = QMessageBox.question(
                            self, "기존 저장 채널과 불일치",
                            f"이전에 저장된 파일의 채널({old_ch})과 현재 선택한 저장 지정 채널({save_ch})이 다릅니다.\n"
                            f"기존 파일을 {save_ch} 채널로 덮어씌우시겠습니까?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                        )
                        if reply != QMessageBox.Yes:
                            return
            except Exception:
                pass

        if not self.segments:
            reply = QMessageBox.question(
                self, "확인",
                "저장할 구간이 없습니다. 빈 파일로 저장(또는 기존 파일 비우기)할까요?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        # 시작시간 기준 정렬
        segs = sorted(self.segments, key=lambda x: x['start'])
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['segment_id', 'start_time', 'end_time', 'length', 'channel'])
                for i, s in enumerate(segs, 1):
                    w.writerow([
                        i, 
                        f"{s['start']:.4f}", 
                        f"{s['end']:.4f}", 
                        f"{s['end']-s['start']:.4f}",
                        save_ch
                    ])
            self._unsaved = False
            self.lbl_status.setText(f"저장 완료: {path}")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", str(e))
            QMessageBox.warning(self, "저장 실패", str(e))

    def reset_labels(self):
        if not self.current_folder:
            return
        reply = QMessageBox.question(
            self, "확인",
            "현재 라벨링된 모든 구간을 지웁니다.\n(파일은 삭제하지 않음) 진행할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for s in self.segments:
            self.plot_widget.removeItem(s['region'])
        self.segments = []
        self.selected_seg = None
        self._unsaved = False
        self.preview_curve.setData([], [])
        self.lbl_preview_stats.setText("")
        self.refresh_table()
        self._update_status()

    # ---------------- 구간 관리 ----------------
    def add_segment(self):
        if self.t_rel is None:
            QMessageBox.warning(self, "오류", "먼저 폴더를 선택하세요.")
            return
        # 현재 보이는 X 범위의 중앙에 길이 5초 (또는 가능 범위) 구간 생성
        view = self.plot_widget.viewRange()
        x_min, x_max = view[0]
        center = max(0.0, min(self.t_rel[-1], (x_min + x_max) / 2.0))
        span = max(0.5, min(5.0, (x_max - x_min) * 0.2))
        start = max(0.0, center - span / 2.0)
        end = min(self.t_rel[-1], center + span / 2.0)
        if end <= start:
            start, end = 0.0, min(2.0, float(self.t_rel[-1]))
        self._add_segment_internal(start, end)
        # 방금 추가한 행 선택
        self.table.selectRow(len(self.segments) - 1)

    def _add_segment_internal(self, start, end):
        if end < start:
            start, end = end, start
        region = pg.LinearRegionItem(
            values=[start, end],
            brush=BRUSH_NORMAL,
            pen=PEN_NORMAL,
            movable=True
        )
        region.setZValue(-10)
        self.plot_widget.addItem(region)
        seg = {'start': float(start), 'end': float(end), 'region': region}
        self.segments.append(seg)
        region.sigRegionChanged.connect(lambda r, sref=seg: self._on_region_changed(sref))
        self._unsaved = True
        self.refresh_table()
        self._update_status()

    def delete_segment(self):
        if not self.selected_seg:
            return
        self.plot_widget.removeItem(self.selected_seg['region'])
        self.segments.remove(self.selected_seg)
        self.selected_seg = None
        self._unsaved = True
        self.refresh_table()
        self._update_status()

    def _on_region_changed(self, seg):
        s, e = seg['region'].getRegion()
        if s > e:
            s, e = e, s
        seg['start'], seg['end'] = float(s), float(e)
        self._unsaved = True
        # 같은 seg를 가리키는 행 갱신
        idx = self.segments.index(seg)
        self._refresh_table_row(idx)
        if self.selected_seg is seg:
            self.spin_start.blockSignals(True); self.spin_end.blockSignals(True)
            self.spin_start.setValue(seg['start']); self.spin_end.setValue(seg['end'])
            self.spin_start.blockSignals(False); self.spin_end.blockSignals(False)

    # ---------------- 테이블 ----------------
    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.segments))
        for i in range(len(self.segments)):
            self._refresh_table_row(i)
        self.table.blockSignals(False)

    def _refresh_table_row(self, i):
        s = self.segments[i]
        items = [
            QTableWidgetItem(str(i + 1)),
            QTableWidgetItem(f"{s['start']:.3f}"),
            QTableWidgetItem(f"{s['end']:.3f}"),
            QTableWidgetItem(f"{s['end'] - s['start']:.3f}"),
        ]
        for c, it in enumerate(items):
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, c, it)

    def on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected_seg = None
            self._highlight_active(None)
            return
        i = rows[0].row()
        if 0 <= i < len(self.segments):
            seg = self.segments[i]
            self.selected_seg = seg
            self.spin_start.blockSignals(True); self.spin_end.blockSignals(True)
            self.spin_start.setValue(seg['start']); self.spin_end.setValue(seg['end'])
            self.spin_start.blockSignals(False); self.spin_end.blockSignals(False)
            self._highlight_active(seg)

    def _highlight_active(self, active_seg):
        for s in self.segments:
            if s is active_seg:
                s['region'].setBrush(BRUSH_ACTIVE)
                s['region'].setRegion([s['start'], s['end']])  # ensure on top visually
            else:
                s['region'].setBrush(BRUSH_NORMAL)

    def on_spin_changed(self):
        if not self.selected_seg:
            return
        s = self.spin_start.value()
        e = self.spin_end.value()
        if s > e:
            return
        seg = self.selected_seg
        seg['start'], seg['end'] = float(s), float(e)
        seg['region'].blockSignals(True)
        seg['region'].setRegion([s, e])
        seg['region'].blockSignals(False)
        self._unsaved = True
        idx = self.segments.index(seg)
        self._refresh_table_row(idx)

    # ---------------- 그래프 클릭 → 가까운 경계 스냅 ----------------
    def _connect_plot_click(self):
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

    def _on_plot_clicked(self, ev):
        if self.selected_seg is None or self.t_rel is None:
            return
        # 좌클릭만, 더블클릭/우클릭 제외
        if ev.button() != Qt.LeftButton:
            return
        # LinearRegionItem 경계를 직접 잡은 경우는 pyqtgraph가 처리하도록 두기
        if ev.isAccepted():
            return
        vb = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(ev.scenePos())
        x = float(pos.x())
        x = max(0.0, min(float(self.t_rel[-1]), x))
        seg = self.selected_seg
        # 가까운 경계로 스냅
        if abs(x - seg['start']) <= abs(x - seg['end']):
            new_start = min(x, seg['end'])
            seg['start'] = new_start
        else:
            new_end = max(x, seg['start'])
            seg['end'] = new_end
        seg['region'].blockSignals(True)
        seg['region'].setRegion([seg['start'], seg['end']])
        seg['region'].blockSignals(False)
        self._unsaved = True
        # SpinBox 동기화
        self.spin_start.blockSignals(True); self.spin_end.blockSignals(True)
        self.spin_start.setValue(seg['start']); self.spin_end.setValue(seg['end'])
        self.spin_start.blockSignals(False); self.spin_end.blockSignals(False)
        idx = self.segments.index(seg)
        self._refresh_table_row(idx)

    # ---------------- 미리보기 (detrend + BPF + zscore) ----------------
    def preview_segment(self):
        if not self.selected_seg:
            QMessageBox.information(self, "안내", "구간을 먼저 선택하세요.")
            return
        if self.t_rel is None:
            return
        seg = self.selected_seg
        y_all = self._active_signal()
        mask = (self.t_rel >= seg['start']) & (self.t_rel <= seg['end'])
        if not np.any(mask):
            return
        x = self.t_rel[mask]
        y = y_all[mask].astype(np.float64)
        if len(y) < 20:
            QMessageBox.warning(self, "오류", "구간이 너무 짧습니다.")
            return

        # 1) Detrend
        y_det = sp_signal.detrend(y, type='linear')

        # 2) Bandpass filter
        low = float(self.spin_bpf_low.value())
        high = float(self.spin_bpf_high.value())
        order = int(self.spin_order.value())
        nyq = self.fs / 2.0
        low_n = max(1e-4, min(0.999, low / nyq))
        high_n = max(low_n + 1e-4, min(0.999, high / nyq))
        try:
            b, a = sp_signal.butter(order, [low_n, high_n], btype='band')
            # filtfilt padlen 안전
            padlen = min(3 * max(len(a), len(b)), len(y_det) - 1)
            y_bpf = sp_signal.filtfilt(b, a, y_det, padlen=padlen)
        except Exception as e:
            QMessageBox.warning(self, "필터 오류", str(e))
            return

        # 3) Z-score
        std = np.std(y_bpf)
        z = (y_bpf - np.mean(y_bpf)) / (std + 1e-12)

        self.preview_curve.setData(x, z)
        self.preview_widget.enableAutoRange()

        # 4) 피크 검출 → BPM / 주기 통계
        try:
            min_dist = max(1, int(self.fs * 0.1))
            peaks, _ = sp_signal.find_peaks(z, distance=min_dist, prominence=0.5)
            if len(peaks) >= 2:
                peak_times = x[peaks]
                intervals = np.diff(peak_times)
                mean_period = float(np.mean(intervals))
                median_period = float(np.median(intervals))
                bpm = 60.0 / mean_period if mean_period > 0 else 0.0
                stats_text = (
                    f"피크 수: {len(peaks)}  |  "
                    f"BPM: {bpm:.1f}  |  "
                    f"평균 주기: {mean_period:.3f}s  |  "
                    f"중앙값 주기: {median_period:.3f}s"
                )
            else:
                stats_text = "피크 감지 부족 (최소 2개 이상 필요 — 구간을 넓히거나 필터 범위를 조정하세요)"
            self.lbl_preview_stats.setText(stats_text)
        except Exception as e:
            self.lbl_preview_stats.setText(f"피크 분석 실패: {e}")


def main():
    app = QApplication(sys.argv)
    w = SegmentLabelerApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
