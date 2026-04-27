import os
import cv2
import json
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tifffile
from scipy.signal import butter, filtfilt, find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                             QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
                             QFileDialog, QMessageBox, QGroupBox, QProgressBar, QSplitter,
                             QScrollArea, QDialog, QCheckBox, QScrollBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer

# =====================================================================
# 분석 스레드 (QThread)
# =====================================================================
class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict) # 결과 데이터 딕셔너리 전달
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path, start_frame_idx, duration_sec, rois_info, bg_roi_info, bpm_min, bpm_max):
        super().__init__()
        self.folder_path = folder_path
        self.start_frame_idx = start_frame_idx
        self.duration_sec = duration_sec
        self.rois_info = rois_info # list of (x, y, w, h) for ROI 1~3
        self.bg_roi_info = bg_roi_info # (x, y, w, h) or None
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.is_running = True

    def run(self):
        try:
            self.progress_signal.emit(0, "메타데이터 및 CSV 확인 중...")
            frames_dir = os.path.join(self.folder_path, "frames")
            if not os.path.exists(frames_dir):
                frames_dir = self.folder_path

            # 타임스탬프 로드
            cam_csv_path = os.path.join(self.folder_path, "camera_timestamps.csv")
            if not os.path.exists(cam_csv_path):
                raise Exception("camera_timestamps.csv 파일이 없습니다.")
            df_cam = pd.read_csv(cam_csv_path)

            sensor_csv_path = os.path.join(self.folder_path, "ppg_sensor.csv")
            has_sensor = os.path.exists(sensor_csv_path)
            if has_sensor:
                df_sensor = pd.read_csv(sensor_csv_path)
                t0 = min(df_sensor['Timestamp'].iloc[0], df_cam['Timestamp'].iloc[0])
            else:
                df_sensor = None
                t0 = df_cam['Timestamp'].iloc[0]

            # 분석 시간 구간 마스크
            start_time_offset = df_cam.loc[df_cam['Frame_Index'] == self.start_frame_idx, 'Timestamp'].values[0] - t0
            end_time_offset = start_time_offset + self.duration_sec

            cam_mask = (df_cam['Timestamp'] - t0 >= start_time_offset) & (df_cam['Timestamp'] - t0 <= end_time_offset)
            df_cam_cut = df_cam[cam_mask].copy()

            if len(df_cam_cut) == 0:
                raise Exception("지정된 시간에 해당하는 프레임 데이터가 없습니다.")

            # 센서 데이터 자르기
            t_sensor = np.array([])
            y_sensor = np.array([])
            if has_sensor:
                sensor_mask = (df_sensor['Timestamp'] - t0 >= start_time_offset) & (df_sensor['Timestamp'] - t0 <= end_time_offset)
                df_sensor_cut = df_sensor[sensor_mask]
                t_sensor = df_sensor_cut['Timestamp'].values - t0
                if 'IR_Value_Raw' in df_sensor_cut.columns:
                    y_sensor = df_sensor_cut['IR_Value_Raw'].values
                elif 'IR_Value' in df_sensor_cut.columns:
                    y_sensor = df_sensor_cut['IR_Value'].values

            # 데이터 추출 루프
            total_frames = len(df_cam_cut)
            t_cam_list = []
            frame_indices = []
            
            # shape: [ROI수+배경수, 프레임수, 3(RGB)]
            roi_count = len(self.rois_info)
            has_bg = self.bg_roi_info is not None
            total_regions = roi_count + (1 if has_bg else 0)
            
            all_rois = list(self.rois_info)
            if has_bg:
                all_rois.append(self.bg_roi_info)

            # [region_idx][frame_idx][channel(0:R, 1:G, 2:B)]
            raw_rgb_data = np.zeros((total_regions, total_frames, 3), dtype=np.float32)

            self.progress_signal.emit(10, f"프레임 데이터 추출 시작 (총 {total_frames} 프레임)")

            for idx, (index, row) in enumerate(df_cam_cut.iterrows()):
                if not self.is_running:
                    return

                f_idx = int(row['Frame_Index'])
                f_time = row['Timestamp'] - t0
                
                f_path = os.path.join(frames_dir, f"frame_{f_idx:04d}.tiff")
                if not os.path.exists(f_path):
                    f_path = os.path.join(frames_dir, f"frame_{f_idx:04d}.tif")
                    if not os.path.exists(f_path):
                        continue # 프레임 스킵

                img = tifffile.imread(f_path) if f_path.lower().endswith(('.tiff', '.tif')) else cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                
                # 컬러 변환 (BayerRG12 픽셀 포맷에 맞춤)
                if len(img.shape) == 2:
                    # BayerRG to RGB 직접 변환
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for r_idx, (x, y, w, h) in enumerate(all_rois):
                    # 경계 처리
                    y_start = max(0, int(y))
                    y_end = min(img_rgb.shape[0], int(y+h))
                    x_start = max(0, int(x))
                    x_end = min(img_rgb.shape[1], int(x+w))
                    
                    crop = img_rgb[y_start:y_end, x_start:x_end]
                    # R, G, B 평균 계산
                    if crop.size > 0:
                        r_mean = np.mean(crop[:, :, 0])
                        g_mean = np.mean(crop[:, :, 1])
                        b_mean = np.mean(crop[:, :, 2])
                        raw_rgb_data[r_idx, idx, 0] = r_mean
                        raw_rgb_data[r_idx, idx, 1] = g_mean
                        raw_rgb_data[r_idx, idx, 2] = b_mean

                t_cam_list.append(f_time)
                frame_indices.append(f_idx)

                if idx % 10 == 0:
                    prog = 10 + int(70 * (idx / total_frames))
                    self.progress_signal.emit(prog, f"추출 중... ({idx}/{total_frames})")

            t_cam = np.array(t_cam_list)
            
            # FPS 계산
            fs_cam = 1.0 / np.mean(np.diff(t_cam)) if len(t_cam) > 1 else 60.0
            fs_sensor = 1.0 / np.mean(np.diff(t_sensor)) if len(t_sensor) > 1 else 0

            # 결과 데이터 포장
            result = {
                't_cam': t_cam,
                'raw_rgb_data': raw_rgb_data, # [region_idx, frame_idx, RGB]
                'frame_indices': frame_indices,
                'roi_count': roi_count,
                'has_bg': has_bg,
                'has_sensor': has_sensor,
                't_sensor': t_sensor,
                'y_sensor': y_sensor,
                'fs_cam': fs_cam,
                'fs_sensor': fs_sensor,
                'bpm_min': self.bpm_min,
                'bpm_max': self.bpm_max
            }

            self.progress_signal.emit(100, "분석 완료")
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self.is_running = False


# =====================================================================
# 세부 분석 팝업 UI (GraphPopup)
# =====================================================================
class GraphPopup(QDialog):
    def __init__(self, parent=None, analysis_result=None, folder_path="", spin_duration_val=0, start_frame_idx=0):
        super().__init__(parent)
        self.setWindowTitle("상세 그래프 분석")
        self.resize(1000, 800)
        self.setModal(True)
        
        self.analysis_result = analysis_result
        self.folder_path = folder_path
        self.spin_duration_val = spin_duration_val
        self.start_frame_idx = start_frame_idx

        self.setup_ui()
        if self.analysis_result:
            self.update_graphs()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 상단 컨트롤: 채널, 체크박스, CSV저장
        top_layout = QHBoxLayout()
        
        # 채널 선택
        group_channel = QGroupBox("결과 채널 선택")
        chan_layout = QHBoxLayout()
        self.radio_R = QRadioButton("R")
        self.radio_G = QRadioButton("G")
        self.radio_B = QRadioButton("B")
        self.radio_R.setChecked(True)
        self.bg_channel = QButtonGroup()
        self.bg_channel.addButton(self.radio_R, 0)
        self.bg_channel.addButton(self.radio_G, 1)
        self.bg_channel.addButton(self.radio_B, 2)
        chan_layout.addWidget(self.radio_R)
        chan_layout.addWidget(self.radio_G)
        chan_layout.addWidget(self.radio_B)
        group_channel.setLayout(chan_layout)
        top_layout.addWidget(group_channel)
        
        self.bg_channel.idToggled.connect(self.on_channel_changed)

        # 그래프 표시 선택
        group_show = QGroupBox("분석할 그래프 보기")
        show_layout = QHBoxLayout()
        self.chk_roi1 = QCheckBox("ROI 1")
        self.chk_roi2 = QCheckBox("ROI 2")
        self.chk_roi3 = QCheckBox("ROI 3")
        self.chk_filt = QCheckBox("Filtered Signals")
        self.chk_ppg = QCheckBox("PPG Raw")
        self.chk_bg = QCheckBox("Background")
        
        self.chks = [self.chk_roi1, self.chk_roi2, self.chk_roi3, self.chk_filt, self.chk_ppg, self.chk_bg]
        for chk in self.chks:
            chk.setChecked(True)
            chk.stateChanged.connect(self.update_graphs)
            show_layout.addWidget(chk)
            
        group_show.setLayout(show_layout)
        top_layout.addWidget(group_show)

        top_layout.addStretch()

        self.btn_save_csv = QPushButton("Csv로 저장")
        self.btn_save_csv.clicked.connect(self.save_csv)
        top_layout.addWidget(self.btn_save_csv)

        main_layout.addLayout(top_layout)

        # 그래프 영역
        self.graph_layout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_layout, stretch=1)

        self.plots = []
        for i in range(6):
            p = self.graph_layout.addPlot()
            p.showGrid(x=True, y=True)
            self.plots.append(p)
            self.graph_layout.nextRow()

        # 하단: Zoom / Pan (QSlider + QScrollBar)
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("X축 줌:"))
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(10, 100) # 10% to 100% of data
        self.slider_zoom.setValue(100)
        self.slider_zoom.valueChanged.connect(self.update_x_range)
        bottom_layout.addWidget(self.slider_zoom)

        bottom_layout.addWidget(QLabel("X축 이동:"))
        self.scrollbar_pan = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar_pan.setRange(0, 0)
        self.scrollbar_pan.valueChanged.connect(self.update_x_range)
        bottom_layout.addWidget(self.scrollbar_pan)

        main_layout.addLayout(bottom_layout)

    def on_channel_changed(self, btn_id, checked):
        if checked:
            self.update_graphs()

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1.0: high = 0.99
        if low <= 0: low = 0.01
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def update_graphs(self):
        if not self.analysis_result: return

        ch_idx = self.bg_channel.checkedId()
        ch_name = ['R', 'G', 'B'][ch_idx]

        res = self.analysis_result
        t_cam = res['t_cam']
        t_cam_plot = t_cam - t_cam[0] if len(t_cam) > 0 else t_cam
        raw_rgb = res['raw_rgb_data']
        roi_count = res['roi_count']
        has_bg = res['has_bg']
        has_sensor = res['has_sensor']
        fs_cam = res['fs_cam']
        bpm_min, bpm_max = res['bpm_min'], res['bpm_max']
        low_cut = bpm_min / 60.0
        high_cut = bpm_max / 60.0

        plot_colors = [(255,0,0), (0,255,0), (0,0,255)]

        self.graph_layout.clear()
        
        if len(t_cam_plot) > 0:
            self.max_time = t_cam_plot[-1]
        else:
            self.max_time = 1.0

        active_plots = []

        for i in range(roi_count):
            if i < 3 and self.chks[i].isChecked():
                p = self.plots[i]
                p.clear()
                self.graph_layout.addItem(p)
                self.graph_layout.nextRow()
                y_raw = raw_rgb[i, :, ch_idx]
                p.plot(t_cam_plot, y_raw, pen=pg.mkPen(plot_colors[i], width=2))
                p.setTitle(f"ROI {i+1} Raw [{ch_name}]")
                p.enableAutoRange(axis=pg.ViewBox.YAxis)
                active_plots.append(p)

        if self.chk_filt.isChecked():
            p_filt = self.plots[3]
            p_filt.clear()
            self.graph_layout.addItem(p_filt)
            self.graph_layout.nextRow()
            p_filt.setTitle(f"Filtered Signals [{ch_name}]")
            p_filt.addLegend()
            for i in range(roi_count):
                y_raw = raw_rgb[i, :, ch_idx]
                y_c = y_raw - np.mean(y_raw)
                try:
                    y_filt = self.butter_bandpass_filter(y_c, low_cut, high_cut, fs_cam)
                    min_dist = 60.0 / bpm_max
                    peaks, _ = find_peaks(y_filt, distance=int(min_dist * fs_cam), prominence=np.std(y_filt)*0.5)
                    bpm = 60.0 / np.mean(np.diff(t_cam[peaks])) if len(peaks) > 1 else 0
                    p_filt.plot(t_cam_plot, y_filt, pen=pg.mkPen(plot_colors[i], width=2), name=f"ROI {i+1} (BPM: {bpm:.1f})")
                    if len(peaks) > 0:
                        scatter = pg.ScatterPlotItem(x=t_cam_plot[peaks], y=y_filt[peaks], pen=None, brush=pg.mkBrush(plot_colors[i]), size=8, symbol='x')
                        p_filt.addItem(scatter)
                except:
                    pass
            p_filt.enableAutoRange(axis=pg.ViewBox.YAxis)
            active_plots.append(p_filt)

        if has_sensor and self.chk_ppg.isChecked():
            p_ppg = self.plots[4]
            p_ppg.clear()
            self.graph_layout.addItem(p_ppg)
            self.graph_layout.nextRow()
            t_sensor = res['t_sensor']
            y_sensor = res['y_sensor']
            t_sensor_plot = t_sensor - t_cam[0] if len(t_sensor) > 0 else t_sensor
            p_ppg.plot(t_sensor_plot, y_sensor, pen=pg.mkPen(200,200,200, width=2))
            p_ppg.setTitle("PPG Raw Sensor Data")
            p_ppg.enableAutoRange(axis=pg.ViewBox.YAxis)
            active_plots.append(p_ppg)

        if has_bg and self.chk_bg.isChecked():
            p_bg = self.plots[5]
            p_bg.clear()
            self.graph_layout.addItem(p_bg)
            self.graph_layout.nextRow()
            y_bg_raw = raw_rgb[roi_count, :, ch_idx]
            p_bg.plot(t_cam_plot, y_bg_raw, pen=pg.mkPen((200,0,200), width=2))
            p_bg.setTitle(f"Background Raw [{ch_name}]")
            p_bg.enableAutoRange(axis=pg.ViewBox.YAxis)
            active_plots.append(p_bg)

        for i, p in enumerate(active_plots):
            if i > 0:
                p.setXLink(active_plots[0])
            else:
                p.setXLink(None)

        self.scrollbar_pan.blockSignals(True)
        self.slider_zoom.blockSignals(True)
        self.scrollbar_pan.setRange(0, 1000)
        self.scrollbar_pan.setValue(0)
        self.slider_zoom.setValue(100)
        self.scrollbar_pan.blockSignals(False)
        self.slider_zoom.blockSignals(False)

        self.update_x_range()

    def update_x_range(self):
        if not hasattr(self, 'max_time'): return
        zoom_pct = self.slider_zoom.value() / 100.0
        window_size = self.max_time * zoom_pct
        
        if zoom_pct >= 0.999:
            self.scrollbar_pan.setEnabled(False)
            start_x = 0.0
        else:
            self.scrollbar_pan.setEnabled(True)
            pan_val = self.scrollbar_pan.value() / 1000.0
            max_start = self.max_time - window_size
            start_x = max_start * pan_val

        end_x = start_x + window_size
        
        for p in self.plots:
            if p.isVisible() or p in [i for i in self.plots if i.plotItem.scene() is not None]:
                p.setXRange(start_x, end_x, padding=0)
                break

    def save_csv(self):
        if not self.analysis_result: return
        folder_name = os.path.basename(self.folder_path)
        default_filename = f"{folder_name}_{self.start_frame_idx}_{self.spin_duration_val}s.csv"
        
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        default_filepath = os.path.join(output_dir, default_filename)
        
        save_path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", default_filepath, "CSV Files (*.csv)")
        if not save_path: return

        res = self.analysis_result
        frame_indices = res['frame_indices']
        raw_rgb = res['raw_rgb_data']
        roi_count = res['roi_count']
        has_bg = res['has_bg']
        has_sensor = res['has_sensor']
        t_cam = res['t_cam']
        t_sensor = res['t_sensor']
        y_sensor = res['y_sensor']
        
        mapped_ppg = []
        if has_sensor and len(t_sensor) > 0:
            for tc in t_cam:
                idx = (np.abs(t_sensor - tc)).argmin()
                mapped_ppg.append(y_sensor[idx])
        else:
            mapped_ppg = [np.nan] * len(t_cam)

        fps = res['fs_cam']

        rows = []
        for i, f_idx in enumerate(frame_indices):
            save_f_idx = i + 1
            for r in range(roi_count):
                rows.append({
                    "Frame": save_f_idx,
                    "Type": f"roi{r+1}",
                    "R": raw_rgb[r, i, 0],
                    "G": raw_rgb[r, i, 1],
                    "B": raw_rgb[r, i, 2],
                    "PPG": mapped_ppg[i],
                    "FPS": fps
                })
            if has_bg:
                bg_idx = roi_count
                rows.append({
                    "Frame": save_f_idx,
                    "Type": "background",
                    "R": raw_rgb[bg_idx, i, 0],
                    "G": raw_rgb[bg_idx, i, 1],
                    "B": raw_rgb[bg_idx, i, 2],
                    "PPG": mapped_ppg[i],
                    "FPS": fps
                })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        QMessageBox.information(self, "완료", f"데이터가 저장되었습니다.\n{save_path}")


# =====================================================================
# 독립 RGB 히스토그램 창 (HistPopup)
# =====================================================================
class HistPopup(QDialog):
    def __init__(self, parent=None, frames_dir="", df_cam=None, fps=30.0, current_f_idx=0, min_f_idx=0, max_f_idx=0):
        super().__init__(parent)
        self.setWindowTitle("RGB 히스토그램 전용 창")
        self.resize(1000, 600)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.frames_dir = frames_dir
        self.df_cam = df_cam
        self.fps = fps
        self.current_frame_idx = current_f_idx
        self.min_frame_idx = min_f_idx
        self.max_frame_idx = max_f_idx

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)

        self.setup_ui()
        self.show_frame(self.current_frame_idx)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # 좌측: 뷰어 및 탐색바
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)

        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        left_layout.addWidget(self.image_view, stretch=1)

        # 단일 ROI 지정
        self.roi = pg.RectROI([100, 100], [150, 150], pen=pg.mkPen('y', width=3))
        self.image_view.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_histogram)

        # 하단 탐색바
        nav_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("재생")
        self.btn_play_pause.clicked.connect(self.toggle_play)
        
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_prev.clicked.connect(lambda: self.seek_frame(-1))
        self.btn_next.clicked.connect(lambda: self.seek_frame(1))

        self.slider_play = QSlider(Qt.Orientation.Horizontal)
        self.slider_play.setRange(self.min_frame_idx, self.max_frame_idx)
        self.slider_play.setValue(self.current_frame_idx)
        self.slider_play.valueChanged.connect(self.on_slider_changed)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(self.min_frame_idx, self.max_frame_idx)
        self.spin_frame.setValue(self.current_frame_idx)
        self.spin_frame.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_frame.valueChanged.connect(self.on_spin_changed)

        nav_layout.addWidget(self.btn_play_pause)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.slider_play)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.spin_frame)
        left_layout.addLayout(nav_layout)
        main_layout.addWidget(left_panel, stretch=2)

        # 우측: 히스토그램
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.hist_plot = pg.PlotWidget(title="ROI RGB 히스토그램")
        self.hist_plot.showGrid(x=False, y=True)
        self.curve_hist_r = self.hist_plot.plot(pen=pg.mkPen('r', width=2))
        self.curve_hist_g = self.hist_plot.plot(pen=pg.mkPen('g', width=2))
        self.curve_hist_b = self.hist_plot.plot(pen=pg.mkPen('b', width=2))
        
        right_layout.addWidget(self.hist_plot)

        # X축 조절 UI 추가
        x_ctrl_layout = QHBoxLayout()
        x_ctrl_layout.addWidget(QLabel("X축 최소:"))
        self.spin_x_min = QSpinBox()
        self.spin_x_min.setRange(0, 65535)
        self.spin_x_min.setValue(0)
        self.spin_x_min.setKeyboardTracking(False)
        x_ctrl_layout.addWidget(self.spin_x_min)
        
        x_ctrl_layout.addWidget(QLabel("최대:"))
        self.spin_x_max = QSpinBox()
        self.spin_x_max.setRange(1, 65536)
        self.spin_x_max.setValue(256)
        self.spin_x_max.setKeyboardTracking(False)
        x_ctrl_layout.addWidget(self.spin_x_max)

        self.spin_x_min.valueChanged.connect(self.update_histogram)
        self.spin_x_max.valueChanged.connect(self.update_histogram)
        
        right_layout.addLayout(x_ctrl_layout)
        main_layout.addWidget(right_panel, stretch=1)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play_pause.setText("재생")
        else:
            if self.current_frame_idx >= self.max_frame_idx:
                self.current_frame_idx = self.min_frame_idx
                self.spin_frame.setValue(self.current_frame_idx)
            fps_val = self.fps if self.fps > 0 else 30
            self.play_timer.start(int(1000 / fps_val))
            self.btn_play_pause.setText("일시정지")

    def play_next_frame(self):
        if self.current_frame_idx < self.max_frame_idx:
            self.spin_frame.setValue(self.current_frame_idx + 1)
        else:
            self.toggle_play()

    def on_slider_changed(self, val):
        if self.spin_frame.value() != val:
            self.spin_frame.setValue(val)

    def seek_frame(self, offset):
        new_val = self.spin_frame.value() + offset
        if self.min_frame_idx <= new_val <= self.max_frame_idx:
            self.spin_frame.setValue(new_val)

    def on_spin_changed(self, val):
        self.current_frame_idx = val
        if self.slider_play.value() != val:
            self.slider_play.blockSignals(True)
            self.slider_play.setValue(val)
            self.slider_play.blockSignals(False)
        self.show_frame(val)

    def show_frame(self, f_idx):
        if not self.frames_dir: return
        f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tiff")
        if not os.path.exists(f_path):
            f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tif")
            if not os.path.exists(f_path):
                return

        img = tifffile.imread(f_path) if f_path.lower().endswith(('.tiff', '.tif')) else cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        
        # pyqtgraph ImageView는 BGR 순서를 기대하므로 BayerRG2BGR 사용
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
        else:
            img_display = img  # BGR8 저장 파일은 이미 BGR

        # 히스토그램 계산용 RGB 별도 보관
        if len(img.shape) == 2:
            self.current_img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        else:
            self.current_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pg = np.transpose(img_display, (1, 0, 2))
        self.image_view.setImage(img_pg, autoRange=False)

        if not hasattr(self, 'x_range_initialized'):
            self.spin_x_min.blockSignals(True)
            self.spin_x_max.blockSignals(True)
            
            self.spin_x_min.setValue(0)
            if img_rgb.dtype == np.uint8:
                self.spin_x_max.setValue(256)
            else:
                img_max = img_rgb.max()
                if img_max <= 4096:
                    self.spin_x_max.setValue(4096)
                elif img_max <= 16384:
                    self.spin_x_max.setValue(16384)
                else:
                    self.spin_x_max.setValue(65536)
                    
            self.spin_x_min.blockSignals(False)
            self.spin_x_max.blockSignals(False)
            self.x_range_initialized = True

        self.update_histogram()

    def update_histogram(self):
        if not hasattr(self, 'current_img_rgb'): return
        img_rgb = self.current_img_rgb
        
        pos = self.roi.pos()
        size = self.roi.size()
        
        x, y = int(pos[0]), int(pos[1])
        w, h = int(size[0]), int(size[1])
        
        img_h, img_w = img_rgb.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return

        roi_img = img_rgb[y1:y2, x1:x2]
        if roi_img.size == 0: return

        xmin = self.spin_x_min.value()
        xmax = self.spin_x_max.value()
        if xmin >= xmax:
            xmax = xmin + 1

        bins = 256
        ranges = [float(xmin), float(xmax)]
        
        hist_r = cv2.calcHist([roi_img], [0], None, [bins], ranges)
        hist_g = cv2.calcHist([roi_img], [1], None, [bins], ranges)
        hist_b = cv2.calcHist([roi_img], [2], None, [bins], ranges)
        
        x_vals = np.linspace(ranges[0], ranges[1], bins, endpoint=False)
        self.curve_hist_r.setData(x=x_vals, y=hist_r.flatten())
        self.curve_hist_g.setData(x=x_vals, y=hist_g.flatten())
        self.curve_hist_b.setData(x=x_vals, y=hist_b.flatten())
        
        self.hist_plot.setXRange(xmin, xmax, padding=0)

    def closeEvent(self, event):
        if self.play_timer.isActive():
            self.play_timer.stop()
        event.accept()


# =====================================================================
# 메인 윈도우 UI
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG Analysis Tool")
        self.resize(1400, 900)

        # 상태 변수
        self.folder_path = ""
        self.df_cam = None
        self.current_frame_idx = 0
        self.min_frame_idx = 0
        self.max_frame_idx = 0
        self.frames_dir = ""
        self.pixel_format = "BayerRG12"
        self.fps = 60.0
        
        self.roi_items = [] # pyqtgraph ROI 객체 리스트
        self.bg_roi_item = None
        self.saved_roi_states = []
        self.saved_bg_state = None
        
        self.analysis_result = None # Worker에서 받은 결과 저장

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ==================== 좌측 패널 (영상 및 컨트롤) ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 영상 Viewer
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        left_layout.addWidget(self.image_view, stretch=1)

        # 2. 재생 바 컨트롤
        play_ctrl_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("재생")
        self.btn_play_pause.setEnabled(False)
        self.slider_play = QSlider(Qt.Orientation.Horizontal)
        self.slider_play.setEnabled(False)
        
        play_ctrl_layout.addWidget(self.btn_play_pause)
        play_ctrl_layout.addWidget(self.slider_play)
        left_layout.addLayout(play_ctrl_layout)
        
        self.btn_play_pause.clicked.connect(self.toggle_play)
        self.slider_play.valueChanged.connect(self.on_slider_changed)

        # 3. 프레임 탐색
        nav_layout = QHBoxLayout()
        self.btn_ff_back = QPushButton("<<60")
        self.btn_prev = QPushButton("<")
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 999999)
        self.spin_frame.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.lbl_total_frames = QLabel("/ 0")
        self.btn_next = QPushButton(">")
        self.btn_ff_fwd = QPushButton("60>>")
        
        nav_layout.addWidget(self.btn_ff_back)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.spin_frame)
        nav_layout.addWidget(self.lbl_total_frames)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_ff_fwd)
        left_layout.addLayout(nav_layout)

        self.btn_ff_back.clicked.connect(lambda: self.seek_frame(-int(self.fps)))
        self.btn_prev.clicked.connect(lambda: self.seek_frame(-1))
        self.btn_next.clicked.connect(lambda: self.seek_frame(1))
        self.btn_ff_fwd.clicked.connect(lambda: self.seek_frame(int(self.fps)))
        self.spin_frame.valueChanged.connect(self.on_frame_spin_changed)

        # 4. 파일 탐색
        file_nav_layout = QHBoxLayout()
        self.btn_prev_folder = QPushButton("이전 폴더")
        self.btn_load_folder = QPushButton("데이터셋 불러오기")
        self.btn_next_folder = QPushButton("다음 폴더")
        file_nav_layout.addWidget(self.btn_prev_folder)
        file_nav_layout.addWidget(self.btn_load_folder)
        file_nav_layout.addWidget(self.btn_next_folder)
        left_layout.addLayout(file_nav_layout)

        self.btn_load_folder.clicked.connect(self.load_folder_dialog)
        self.btn_prev_folder.clicked.connect(lambda: self.navigate_folder(-1))
        self.btn_next_folder.clicked.connect(lambda: self.navigate_folder(1))

        splitter.addWidget(left_panel)

        # ==================== 중앙 패널 (설정 컨트롤) ====================
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_panel.setMaximumWidth(300)

        # 영상 정보
        group_info = QGroupBox("영상 정보")
        info_layout = QVBoxLayout()
        self.lbl_info_folder = QLabel("데이터셋: -")
        self.lbl_info_duration = QLabel("Record_Duration: -")
        self.lbl_info_format = QLabel("PixelFormat: -")
        self.lbl_info_fps = QLabel("FPS: -")
        info_layout.addWidget(self.lbl_info_folder)
        info_layout.addWidget(self.lbl_info_duration)
        info_layout.addWidget(self.lbl_info_format)
        info_layout.addWidget(self.lbl_info_fps)

        self.btn_hist_popup = QPushButton("RGB 히스토그램")
        self.btn_hist_popup.setEnabled(False)
        self.btn_hist_popup.clicked.connect(self.open_hist_popup)
        info_layout.addWidget(self.btn_hist_popup)

        group_info.setLayout(info_layout)
        center_layout.addWidget(group_info)

        # 분석 설정
        group_settings = QGroupBox("분석 설정")
        set_layout = QVBoxLayout()
        
        # ROI 개수
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI 개수:"))
        self.spin_roi_count = QSpinBox()
        self.spin_roi_count.setRange(1, 3)
        self.spin_roi_count.setValue(3)
        roi_layout.addWidget(self.spin_roi_count)
        set_layout.addLayout(roi_layout)

        # 배경 유무
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("배경 영역:"))
        self.spin_bg_count = QSpinBox()
        self.spin_bg_count.setRange(0, 1)
        self.spin_bg_count.setValue(1)
        bg_layout.addWidget(self.spin_bg_count)
        set_layout.addLayout(bg_layout)

        # 분석 시간
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("분석시간(초):"))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(1.0, 300.0)
        self.spin_duration.setValue(5.0)
        time_layout.addWidget(self.spin_duration)
        set_layout.addLayout(time_layout)

        # BPM 범위 입력
        bpm_layout = QHBoxLayout()
        bpm_layout.addWidget(QLabel("BPM 범위:"))
        self.spin_bpm_min = QSpinBox()
        self.spin_bpm_min.setRange(30, 900)
        self.spin_bpm_min.setValue(200)
        self.spin_bpm_min.setMinimumWidth(60)
        self.spin_bpm_max = QSpinBox()
        self.spin_bpm_max.setRange(30, 900)
        self.spin_bpm_max.setValue(600)
        self.spin_bpm_max.setMinimumWidth(60)
        bpm_layout.addWidget(self.spin_bpm_min)
        bpm_layout.addWidget(QLabel("~"))
        bpm_layout.addWidget(self.spin_bpm_max)
        set_layout.addLayout(bpm_layout)

        self.btn_setup = QPushButton("설정 완료")
        self.btn_setup.setCheckable(True)
        set_layout.addWidget(self.btn_setup)
        
        group_settings.setLayout(set_layout)
        center_layout.addWidget(group_settings)

        self.btn_setup.clicked.connect(self.toggle_setup)

        # 분석 실행
        self.btn_analyze = QPushButton("분석 시작")
        self.btn_analyze.setEnabled(False)
        center_layout.addWidget(self.btn_analyze)
        self.btn_analyze.clicked.connect(self.start_analysis)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        center_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("대기 중")
        center_layout.addWidget(self.lbl_status)

        # 채널 선택 (분석 완료 후 사용)
        group_channel = QGroupBox("결과 채널 선택")
        chan_layout = QHBoxLayout()
        self.radio_R = QRadioButton("R")
        self.radio_G = QRadioButton("G")
        self.radio_B = QRadioButton("B")
        self.radio_R.setChecked(True)
        self.bg_channel = QButtonGroup()
        self.bg_channel.addButton(self.radio_R, 0)
        self.bg_channel.addButton(self.radio_G, 1)
        self.bg_channel.addButton(self.radio_B, 2)
        chan_layout.addWidget(self.radio_R)
        chan_layout.addWidget(self.radio_G)
        chan_layout.addWidget(self.radio_B)
        group_channel.setLayout(chan_layout)
        center_layout.addWidget(group_channel)
        
        self.bg_channel.idToggled.connect(self.on_channel_changed)
        group_channel.setEnabled(False)
        self.group_channel = group_channel

        self.btn_graph_popup = QPushButton("상세 그래프 분석")
        self.btn_graph_popup.setEnabled(False)
        self.btn_graph_popup.clicked.connect(self.open_graph_popup)
        center_layout.addWidget(self.btn_graph_popup)

        # CSV 저장
        center_layout.addStretch()
        self.btn_save_csv = QPushButton("CSV로 저장")
        self.btn_save_csv.setEnabled(False)
        self.btn_save_csv.clicked.connect(self.save_csv)
        center_layout.addWidget(self.btn_save_csv)

        splitter.addWidget(center_panel)

        # ==================== 우측 패널 (그래프) ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.graph_layout)
        
        # 6개의 플롯 생성
        self.plots = []
        for i in range(6):
            p = self.graph_layout.addPlot()
            p.showGrid(x=True, y=True)
            self.plots.append(p)
            self.graph_layout.nextRow()

        # 플롯 타이틀 설정
        self.plots[0].setTitle("ROI 1 Raw")
        self.plots[1].setTitle("ROI 2 Raw")
        self.plots[2].setTitle("ROI 3 Raw")
        self.plots[3].setTitle("Filtered ROI Signals")
        self.plots[4].setTitle("PPG Raw Data")
        self.plots[5].setTitle("Background Raw")

        splitter.addWidget(right_panel)

        # Splitter 비율 설정 (영상 : 설정 : 그래프 = 4 : 2 : 4)
        splitter.setSizes([500, 250, 600])

    # -----------------------------------------------------------------
    # 파일 탐색 및 폴더 이동
    # -----------------------------------------------------------------
    def load_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택", self.folder_path)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.folder_path = os.path.normpath(folder)
        self.frames_dir = os.path.join(self.folder_path, "frames")
        if not os.path.exists(self.frames_dir):
            self.frames_dir = self.folder_path

        cam_csv = os.path.join(self.folder_path, "camera_timestamps.csv")
        if not os.path.exists(cam_csv):
            QMessageBox.warning(self, "경고", "camera_timestamps.csv 파일이 없습니다.")
            return

        # 메타데이터 로드
        self.load_camera_settings()

        self.df_cam = pd.read_csv(cam_csv)
        if self.df_cam.empty:
            QMessageBox.warning(self, "경고", "타임스탬프 데이터가 비어있습니다.")
            return

        folder_name = os.path.basename(self.folder_path)
        self.lbl_info_folder.setText(f"데이터셋: {folder_name}")
        self.lbl_info_duration.setText(f"Record_Duration: {getattr(self, 'record_duration', 0.0)}s")
        self.lbl_info_format.setText(f"PixelFormat: {self.pixel_format}")
        self.lbl_info_fps.setText(f"FPS: {self.fps:.2f}")

        # Update FF buttons based on FPS
        fps_int = int(self.fps) if self.fps > 0 else 30
        self.btn_ff_back.setText(f"<<{fps_int}")
        self.btn_ff_fwd.setText(f"{fps_int}>>")

        self.min_frame_idx = int(self.df_cam['Frame_Index'].min())
        self.max_frame_idx = int(self.df_cam['Frame_Index'].max())
        
        self.spin_frame.setRange(self.min_frame_idx, self.max_frame_idx)
        self.lbl_total_frames.setText(f"/ {self.max_frame_idx}")
        self.slider_play.setRange(self.min_frame_idx, self.max_frame_idx)
        self.slider_play.setEnabled(True)
        self.btn_play_pause.setEnabled(True)
        self.btn_hist_popup.setEnabled(True)
        
        # 강제로 첫 프레임 표시되도록 이벤트 발생 및 직접 호출
        self.spin_frame.setValue(self.min_frame_idx)
        self.show_frame(self.min_frame_idx)
        
        self.reset_state()
        self.lbl_status.setText("준비됨")

    def load_camera_settings(self):
        settings_path = os.path.join(self.folder_path, "camera_all_settings.txt")
        json_path = os.path.join(self.folder_path, "camera_summary.json")
        self.pixel_format = "BayerRG12"
        self.fps = 60.0
        self.record_duration = 0.0

        # json 파일이 있으면 우선 시도
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    self.pixel_format = summary.get("PixelFormat", self.pixel_format)
                    self.fps = summary.get("FPS_Result", summary.get("FPS_Target", self.fps))
                    self.record_duration = summary.get("Record_Duration_sec", self.record_duration)
                return
            except:
                pass
        
        # txt에서 파싱 (refer.py 로직)
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                start_marker = "=== [Summary] Main Camera Settings ==="
                end_marker = "=== [Full Dump] Camera All Settings ==="
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx + len(start_marker):end_idx].strip()
                    summary = json.loads(json_str)
                    self.pixel_format = summary.get("PixelFormat", self.pixel_format)
                    self.fps = summary.get("FPS_Result", summary.get("FPS_Target", self.fps))
                    self.record_duration = summary.get("Record_Duration_sec", self.record_duration)
            except:
                pass

    def open_hist_popup(self):
        if not self.frames_dir: return
        self.hist_popup = HistPopup(
            self,
            frames_dir=self.frames_dir,
            df_cam=self.df_cam,
            fps=self.fps,
            current_f_idx=self.current_frame_idx,
            min_f_idx=self.min_frame_idx,
            max_f_idx=self.max_frame_idx
        )
        self.hist_popup.show()

    def navigate_folder(self, direction):
        if not self.folder_path:
            return
        parent_dir = os.path.dirname(self.folder_path)
        try:
            subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                       if os.path.isdir(os.path.join(parent_dir, d))]
            subdirs.sort()
            idx = subdirs.index(self.folder_path)
            new_idx = idx + direction
            if 0 <= new_idx < len(subdirs):
                self.load_folder(subdirs[new_idx])
            else:
                QMessageBox.information(self, "안내", "이전/다음 폴더가 없습니다.")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"폴더 탐색 중 오류 발생:\n{str(e)}")

    # -----------------------------------------------------------------
    # 프레임 탐색 및 영상 표시
    # -----------------------------------------------------------------
    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play_pause.setText("재생")
        else:
            if self.current_frame_idx >= self.max_frame_idx:
                self.spin_frame.setValue(self.min_frame_idx)
            fps_val = self.fps if self.fps > 0 else 30
            self.play_timer.start(int(1000 / fps_val))
            self.btn_play_pause.setText("일시정지")

    def play_next_frame(self):
        if self.current_frame_idx < self.max_frame_idx:
            self.spin_frame.setValue(self.current_frame_idx + 1)
        else:
            self.toggle_play()

    def on_slider_changed(self, val):
        if self.spin_frame.value() != val:
            self.spin_frame.setValue(val)

    def seek_frame(self, offset):
        if self.df_cam is None: return
        new_val = self.spin_frame.value() + offset
        self.spin_frame.setValue(new_val)

    def on_frame_spin_changed(self, val):
        self.current_frame_idx = val
        if self.slider_play.value() != val:
            self.slider_play.blockSignals(True)
            self.slider_play.setValue(val)
            self.slider_play.blockSignals(False)
        self.show_frame(val)

    def show_frame(self, f_idx):
        if not self.frames_dir: return
        f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tiff")
        if not os.path.exists(f_path):
            f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tif")
            if not os.path.exists(f_path):
                return

        img = tifffile.imread(f_path) if f_path.lower().endswith(('.tiff', '.tif')) else cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        
        # pyqtgraph ImageView는 BGR 순서를 기대하므로 BayerRG2BGR 사용
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
        else:
            img_display = img  # BGR8 저장 파일은 이미 BGR, 그대로 사용

        # pyqtgraph ImageView expects axes (x, y, color) usually shape (W, H, 3)
        img_pg = np.transpose(img_display, (1, 0, 2))
        self.image_view.setImage(img_pg, autoRange=False)

        # ROI 히스토그램용 rgb 참조 별도 보관 (histogram은 cv2 RGB 순서 필요)
        if len(img.shape) == 2:
            self.current_img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        else:
            self.current_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------------------
    # ROI 설정
    # -----------------------------------------------------------------
    def toggle_setup(self):
        if not self.folder_path:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            self.btn_setup.setChecked(False)
            return

        if self.btn_setup.isChecked():
            # 설정 완료 클릭 됨 -> 박스 띄우기
            self.btn_setup.setText("재설정")
            self.btn_analyze.setEnabled(True)
            self.create_roi_boxes()
        else:
            # 재설정 클릭 됨 -> 박스 제거
            self.btn_setup.setText("설정 완료")
            self.btn_analyze.setEnabled(False)
            self.remove_roi_boxes()

    def create_roi_boxes(self):
        self.remove_roi_boxes()
        roi_count = self.spin_roi_count.value()
        has_bg = self.spin_bg_count.value() > 0

        colors = ['r', 'g', 'b'] # ROI 1, 2, 3
        # 중앙 기준으로 적당한 크기 배치
        default_size = 100
        x_start = 100
        y_start = 100

        for i in range(roi_count):
            if i < len(self.saved_roi_states):
                x, y, w, h = self.saved_roi_states[i]
            else:
                x, y, w, h = x_start + i*(default_size+20), y_start, default_size, default_size

            roi = pg.RectROI([x, y], [w, h], pen=pg.mkPen(colors[i], width=3))
            
            text = pg.TextItem(f"ROI {i+1}\n({int(w)}x{int(h)})", color=colors[i], anchor=(0, 1))
            text.setParentItem(roi)
            roi.sigRegionChanged.connect(lambda r, t=text, name=f"ROI {i+1}": t.setText(f"{name}\n({int(r.size()[0])}x{int(r.size()[1])})"))

            self.image_view.addItem(roi)
            self.roi_items.append(roi)

        if has_bg:
            if self.saved_bg_state:
                x, y, w, h = self.saved_bg_state
            else:
                x, y, w, h = x_start, y_start + default_size + 20, default_size, default_size

            self.bg_roi_item = pg.RectROI([x, y], [w, h], pen=pg.mkPen('m', width=3))
            
            text_bg = pg.TextItem(f"Background\n({int(w)}x{int(h)})", color='m', anchor=(0, 1))
            text_bg.setParentItem(self.bg_roi_item)
            self.bg_roi_item.sigRegionChanged.connect(lambda r, t=text_bg: t.setText(f"Background\n({int(r.size()[0])}x{int(r.size()[1])})"))

            self.image_view.addItem(self.bg_roi_item)

    def remove_roi_boxes(self):
        if self.roi_items:
            self.saved_roi_states = []
            for roi in self.roi_items:
                pos = roi.pos()
                size = roi.size()
                self.saved_roi_states.append((pos[0], pos[1], size[0], size[1]))
                self.image_view.removeItem(roi)
            self.roi_items.clear()
        
        if self.bg_roi_item:
            pos = self.bg_roi_item.pos()
            size = self.bg_roi_item.size()
            self.saved_bg_state = (pos[0], pos[1], size[0], size[1])
            self.image_view.removeItem(self.bg_roi_item)
            self.bg_roi_item = None

    def reset_state(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.btn_play_pause.setText("재생")
        
        self.remove_roi_boxes()
        self.btn_setup.setChecked(False)
        self.btn_setup.setText("설정 완료")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("분석 시작")
        self.progress_bar.setValue(0)
        self.lbl_status.setText("준비됨")
        self.analysis_result = None
        self.group_channel.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.btn_graph_popup.setEnabled(False)
        for p in self.plots:
            p.clear()

    # -----------------------------------------------------------------
    # 분석 시작
    # -----------------------------------------------------------------
    def start_analysis(self):
        if self.btn_analyze.text() == "분석중":
            return # 이미 실행중
        
        # ROI 정보 추출 [x, y, w, h]
        # pyqtgraph 좌표는 이미지의 transpose된 좌표이므로 (x, y) 그대로 사용하면 됨
        rois_info = []
        for roi in self.roi_items:
            pos = roi.pos()
            size = roi.size()
            rois_info.append((pos[0], pos[1], size[0], size[1]))
            
        bg_roi_info = None
        if self.bg_roi_item:
            pos = self.bg_roi_item.pos()
            size = self.bg_roi_item.size()
            bg_roi_info = (pos[0], pos[1], size[0], size[1])

        bpm_min = self.spin_bpm_min.value()
        bpm_max = self.spin_bpm_max.value()
        duration = self.spin_duration.value()

        self.btn_analyze.setText("분석중")
        self.btn_setup.setEnabled(False)
        self.group_channel.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.btn_graph_popup.setEnabled(False)

        # QThread 시작
        self.worker = AnalysisWorker(
            self.folder_path, self.current_frame_idx, duration,
            rois_info, bg_roi_info, bpm_min, bpm_max
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.start()

    @pyqtSlot(int, str)
    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    @pyqtSlot(dict)
    def analysis_finished(self, result):
        self.analysis_result = result
        self.btn_analyze.setText("재분석")
        self.btn_setup.setEnabled(True)
        self.group_channel.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_graph_popup.setEnabled(True)
        self.lbl_status.setText("분석 완료")
        
        # 최초 기본 채널(R) 그래프 표시
        self.update_graphs()

    @pyqtSlot(str)
    def analysis_error(self, err_msg):
        self.btn_analyze.setText("재분석")
        self.btn_setup.setEnabled(True)
        QMessageBox.critical(self, "오류", f"분석 중 오류 발생:\n{err_msg}")
        self.lbl_status.setText("분석 실패")

    # -----------------------------------------------------------------
    # 신호처리 및 그래프 업데이트
    # -----------------------------------------------------------------
    def open_graph_popup(self):
        if not self.analysis_result: return
        duration_val = int(self.spin_duration.value())
        popup = GraphPopup(
            self, 
            analysis_result=self.analysis_result, 
            folder_path=self.folder_path,
            spin_duration_val=duration_val,
            start_frame_idx=self.current_frame_idx
        )
        popup.exec()
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # check nyquist
        if high >= 1.0: high = 0.99
        if low <= 0: low = 0.01
        
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def on_channel_changed(self, btn_id, checked):
        if checked and self.analysis_result:
            self.update_graphs()

    def update_graphs(self):
        if not self.analysis_result: return

        # 선택된 채널 인덱스
        ch_idx = self.bg_channel.checkedId() # 0: R, 1: G, 2: B
        ch_name = ['R', 'G', 'B'][ch_idx]

        res = self.analysis_result
        t_cam = res['t_cam']
        t_cam_plot = t_cam - t_cam[0] if len(t_cam) > 0 else t_cam
        raw_rgb = res['raw_rgb_data'] # shape: [regions, frames, 3]
        roi_count = res['roi_count']
        has_bg = res['has_bg']
        has_sensor = res['has_sensor']
        
        fs_cam = res['fs_cam']
        bpm_min, bpm_max = res['bpm_min'], res['bpm_max']
        low_cut = bpm_min / 60.0
        high_cut = bpm_max / 60.0

        plot_colors = [(255,0,0), (0,255,0), (0,0,255)]

        # 기존 플롯들 초기화 및 레이아웃 정리
        self.graph_layout.clear()

        # X축 동기화 설정 (메인 윈도우)
        active_plots = []

        # 1~3: ROI Raw 그리기
        for i in range(roi_count):
            if i < 3:
                p = self.plots[i]
                p.clear()
                self.graph_layout.addItem(p)
                self.graph_layout.nextRow()
                y_raw = raw_rgb[i, :, ch_idx]
                p.plot(t_cam_plot, y_raw, pen=pg.mkPen(plot_colors[i], width=2))
                p.setTitle(f"ROI {i+1} Raw [{ch_name}]")
                p.enableAutoRange()
                active_plots.append(p)

        # 4: Filtered Signals (Overlap)
        p_filt = self.plots[3]
        p_filt.clear()
        self.graph_layout.addItem(p_filt)
        self.graph_layout.nextRow()
        p_filt.setTitle(f"Filtered Signals [{ch_name}]")
        p_filt.addLegend()
        
        for i in range(roi_count):
            y_raw = raw_rgb[i, :, ch_idx]
            y_c = y_raw - np.mean(y_raw)
            try:
                y_filt = self.butter_bandpass_filter(y_c, low_cut, high_cut, fs_cam)
                
                # Peak detection
                min_dist = 60.0 / bpm_max
                peaks, _ = find_peaks(y_filt, distance=int(min_dist * fs_cam), prominence=np.std(y_filt)*0.5)
                
                bpm = 60.0 / np.mean(np.diff(t_cam[peaks])) if len(peaks) > 1 else 0
                
                p_filt.plot(t_cam_plot, y_filt, pen=pg.mkPen(plot_colors[i], width=2), name=f"ROI {i+1} (BPM: {bpm:.1f})")
                
                # Peaks scatter
                if len(peaks) > 0:
                    scatter = pg.ScatterPlotItem(x=t_cam_plot[peaks], y=y_filt[peaks], pen=None, brush=pg.mkBrush(plot_colors[i]), size=8, symbol='x')
                    p_filt.addItem(scatter)
                    
            except Exception as e:
                print(f"Filter error ROI {i}:", e)
        p_filt.enableAutoRange()
        active_plots.append(p_filt)

        # 5: PPG Raw Data
        if has_sensor:
            p_ppg = self.plots[4]
            p_ppg.clear()
            self.graph_layout.addItem(p_ppg)
            self.graph_layout.nextRow()
            t_sensor = res['t_sensor']
            y_sensor = res['y_sensor']
            t_sensor_plot = t_sensor - t_cam[0] if len(t_sensor) > 0 else t_sensor
            p_ppg.plot(t_sensor_plot, y_sensor, pen=pg.mkPen(200,200,200, width=2))
            p_ppg.setTitle("PPG Raw Sensor Data")
            p_ppg.enableAutoRange()
            active_plots.append(p_ppg)

        # 6: Background Raw
        if has_bg:
            p_bg = self.plots[5]
            p_bg.clear()
            self.graph_layout.addItem(p_bg)
            self.graph_layout.nextRow()
            y_bg_raw = raw_rgb[roi_count, :, ch_idx]
            p_bg.plot(t_cam_plot, y_bg_raw, pen=pg.mkPen((200,0,200), width=2))
            p_bg.setTitle(f"Background Raw [{ch_name}]")
            p_bg.enableAutoRange()
            active_plots.append(p_bg)

        # X축 동기화 연결
        for i, p in enumerate(active_plots):
            if i > 0:
                p.setXLink(active_plots[0])
            else:
                p.setXLink(None)

    # -----------------------------------------------------------------
    # CSV 저장
    # -----------------------------------------------------------------
    def save_csv(self):
        if not self.analysis_result: return
        
        folder_name = os.path.basename(self.folder_path)
        duration_val = int(self.spin_duration.value())
        start_f = self.current_frame_idx
        default_filename = f"{folder_name}_{start_f}_{duration_val}s.csv"
        
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        default_filepath = os.path.join(output_dir, default_filename)
        
        save_path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", default_filepath, "CSV Files (*.csv)")
        if not save_path: return

        res = self.analysis_result
        frame_indices = res['frame_indices']
        raw_rgb = res['raw_rgb_data']
        roi_count = res['roi_count']
        has_bg = res['has_bg']
        has_sensor = res['has_sensor']
        
        # PPG 동기화 (가장 가까운 시간의 PPG 값 매핑)
        t_cam = res['t_cam']
        t_sensor = res['t_sensor']
        y_sensor = res['y_sensor']
        
        mapped_ppg = []
        if has_sensor and len(t_sensor) > 0:
            for tc in t_cam:
                idx = (np.abs(t_sensor - tc)).argmin()
                mapped_ppg.append(y_sensor[idx])
        else:
            mapped_ppg = [np.nan] * len(t_cam)

        fps = res['fs_cam']

        rows = []
        for i, f_idx in enumerate(frame_indices):
            save_f_idx = i + 1
            # ROI 1~3
            for r in range(roi_count):
                rows.append({
                    "Frame": save_f_idx,
                    "Type": f"roi{r+1}",
                    "R": raw_rgb[r, i, 0],
                    "G": raw_rgb[r, i, 1],
                    "B": raw_rgb[r, i, 2],
                    "PPG": mapped_ppg[i],
                    "FPS": fps
                })
            # Background
            if has_bg:
                bg_idx = roi_count
                rows.append({
                    "Frame": save_f_idx,
                    "Type": "background",
                    "R": raw_rgb[bg_idx, i, 0],
                    "G": raw_rgb[bg_idx, i, 1],
                    "B": raw_rgb[bg_idx, i, 2],
                    "PPG": mapped_ppg[i],
                    "FPS": fps
                })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        QMessageBox.information(self, "완료", f"데이터가 저장되었습니다.\n{save_path}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
