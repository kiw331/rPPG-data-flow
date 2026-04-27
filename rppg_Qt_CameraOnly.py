# 파일명: rPPG_Qt_CameraOnly.py
# 지시사항: 
# 1. 하드웨어 PPG 센서 관련 로직, UI, 저장 기능 모두 제거 (오직 카메라 단독 실행)
# 2. ROI 1(빨간 박스)과 ROI 2(초록 박스) 2개의 영역을 실시간으로 비교 모니터링 가능하도록 수정
# 3. camera_all_settings.txt에 JSON 요약 저장, 동적 폴더 생성, 조기 종료 등 기존 편리한 기능 유지
# 4. JSON의 Program_Version은 "CameraOnly"로 기록

import sys
import os
import cv2
import time
import csv
import json
import numpy as np
import tifffile
import gc
import queue
import threading
from datetime import datetime
from collections import deque
from pypylon import pylon

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                             QSizePolicy, QMessageBox, QGroupBox, QComboBox, 
                             QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg

# ================= [기본 설정] =================
BASE_OUTPUT_DIR = r"data\basler_0225"  # 동적 폴더 생성에 활용될 기본 경로
# ===============================================

from modules.storage import writer_worker
from modules.camera import BaseCameraThread

# 📷 [카메라 스레드]
class CameraThread(BaseCameraThread):
    def __init__(self, gui_q):
        super().__init__(gui_q=gui_q, user_set="UserSet1", program_version="CameraOnly")
        # 💡 [수정] ROI 1은 위쪽(Y=350), ROI 2는 아래쪽(Y=450)으로 배치 (크기 80 기준, 20픽셀 간격)
        self.roi1_config = (472, 350, 80)
        self.roi2_config = (472, 450, 80)
        
    def process_frame(self, raw_data, capture_time):
        h, w = raw_data.shape[:2]
        
        # 💡 [변경] 두 ROI 영역 안전성 연산 및 추출
        r1x, r1y, r1l = self.roi1_config
        r2x, r2y, r2l = self.roi2_config
        
        r1x, r1y, r1l = r1x & ~1, r1y & ~1, max(2, r1l & ~1)
        r2x, r2y, r2l = r2x & ~1, r2y & ~1, max(2, r2l & ~1)
        
        r1x, r1y = max(0, min(w - r1l, r1x)), max(0, min(h - r1l, r1y))
        r2x, r2y = max(0, min(w - r2l, r2x)), max(0, min(h - r2l, r2y))
        
        roi1_raw = raw_data[r1y:r1y+r1l, r1x:r1x+r1l]
        roi2_raw = raw_data[r2y:r2y+r2l, r2x:r2x+r2l]
        
        import cv2
        if self.current_format == "BGR8":
            roi1_rgb = cv2.cvtColor(roi1_raw, cv2.COLOR_BGR2RGB)
            roi2_rgb = cv2.cvtColor(roi2_raw, cv2.COLOR_BGR2RGB)
        else:
            roi1_rgb = cv2.cvtColor(roi1_raw, cv2.COLOR_BayerBG2RGB)
            roi2_rgb = cv2.cvtColor(roi2_raw, cv2.COLOR_BayerBG2RGB)
        
        roi1_mean = cv2.mean(roi1_rgb)[:3]
        roi2_mean = cv2.mean(roi2_rgb)[:3]
        
        self.gui_q.append((capture_time, roi1_mean, roi2_mean))
        
        if not self.is_recording or self.frame_count % 2 == 0: 
            view_8bit = self.render_base_pixmap(raw_data)
            
            r1x_h, r1y_h, r1l_h = r1x // 2, r1y // 2, r1l // 2
            r2x_h, r2y_h, r2l_h = r2x // 2, r2y // 2, r2l // 2
            
            # ROI 1 = 빨간색, ROI 2 = 초록색 박스
            cv2.rectangle(view_8bit, (r1x_h, r1y_h), (r1x_h+r1l_h, r1y_h+r1l_h), (255, 0, 0), 2)
            cv2.rectangle(view_8bit, (r2x_h, r2y_h), (r2x_h+r2l_h, r2y_h+r2l_h), (0, 255, 0), 2)
            
            self.change_pixmap_signal.emit(view_8bit)

    def start_recording(self, save_dir):
        self.start_recording_with_worker(save_dir, writer_worker)
        
    def stop_recording(self):
        self.stop_recording_and_save_settings()

# [UI 클래스] 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG_Qt_CameraOnly (Dual ROI Monitoring)")
        self.resize(1350, 720) 
        
        self.load_roi_settings()

        self.recording_duration = 60 
        self.remaining_time = 0
        self.current_save_dir = "" 
        self.is_currently_recording = False
        
        self.cam_x_range = 120 
        self.display_window_sec = 2.0 

        MAX_LEN_CAM = 60 * 20
        self.cam_q = deque(maxlen=MAX_LEN_CAM)

        self.init_ui()
        
        self.video_thread = CameraThread(self.cam_q)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.queue_status_signal.connect(self.update_queue_status)
        self.video_thread.recording_finished_signal.connect(self.on_stop_signal)
        self.video_thread.initial_settings_signal.connect(self.sync_ui_with_camera) 
        
        self.update_areas() # UI에 로드된 ROI 설정값을 카메라 스레드에 동기화
        
        self.video_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.save_check_timer = QTimer()
        self.save_check_timer.timeout.connect(self.check_save_complete_console)
        
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.redraw_graphs)
        self.graph_timer.start(33) 

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # =========================================================
        # ⬅️ [좌측 패널] 비디오 및 컨트롤 영역
        # =========================================================
        video_group = QGroupBox("Camera Feed")
        video_inner = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 250)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        video_inner.setContentsMargins(5, 5, 5, 5)
        video_inner.addWidget(self.video_label)
        video_group.setLayout(video_inner)
        left_layout.addWidget(video_group, stretch=1) 

        cam_ctrl_group = QGroupBox("Camera Hardware Settings (Override)")
        cam_ctrl_layout = QVBoxLayout()
        
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Resolution & Offset:"))
        self.combo_cam_res = QComboBox()
        self.combo_cam_res.addItems([
            "UserSet Default 유지",
            "옵션 1: 1024x1024 (Offset 208, 28)",
            "옵션 2: 1440x864 (Offset 0, 0)"
        ])
        row_res.addWidget(self.combo_cam_res)
        
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Pixel Format:"))
        self.combo_cam_format = QComboBox()
        self.combo_cam_format.addItems(["Bayer RG 12", "Bayer RG 8", "BGR 8"])
        row_fmt.addWidget(self.combo_cam_format)
        
        row_fps = QHBoxLayout()
        row_fps.addWidget(QLabel("FPS:"))
        self.spin_cam_fps = QDoubleSpinBox()
        self.spin_cam_fps.setDecimals(0) 
        self.spin_cam_fps.setRange(1.0, 200.0)
        self.spin_cam_fps.setValue(60.0)
        row_fps.addWidget(self.spin_cam_fps)

        row_exp = QHBoxLayout()
        row_exp.addWidget(QLabel("Exposure (us):"))
        self.spin_cam_exp = QSpinBox()
        self.spin_cam_exp.setRange(100, 200000)
        self.spin_cam_exp.setSingleStep(1000)
        self.spin_cam_exp.setValue(10000)
        row_exp.addWidget(self.spin_cam_exp)

        row_col = QHBoxLayout()
        row_col.addWidget(QLabel("Color Temp:"))
        self.combo_cam_color = QComboBox()
        self.combo_cam_color.addItems(["Daylight5000K", "Daylight6500K", "Tungsten2800K", "Off"])
        row_col.addWidget(self.combo_cam_color)

        self.btn_apply_cam = QPushButton("⚙️ 카메라에 설정 덮어쓰기")
        self.btn_apply_cam.clicked.connect(self.apply_camera_settings)
        self.btn_apply_cam.setStyleSheet("background-color: #eab308; font-weight: bold;")

        cam_ctrl_layout.addLayout(row_res)
        cam_ctrl_layout.addLayout(row_fmt)
        cam_ctrl_layout.addLayout(row_fps)
        cam_ctrl_layout.addLayout(row_exp)
        cam_ctrl_layout.addLayout(row_col)
        cam_ctrl_layout.addWidget(self.btn_apply_cam)
        cam_ctrl_group.setLayout(cam_ctrl_layout)
        left_layout.addWidget(cam_ctrl_group)

        # 💡 두 개의 ROI 영역 컨트롤 패널
        area_group = QGroupBox("Dual ROI Config (X, Y, Length)")
        area_layout = QVBoxLayout()
        
        roi1_layout = QHBoxLayout()
        roi1_layout.addWidget(QLabel("🟥 ROI 1:"))
        self.spin_roi1_x = QSpinBox(); self.spin_roi1_x.setRange(0, 4000); self.spin_roi1_x.setValue(self.roi1_x) 
        self.spin_roi1_y = QSpinBox(); self.spin_roi1_y.setRange(0, 4000); self.spin_roi1_y.setValue(self.roi1_y) 
        self.spin_roi1_l = QSpinBox(); self.spin_roi1_l.setRange(2, 1000); self.spin_roi1_l.setValue(self.roi1_l)
        roi1_layout.addWidget(self.spin_roi1_x); roi1_layout.addWidget(self.spin_roi1_y); roi1_layout.addWidget(self.spin_roi1_l)
        
        roi2_layout = QHBoxLayout()
        roi2_layout.addWidget(QLabel("🟩 ROI 2:"))
        self.spin_roi2_x = QSpinBox(); self.spin_roi2_x.setRange(0, 4000); self.spin_roi2_x.setValue(self.roi2_x) 
        self.spin_roi2_y = QSpinBox(); self.spin_roi2_y.setRange(0, 4000); self.spin_roi2_y.setValue(self.roi2_y) 
        self.spin_roi2_l = QSpinBox(); self.spin_roi2_l.setRange(2, 1000); self.spin_roi2_l.setValue(self.roi2_l)
        roi2_layout.addWidget(self.spin_roi2_x); roi2_layout.addWidget(self.spin_roi2_y); roi2_layout.addWidget(self.spin_roi2_l)

        area_layout.addLayout(roi1_layout)
        area_layout.addLayout(roi2_layout)
        area_group.setLayout(area_layout)
        left_layout.addWidget(area_group)
        
        self.spin_roi1_x.valueChanged.connect(self.update_areas)
        self.spin_roi1_y.valueChanged.connect(self.update_areas)
        self.spin_roi1_l.valueChanged.connect(self.update_areas)
        self.spin_roi2_x.valueChanged.connect(self.update_areas)
        self.spin_roi2_y.valueChanged.connect(self.update_areas)
        self.spin_roi2_l.valueChanged.connect(self.update_areas)
        self.update_areas() # ROI 초기 설정 동기화

        rec_group = QGroupBox("Recording Controls")
        rec_layout = QVBoxLayout()
        row_duration = QHBoxLayout()
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 3600)
        self.spin_duration.setValue(60)
        row_duration.addWidget(QLabel("녹화 시간(초):"))
        row_duration.addWidget(self.spin_duration)
        rec_layout.addLayout(row_duration)

        self.lbl_timer = QLabel("Ready")
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        self.lbl_timer.setStyleSheet("background-color: #333; color: #4ade80; font-size: 18px; font-weight: bold; padding: 5px; border-radius: 5px;")
        rec_layout.addWidget(self.lbl_timer)
        
        self.lbl_queue = QLabel("Buffer: 0 / 3000")
        self.lbl_queue.setAlignment(Qt.AlignCenter)
        self.lbl_queue.setStyleSheet("background-color: #333; color: white; padding: 5px; border-radius: 5px;")
        rec_layout.addWidget(self.lbl_queue)

        self.btn_start = QPushButton("🔴 녹화 시작")
        self.btn_start.setFixedHeight(40) 
        self.btn_start.clicked.connect(self.toggle_recording)
        self.btn_start.setStyleSheet("background-color: #f43f5e; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        rec_layout.addWidget(self.btn_start)
        
        rec_group.setLayout(rec_layout)
        left_layout.addWidget(rec_group)

        # =========================================================
        # ➡️ [우측 패널] 2개의 실시간 그래프 및 디스플레이 옵션
        # =========================================================
        graphs_group = QGroupBox("Real-time Camera ROI Signals")
        graphs_layout = QVBoxLayout()
        
        # ROI 1 그래프 (빨간색 펜)
        self.plot_roi1 = pg.PlotWidget(title="Camera ROI 1 (Raw)")
        self.plot_roi1.setBackground('k')
        self.plot_roi1.setMinimumHeight(200) 
        self.curve_roi1 = self.plot_roi1.plot(pen=pg.mkPen(color='#ef4444', width=2))
        
        # ROI 2 그래프 (초록색 펜)
        self.plot_roi2 = pg.PlotWidget(title="Camera ROI 2 (Raw)")
        self.plot_roi2.setBackground('k')
        self.plot_roi2.setMinimumHeight(200)
        self.curve_roi2 = self.plot_roi2.plot(pen=pg.mkPen(color='#22c55e', width=2)) 
        
        graphs_layout.addWidget(self.plot_roi1)
        graphs_layout.addWidget(self.plot_roi2)
        graphs_group.setLayout(graphs_layout)
        right_layout.addWidget(graphs_group)

        bottom_options_group = QGroupBox("Additional Options")
        bottom_options_layout = QHBoxLayout()
        
        bottom_options_layout.addWidget(QLabel("X-Axis (Frames):"))
        self.combo_x_range = QComboBox()
        self.combo_x_range.addItems(["120", "240", "360", "600", "1200"])
        self.combo_x_range.setCurrentText("240")
        self.combo_x_range.currentTextChanged.connect(self.update_x_range)
        bottom_options_layout.addWidget(self.combo_x_range)

        bottom_options_layout.addWidget(QLabel(" | Channel:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(["R", "G (Def)", "B"])
        self.combo_channel.setCurrentIndex(1) 
        self.combo_channel.currentTextChanged.connect(self.clear_cam_buffers)
        bottom_options_layout.addWidget(self.combo_channel)
        
        self.check_invert = QCheckBox("Invert(-G)")
        bottom_options_layout.addWidget(self.check_invert)

        bottom_options_group.setLayout(bottom_options_layout)
        right_layout.addWidget(bottom_options_group)
        
        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(right_layout, stretch=5)

    def sync_ui_with_camera(self, settings):
        self.spin_cam_fps.blockSignals(True)
        self.spin_cam_exp.blockSignals(True)
        self.combo_cam_format.blockSignals(True)
        self.combo_cam_color.blockSignals(True)

        if 'fps' in settings and settings['fps'] is not None: 
            rounded_fps = int(round(float(settings['fps'])))
            self.spin_cam_fps.setValue(rounded_fps)
        if 'exp' in settings and settings['exp'] is not None: 
            self.spin_cam_exp.setValue(int(settings['exp']))
            
        if 'format' in settings and settings['format'] is not None:
            fmt_map = {"BayerRG12": "Bayer RG 12", "BayerRG8": "Bayer RG 8", "BGR8": "BGR 8"}
            if settings['format'] in fmt_map:
                self.combo_cam_format.setCurrentText(fmt_map[settings['format']])
                
        if 'color' in settings and settings['color'] is not None:
            idx = self.combo_cam_color.findText(settings['color'])
            if idx >= 0: self.combo_cam_color.setCurrentIndex(idx)

        self.spin_cam_fps.blockSignals(False)
        self.spin_cam_exp.blockSignals(False)
        self.combo_cam_format.blockSignals(False)
        self.combo_cam_color.blockSignals(False)

    def apply_camera_settings(self):
        if hasattr(self, 'video_thread') and self.video_thread.is_running:
            format_map = {
                "Bayer RG 12": "BayerRG12",
                "Bayer RG 8": "BayerRG8",
                "BGR 8": "BGR8"
            }
            selected_fmt = format_map[self.combo_cam_format.currentText()]
            
            res_text = self.combo_cam_res.currentText()
            res_dict = None
            if "옵션 1" in res_text:
                res_dict = {'w': 1024, 'h': 1024, 'ox': 208, 'oy': 28}
            elif "옵션 2" in res_text:
                res_dict = {'w': 1440, 'h': 864, 'ox': 0, 'oy': 0}
            
            cmd = {
                'format': selected_fmt,
                'fps': self.spin_cam_fps.value(),
                'exp': self.spin_cam_exp.value(),
                'color': self.combo_cam_color.currentText()
            }
            if res_dict is not None:
                cmd['resolution'] = res_dict
            self.video_thread.cmd_queue.put(cmd)

    def update_areas(self):
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            self.video_thread.roi1_config = (self.spin_roi1_x.value(), self.spin_roi1_y.value(), self.spin_roi1_l.value())
            self.video_thread.roi2_config = (self.spin_roi2_x.value(), self.spin_roi2_y.value(), self.spin_roi2_l.value())

    def update_x_range(self, text):
        try:
            cam_len = int(text)
            self.cam_x_range = cam_len
            self.display_window_sec = cam_len / 60.0 # 대략 60fps 기준으로 환산
        except ValueError: pass

    def clear_cam_buffers(self):
        self.cam_q.clear()

    def redraw_graphs(self):
        cam_list = list(self.cam_q)
        
        if len(cam_list) > 10:
            latest_t = cam_list[-1][0]
            c_idx = max(0, len(cam_list) - self.cam_x_range)
            
            t_cam = np.array([item[0] - latest_t for item in cam_list[c_idx:]])
            
            ch_idx = self.combo_channel.currentIndex()
            y_roi1 = np.array([item[1][ch_idx] for item in cam_list[c_idx:]])
            y_roi2 = np.array([item[2][ch_idx] for item in cam_list[c_idx:]])
            
            if self.check_invert.isChecked():
                y_roi1 = -y_roi1
                y_roi2 = -y_roi2

            self.plot_roi1.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot_roi2.enableAutoRange(axis=pg.ViewBox.YAxis)

            self.curve_roi1.setData(x=t_cam, y=y_roi1)
            self.curve_roi2.setData(x=t_cam, y=y_roi2)
            
            self.plot_roi1.setXRange(-self.display_window_sec, 0, padding=0)
            self.plot_roi2.setXRange(-self.display_window_sec, 0, padding=0)

    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        qt_img = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
        label_size = self.video_label.size()
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_queue_status(self, qsize):
        self.lbl_queue.setText(f"Buffer: {qsize} / 3000")
        if qsize < 500: self.lbl_queue.setStyleSheet("background-color: #333; color: #4ade80;")
        else: self.lbl_queue.setStyleSheet("background-color: #333; color: #ef4444;")

    def toggle_recording(self):
        if not self.is_currently_recording:
            self.start_recording()
        else:
            self.stop_recording_phase()

    def start_recording(self):
        today_str = datetime.now().strftime("%m%d")
        base_dir = os.path.join("data", f"basler_{today_str}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_save_dir = os.path.join(base_dir, timestamp)
        self.frames_save_dir = os.path.join(self.current_save_dir, "frames")
        if not os.path.exists(self.frames_save_dir): os.makedirs(self.frames_save_dir)
        
        self.recording_duration = self.spin_duration.value()
        self.remaining_time = self.recording_duration
        
        self.is_currently_recording = True
        self.btn_start.setText("⏹ 녹화 조기 종료")
        self.btn_start.setStyleSheet("background-color: #64748b; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        self.spin_duration.setEnabled(False)
        self.lbl_timer.setText(f"REC: {self.remaining_time}s")
        
        self.video_thread.start_recording(self.frames_save_dir) 
        self.timer.start(1000)

    def update_timer(self):
        self.remaining_time -= 1
        self.lbl_timer.setText(f"REC: {self.remaining_time}s")
        if self.remaining_time <= 0: self.stop_recording_phase()

    def stop_recording_phase(self):
        self.timer.stop()
        self.is_currently_recording = False
        self.video_thread.stop_recording()
        self.lbl_timer.setText("Ready")
        
        self.btn_start.setText("🔴 녹화 시작")
        self.btn_start.setStyleSheet("background-color: #f43f5e; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        self.spin_duration.setEnabled(True)

    def on_stop_signal(self, save_path):
        self.save_check_timer.start(500)

    def check_save_complete_console(self):
        q = self.video_thread.img_queue
        if (q is not None and q.empty()) or (not self.video_thread.writer_thread.is_alive()):
            self.save_check_timer.stop()
            print("\n✅ 저장 완료.")
            self.validate_recording()
            self.video_thread.is_running = False
            self.video_thread.wait(1000)
            self.close()

    def validate_recording(self):
        try:
            summary_path = os.path.join(self.current_save_dir, "camera_summary.json")
            if not os.path.exists(summary_path): return
            
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            duration = summary.get("Record_Duration_sec", 0)
            target_fps = summary.get("FPS_Target", 60)
            if duration <= 0: return
            
            expected_frames = int(duration * target_fps)
            csv_path = os.path.join(self.current_save_dir, "camera_timestamps.csv")
            actual_frames = 0
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    actual_frames = max(0, sum(1 for _ in f) - 1)
            
            cam_diff = abs(actual_frames - expected_frames) / max(1, expected_frames)
            msg_lines = [f"▶ 예상 카메라 프레임: {expected_frames} | 실제 저장: {actual_frames} (차이: {cam_diff*100:.1f}%)"]
            
            warning_triggered = False
            if cam_diff >= 0.02:
                warning_triggered = True
                msg_lines.append("  ⚠️ [경고] 카메라 데이터 기록 수가 기대값과 2% 이상 차이가 납니다.")
                
            if hasattr(self, 'sensor_thread'):
                expected_sensor = int(duration * 200)
                sensor_csv = os.path.join(self.current_save_dir, "ppg_sensor.csv")
                actual_sensor = 0
                if os.path.exists(sensor_csv):
                    with open(sensor_csv, 'r', encoding='utf-8') as f:
                        actual_sensor = max(0, sum(1 for _ in f) - 1)
                
                sen_diff = abs(actual_sensor - expected_sensor) / max(1, expected_sensor)
                msg_lines.append(f"▶ 예상 센서 데이터 수: {expected_sensor} | 실제 저장: {actual_sensor} (차이: {sen_diff*100:.1f}%)")
                if sen_diff >= 0.02:
                    warning_triggered = True
                    msg_lines.append("  ⚠️ [경고] 센서 데이터 기록 수가 기대값과 2% 이상 차이가 납니다.")
                    
            print("\n=== 데이터 저장 결과 검증 ===")
            print("\n".join(msg_lines))
            print("===========================\n")
            
            if warning_triggered:
                QMessageBox.warning(self, "저장 데이터 검증 경고", "\n".join(msg_lines))
                
        except Exception as e:
            print(f"검증 중 오류 발생: {e}") 

    def load_roi_settings(self):
        self.roi1_x, self.roi1_y, self.roi1_l = 472, 350, 80
        self.roi2_x, self.roi2_y, self.roi2_l = 472, 450, 80
        setting_path = os.path.join("setting", "roi_CameraOnly.json")
        if os.path.exists(setting_path):
            try:
                with open(setting_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.roi1_x = data.get("roi1_x", 472)
                    self.roi1_y = data.get("roi1_y", 350)
                    self.roi1_l = data.get("roi1_l", 80)
                    self.roi2_x = data.get("roi2_x", 472)
                    self.roi2_y = data.get("roi2_y", 450)
                    self.roi2_l = data.get("roi2_l", 80)
            except Exception as e:
                print(f"Failed to load ROI settings: {e}")

    def save_roi_settings(self):
        os.makedirs("setting", exist_ok=True)
        setting_path = os.path.join("setting", "roi_CameraOnly.json")
        data = {
            "roi1_x": self.spin_roi1_x.value(),
            "roi1_y": self.spin_roi1_y.value(),
            "roi1_l": self.spin_roi1_l.value(),
            "roi2_x": self.spin_roi2_x.value(),
            "roi2_y": self.spin_roi2_y.value(),
            "roi2_l": self.spin_roi2_l.value()
        }
        try:
            with open(setting_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save ROI settings: {e}")

    def closeEvent(self, event):
        self.save_roi_settings()
        self.video_thread.is_running = False
        self.video_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())