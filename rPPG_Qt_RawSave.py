# 파일명: rPPG_Qt_RawSave.py
# 지시사항: 
# 1. 단일 .raw 바이너리 고속 저장 (프레임 드랍 원천 차단)
# 2. Basic 버전과 동일한 좌측 비디오 렌더링 최적화 UI 적용
# 3. 해상도/오프셋 등 카메라 수동 제어 및 JSON 요약본 저장 적용
# 4. JSON 요약본에 "Program_Version": "SingleRaw" 명시

import sys
import os
import cv2
import serial
import serial.tools.list_ports
import time
import csv
import json
import numpy as np
import gc
import queue
import threading
import struct
from datetime import datetime
from collections import deque
from pypylon import pylon

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                             QSizePolicy, QMessageBox, QGroupBox, QComboBox, 
                             QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg


from modules.sensor import SensorThread
from modules.storage import raw_writer_worker
from modules.camera import BaseCameraThread

# 📷 [카메라 스레드]
class CameraThread(BaseCameraThread):
    def __init__(self):
        super().__init__(gui_q=None, user_set="UserSet1", program_version="SingleRaw")
        
    def process_frame(self, raw_data, capture_time):
        if not self.is_recording or self.frame_count % 2 == 0: 
            view_8bit = self.render_base_pixmap(raw_data)
            self.change_pixmap_signal.emit(view_8bit)

    def start_recording(self, save_dir):
        self.start_recording_with_worker(save_dir, raw_writer_worker)
        
    def stop_recording(self):
        # 💡 Raw 버전은 frames 폴더가 없으므로 self.save_dir에 바로 설정 저장
        self.stop_recording_and_save_settings(settings_parent_dir=self.save_dir)

# [UI 클래스] 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG_Qt_RawSave (Single .raw Binary)")
        self.resize(1350, 720) 
        
        self.load_sensor_settings()
        self.active_channel = self.default_graph  # 'IR' 또는 'RED'

        self.recording_duration = 60
        self.remaining_time = 0
        self.current_save_dir = "" 
        self.is_currently_recording = False
        
        self.display_window_sec = 2.0 
        self.sensor_x_range = 400

        MAX_LEN_SEN = 200 * 10
        self.sensor_q = deque(maxlen=MAX_LEN_SEN)

        self.init_ui()
        
        self.video_thread = CameraThread()
        self.sensor_thread = SensorThread(self.sensor_q)

        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.queue_status_signal.connect(self.update_queue_status)
        self.video_thread.recording_finished_signal.connect(self.on_stop_signal)
        self.video_thread.initial_settings_signal.connect(self.sync_ui_with_camera) 
        
        self.sensor_thread.update_stats_signal.connect(self.update_sensor_stats) 
        self.sensor_thread.finished_signal.connect(self.save_csv_data)
        self.sensor_thread.connection_status_signal.connect(self.on_sensor_connected)

        self.video_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.save_check_timer = QTimer()
        self.save_check_timer.timeout.connect(self.check_save_complete_console)
        
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.redraw_graphs)
        self.graph_timer.start(33) 
        
        self.refresh_ports()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # =========================================================
        # ⬅️ [좌측 패널] 비디오 영역 단독 배치 (크기 극대화)
        # =========================================================
        video_group = QGroupBox("Camera Feed")
        video_inner = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 400)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        video_inner.setContentsMargins(5, 5, 5, 5)
        video_inner.addWidget(self.video_label)
        video_group.setLayout(video_inner)
        
        left_layout.addWidget(video_group) 

        # =========================================================
        # ➡️ [우측 패널] 그래프 1개 + 모든 컨트롤 모음
        # =========================================================
        
        # 1. 하드웨어 PPG 그래프 (IR / RED 전환 가능)
        graphs_group = QGroupBox("Real-time Signals (Hardware Sensor)")
        graphs_layout = QVBoxLayout()

        ch_row = QHBoxLayout()
        self.btn_ch_ir = QPushButton("IR 전환")
        self.btn_ch_ir.clicked.connect(lambda: self.set_channel('IR'))
        self.btn_ch_red = QPushButton("RED 전환")
        self.btn_ch_red.clicked.connect(lambda: self.set_channel('RED'))
        self.lbl_channel = QLabel("표시 중: IR")
        self.lbl_channel.setStyleSheet("font-weight: bold;")
        ch_row.addWidget(self.btn_ch_ir)
        ch_row.addWidget(self.btn_ch_red)
        ch_row.addWidget(self.lbl_channel)
        ch_row.addStretch()
        graphs_layout.addLayout(ch_row)

        self.plot_sensor = pg.PlotWidget()
        self.plot_sensor.setBackground('k')
        self.plot_sensor.setMinimumHeight(200)
        self.curve_sensor = self.plot_sensor.plot(pen=pg.mkPen(color='#a78bfa', width=2))
        graphs_layout.addWidget(self.plot_sensor)
        graphs_group.setLayout(graphs_layout)
        right_layout.addWidget(graphs_group)

        # 2. 연결 및 상태 모니터링 (수평 배치)
        conn_group = QGroupBox("Sensor Connection")
        conn_layout = QHBoxLayout()
        self.combo_ports = QComboBox()
        self.btn_refresh = QPushButton("🔄 새로고침")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect = QPushButton("🔌 연결")
        self.btn_connect.clicked.connect(self.connect_sensor)
        self.btn_connect.setStyleSheet("background-color: #333; color: white; font-weight: bold;")
        conn_layout.addWidget(self.combo_ports)
        conn_layout.addWidget(self.btn_refresh)
        conn_layout.addWidget(self.btn_connect)
        conn_group.setLayout(conn_layout)

        monitor_group = QGroupBox("Data Quality")
        monitor_layout = QVBoxLayout()
        self.lbl_hz = QLabel("Sampling Rate: 0 Hz")
        self.lbl_drop = QLabel("Packet Drops: 0")
        monitor_layout.addWidget(self.lbl_hz)
        monitor_layout.addWidget(self.lbl_drop)
        monitor_group.setLayout(monitor_layout)

        row_conn_monitor = QHBoxLayout()
        row_conn_monitor.addWidget(conn_group)
        row_conn_monitor.addWidget(monitor_group)
        right_layout.addLayout(row_conn_monitor)

        # 3. 카메라 세팅 제어
        cam_ctrl_group = QGroupBox("Camera Hardware Settings")
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
        right_layout.addWidget(cam_ctrl_group)

        # 4. 기타 옵션 (그래프 X축)
        bottom_options_group = QGroupBox("Additional Options")
        bottom_options_layout = QHBoxLayout()
        bottom_options_layout.addWidget(QLabel("X-Axis View:"))
        self.combo_x_range = QComboBox()
        self.combo_x_range.addItems(["200", "300", "400", "800", "1200"])
        self.combo_x_range.setCurrentText("400")
        self.combo_x_range.currentTextChanged.connect(self.update_x_range)
        bottom_options_layout.addWidget(self.combo_x_range)
        bottom_options_layout.addStretch()
        bottom_options_group.setLayout(bottom_options_layout)
        right_layout.addWidget(bottom_options_group)

        # 4-1. LED 밝기 제어 (IR / RED)
        led_group = QGroupBox("LED Brightness")
        led_layout = QHBoxLayout()
        led_layout.addWidget(QLabel("IR:"))
        self.spin_ir_brightness = QSpinBox()
        self.spin_ir_brightness.setRange(0, 255)
        self.spin_ir_brightness.setValue(self.ir_brightness)
        led_layout.addWidget(self.spin_ir_brightness)
        self.btn_set_ir = QPushButton("💡 IR 적용")
        self.btn_set_ir.clicked.connect(self.set_ir_brightness)
        led_layout.addWidget(self.btn_set_ir)
        led_layout.addWidget(QLabel(" | RED:"))
        self.spin_red_brightness = QSpinBox()
        self.spin_red_brightness.setRange(0, 255)
        self.spin_red_brightness.setValue(self.red_brightness)
        led_layout.addWidget(self.spin_red_brightness)
        self.btn_set_red = QPushButton("💡 RED 적용")
        self.btn_set_red.clicked.connect(self.set_red_brightness)
        led_layout.addWidget(self.btn_set_red)
        led_group.setLayout(led_layout)
        right_layout.addWidget(led_group)

        # 5. 녹화 제어 패널
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
        right_layout.addWidget(rec_group)
        
        main_layout.addLayout(left_layout, stretch=6)
        main_layout.addLayout(right_layout, stretch=4)

        self.set_channel(self.default_graph)

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

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports: self.combo_ports.addItem(port.device)
    
    def connect_sensor(self):
        port = self.combo_ports.currentText()
        if not port: return
        self.btn_connect.setEnabled(False)
        self.sensor_thread.set_port(port)
        self.sensor_thread.start()

    def on_sensor_connected(self, success, msg):
        if success:
            self.btn_connect.setText("✅ 연결됨")
            self.btn_connect.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold;")
        else:
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("🔌 연결")

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
        else:
            QMessageBox.warning(self, "경고", "카메라가 연결되어 있지 않습니다.")

    def update_x_range(self, text):
        try:
            sensor_len = int(text)
            self.sensor_x_range = sensor_len
            self.display_window_sec = sensor_len / 200.0
        except ValueError: pass

    def redraw_graphs(self):
        sen_list = list(self.sensor_q)

        if len(sen_list) > 10:
            latest_t = sen_list[-1][0]
            s_idx = max(0, len(sen_list) - self.sensor_x_range)
            seg = sen_list[s_idx:]

            col = 1 if self.active_channel == 'IR' else 2
            t_sen = np.array([item[0] - latest_t for item in seg])
            y_sen = np.array([item[col] for item in seg])

            self.plot_sensor.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.curve_sensor.setData(x=t_sen, y=y_sen)
            self.plot_sensor.setXRange(-self.display_window_sec, 0, padding=0)

    def set_channel(self, channel):
        if channel not in ('IR', 'RED'):
            channel = 'IR'
        self.active_channel = channel
        if channel == 'IR':
            self.curve_sensor.setPen(pg.mkPen(color='#a78bfa', width=2))
            self.lbl_channel.setText("표시 중: IR")
            self.btn_ch_ir.setStyleSheet("background-color: #a78bfa; color: white; font-weight: bold;")
            self.btn_ch_red.setStyleSheet("")
        else:
            self.curve_sensor.setPen(pg.mkPen(color='#ef4444', width=2))
            self.lbl_channel.setText("표시 중: RED")
            self.btn_ch_red.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold;")
            self.btn_ch_ir.setStyleSheet("")
        self.curve_sensor.setData(x=[], y=[])

    def update_sensor_stats(self, hz, drops):
        self.lbl_hz.setText(f"Sampling Rate: {hz} Hz")
        self.lbl_drop.setText(f"Packet Drops: {drops}")

    def set_ir_brightness(self):
        val = self.spin_ir_brightness.value()
        if self.sensor_thread.isRunning(): self.sensor_thread.send_ir_brightness_command(val)

    def set_red_brightness(self):
        val = self.spin_red_brightness.value()
        if self.sensor_thread.isRunning(): self.sensor_thread.send_red_brightness_command(val)

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
        if not self.sensor_thread.isRunning():
            QMessageBox.warning(self, "경고", "센서 먼저 연결하세요.")
            return
            
        today_str = datetime.now().strftime("%m%d")
        base_dir = os.path.join("data", f"basler_{today_str}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_save_dir = os.path.join(base_dir, timestamp)
        
        # 💡 Raw 버전은 frames 폴더를 만들지 않음
        if not os.path.exists(self.current_save_dir): 
            os.makedirs(self.current_save_dir)
            
        self.recording_duration = self.spin_duration.value()
        self.remaining_time = self.recording_duration
        
        self.is_currently_recording = True
        self.btn_start.setText("⏹ 녹화 조기 종료")
        self.btn_start.setStyleSheet("background-color: #64748b; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        self.spin_duration.setEnabled(False)
        self.lbl_timer.setText(f"REC: {self.remaining_time}s")
        
        self.sensor_thread.start_recording()
        self.video_thread.start_recording(self.current_save_dir) # 💡 직접 폴더 전달
        self.timer.start(1000)

    def update_timer(self):
        self.remaining_time -= 1
        self.lbl_timer.setText(f"REC: {self.remaining_time}s")
        if self.remaining_time <= 0: self.stop_recording_phase()

    def stop_recording_phase(self):
        self.timer.stop()
        self.is_currently_recording = False
        self.sensor_thread.stop_recording()
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

    def save_csv_data(self, buffer_data):
        csv_path = os.path.join(self.current_save_dir, "ppg_sensor.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Frame_Index', 'IR_Value_Raw', 'RED_Value_Raw'])
                writer.writerows(buffer_data)
        except Exception: pass

    def load_sensor_settings(self):
        self.ir_brightness = 100
        self.red_brightness = 100
        self.default_graph = "IR"
        setting_path = os.path.join("setting", "sensor_settings.json")
        if os.path.exists(setting_path):
            try:
                with open(setting_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ir_brightness = data.get("ir_brightness", 100)
                    self.red_brightness = data.get("red_brightness", 100)
                    dg = data.get("default_graph", "IR")
                    self.default_graph = dg if dg in ("IR", "RED") else "IR"
            except Exception as e:
                print(f"Failed to load sensor settings: {e}")

    def save_sensor_settings(self):
        os.makedirs("setting", exist_ok=True)
        setting_path = os.path.join("setting", "sensor_settings.json")
        try:
            data = {}
            if os.path.exists(setting_path):
                with open(setting_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["ir_brightness"] = self.spin_ir_brightness.value()
            data["red_brightness"] = self.spin_red_brightness.value()
            data["default_graph"] = self.active_channel
            with open(setting_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save sensor settings: {e}")

    def closeEvent(self, event):
        self.save_sensor_settings()
        self.video_thread.is_running = False
        self.video_thread.wait()
        self.sensor_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())