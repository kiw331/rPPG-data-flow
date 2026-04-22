# 파일명: rPPG_Qt_LiveMonitor_JSON_Summary.py
# 지시사항: 
# 1. camera_all_settings.txt 파일 저장 시, 상단에 녹화 시간 및 현재 주요 카메라 설정(해상도, 오프셋, 포맷, FPS, 노출 등)을 JSON 형식으로 요약하여 작성
# 2. 요약 정보 아래에 기존의 전체 카메라 설정 덤프를 이어붙여 빠른 확인과 상세 참고의 편의성을 동시에 제공
# 3. 비디오 패널 동적 렌더링, 압축 UI, 와이드 화면 비율 등 이전 Adaptive UI 개선사항 유지

import sys
import os
import cv2
import serial
import serial.tools.list_ports
import time
import csv
import json # 💡 JSON 포맷 생성을 위해 추가
import numpy as np
import tifffile
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
                             QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg


# 📡 [센서 스레드] 
class SensorThread(QThread):
    update_stats_signal = pyqtSignal(int, int) 
    finished_signal = pyqtSignal(list)   
    connection_status_signal = pyqtSignal(bool, str)

    def __init__(self, gui_q):
        super().__init__()
        self.is_running = False
        self.is_recording = False
        self.record_buffer = [] 
        self.ser = None
        self.port_name = ""
        self.command_queue = queue.Queue()
        self.gui_q = gui_q  
        self.packet_count_1s = 0
        self.last_seq = -1
        self.drop_count = 0
        self.last_check_time = time.time()
        self.DEFAULT_BAUD_RATE = 115200

    def set_port(self, port):
        self.port_name = port

    def send_brightness_command(self, val):
        cmd_str = f"{val}\n" 
        self.command_queue.put(cmd_str.encode('utf-8'))

    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, self.DEFAULT_BAUD_RATE, timeout=0.01)
            time.sleep(2)
            self.ser.reset_input_buffer()
            
            self.is_running = True
            self.connection_status_signal.emit(True, f"연결됨 (Raw Binary): {self.port_name}")
            
            self.last_seq = -1
            self.drop_count = 0
            self.packet_count_1s = 0
            self.last_check_time = time.time()
            
            buffer = bytearray()
            PACKET_SIZE = 7

            while self.is_running:
                while not self.command_queue.empty():
                    try:
                        cmd = self.command_queue.get_nowait()
                        self.ser.write(cmd)
                    except: pass

                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    buffer.extend(chunk)

                    while len(buffer) >= PACKET_SIZE:
                        if buffer[0] == 0xA5 and buffer[1] == 0x5A:
                            seq = buffer[2]
                            ir_val = struct.unpack('<I', buffer[3:7])[0]
                            del buffer[:PACKET_SIZE]

                            if self.last_seq != -1:
                                expected_seq = (self.last_seq + 1) % 256
                                if seq != expected_seq:
                                    diff = (seq - expected_seq + 256) % 256
                                    self.drop_count += diff
                            self.last_seq = seq

                            self.packet_count_1s += 1
                            current_time = time.time()
                            
                            self.gui_q.append((current_time, ir_val))
                            
                            if current_time - self.last_check_time >= 1.0:
                                self.update_stats_signal.emit(self.packet_count_1s, self.drop_count)
                                self.packet_count_1s = 0
                                self.last_check_time = current_time

                            if self.is_recording:
                                self.record_buffer.append([current_time, seq, ir_val])
                        else:
                            del buffer[0]
                else:
                    self.msleep(1)
            self.ser.close()
        except Exception as e:
            self.connection_status_signal.emit(False, f"연결 실패: {e}")

    def start_recording(self):
        self.record_buffer = []
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        self.finished_signal.emit(self.record_buffer)

    def stop(self):
        self.is_running = False
        self.wait()

# 💾 [저장 스레드]
def writer_worker(q, save_dir):
    parent_dir = os.path.dirname(save_dir)
    csv_path = os.path.join(parent_dir, "camera_timestamps.csv")
    f_csv = open(csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Frame_Index", "Timestamp"])
    frame_idx = 0
    while True:
        item = q.get()
        if item is None:
            q.task_done(); break
        raw_data, timestamp = item
        file_name = f"frame_{frame_idx:04d}.tiff"
        tifffile.imwrite(os.path.join(save_dir, file_name), raw_data)
        writer.writerow([frame_idx, timestamp])
        frame_idx += 1
        q.task_done()
    f_csv.close()
    print("\n💾 [Writer] 저장 완료.")

# 📷 [카메라 스레드]
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    queue_status_signal = pyqtSignal(int)
    recording_finished_signal = pyqtSignal(str) 
    initial_settings_signal = pyqtSignal(dict)
    
    def __init__(self, gui_q):
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.img_queue = None 
        self.writer_thread = None
        self.save_dir = "" 
        self.camera = None
        self.gui_q = gui_q
        self.cmd_queue = queue.Queue()
        
        self.roi_config = (472, 472, 80)
        self.current_format = "BayerRG12" 
        
    def run(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            
            self.camera.UserSetSelector.SetValue("UserSet1")
            self.camera.UserSetLoad.Execute()
            self.camera.MaxNumBuffer.SetValue(20)
            
            init_settings = {}
            try: init_settings['format'] = self.camera.PixelFormat.GetValue()
            except Exception: pass
            try: init_settings['fps'] = self.camera.AcquisitionFrameRate.GetValue()
            except Exception: pass
            try: init_settings['exp'] = self.camera.ExposureTime.GetValue()
            except Exception: pass
            try: init_settings['color'] = self.camera.LightSourcePreset.GetValue()
            except Exception: pass
            
            self.current_format = init_settings.get('format', "BayerRG12")
            self.initial_settings_signal.emit(init_settings)
            
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            frame_count = 0
            
            while self.is_running:
                while not self.cmd_queue.empty():
                    cmd = self.cmd_queue.get_nowait()
                    try:
                        was_grabbing = self.camera.IsGrabbing()
                        if was_grabbing:
                            self.camera.StopGrabbing()
                            
                        if 'resolution' in cmd and cmd['resolution'] is not None:
                            res = cmd['resolution']
                            try:
                                self.camera.OffsetX.SetValue(0)
                                self.camera.OffsetY.SetValue(0)
                                self.camera.Width.SetValue(res['w'])
                                self.camera.Height.SetValue(res['h'])
                                self.camera.OffsetX.SetValue(res['ox'])
                                self.camera.OffsetY.SetValue(res['oy'])
                            except Exception as e:
                                print(f"❌ 해상도/오프셋 변경 실패: {e}")

                        if 'fps' in cmd:
                            self.camera.AcquisitionFrameRateEnable.SetValue(True)
                            self.camera.AcquisitionFrameRate.SetValue(float(cmd['fps']))
                        if 'exp' in cmd:
                            self.camera.ExposureTime.SetValue(float(cmd['exp']))
                        if 'color' in cmd:
                            self.camera.LightSourcePreset.SetValue(cmd['color'])
                        if 'format' in cmd:
                            self.camera.PixelFormat.SetValue(cmd['format'])
                            self.current_format = cmd['format']
                            
                        if was_grabbing:
                            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                            
                        print(f"✅ 카메라 세팅 덮어쓰기 완료")
                    except Exception as e:
                        print(f"❌ 카메라 세팅 변경 실패: {e}")
                        if not self.camera.IsGrabbing():
                            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

                if not self.camera.IsGrabbing():
                    time.sleep(0.01)
                    continue

                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    capture_time = time.time()
                    raw_data = grabResult.GetArray()
                    
                    if self.is_recording:
                        if not self.img_queue.full():
                            self.img_queue.put((raw_data.copy(), capture_time))
                        frame_count += 1
                        if frame_count % 30 == 0: 
                            self.queue_status_signal.emit(self.img_queue.qsize())
                            
                    h, w = raw_data.shape[:2]
                    
                    rx, ry, rl = self.roi_config
                    rx, ry, rl = rx & ~1, ry & ~1, max(2, rl & ~1)
                    rx, ry = max(0, min(w - rl, rx)), max(0, min(h - rl, ry))
                    
                    roi_raw = raw_data[ry:ry+rl, rx:rx+rl]
                    
                    if self.current_format == "BGR8":
                        roi_rgb = cv2.cvtColor(roi_raw, cv2.COLOR_BGR2RGB)
                    else:
                        roi_rgb = cv2.cvtColor(roi_raw, cv2.COLOR_BayerBG2RGB)
                    
                    roi_mean = cv2.mean(roi_rgb)[:3]
                    self.gui_q.append((capture_time, roi_mean))
                    
                    if not self.is_recording or frame_count % 2 == 0: 
                        if self.current_format == "BGR8":
                            view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                            view_rgb = cv2.resize(view_rgb, (0, 0), fx=0.5, fy=0.5) 
                            view_8bit = view_rgb
                        else:
                            view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerBG2RGB)
                            view_rgb = cv2.resize(view_rgb, (0, 0), fx=0.5, fy=0.5) 
                            if self.current_format == "BayerRG12":
                                view_8bit = (view_rgb >> 4).astype(np.uint8)
                            else:
                                view_8bit = view_rgb.astype(np.uint8)
                        
                        rx_h, ry_h, rl_h = rx // 2, ry // 2, rl // 2
                        cv2.rectangle(view_8bit, (rx_h, ry_h), (rx_h+rl_h, ry_h+rl_h), (0, 0, 255), 2)
                        
                        self.change_pixmap_signal.emit(view_8bit)
                        
                grabResult.Release()
            self.camera.StopGrabbing(); self.camera.Close()
        except Exception as e:
            print(f"❌ [Camera] 오류: {e}")
            
    def start_recording(self, save_dir):
            gc.collect()
            self.save_dir = save_dir
            self.img_queue = queue.Queue(maxsize=3000)
            self.writer_thread = threading.Thread(target=writer_worker, args=(self.img_queue, self.save_dir), daemon=True)
            self.writer_thread.start()
            self.is_recording = True
            
            # 💡 [추가] 실제 녹화 시작 시간을 기록
            self.rec_start_time = time.time() 
        
    def stop_recording(self):
        self.is_recording = False
        if self.img_queue: self.img_queue.put(None) 
        
        # 💡 [추가] 실제 녹화 소요 시간(초) 계산 (소수점 2자리)
        actual_duration = 0
        if hasattr(self, 'rec_start_time'):
            actual_duration = round(time.time() - self.rec_start_time, 2)
        
        if self.save_dir and self.camera is not None:
            parent_dir = os.path.dirname(self.save_dir)
            settings_path = os.path.join(parent_dir, "camera_all_settings.txt")
            temp_pfs_path = os.path.join(parent_dir, "temp_settings.pfs")
            
            try:
                pylon.FeaturePersistence.Save(temp_pfs_path, self.camera.GetNodeMap())
                with open(temp_pfs_path, 'r', encoding='utf-8') as f:
                    full_dump = f.read()
                os.remove(temp_pfs_path) 
                
                def get_val(node):
                    try: return getattr(self.camera, node).GetValue()
                    except: return None
                
                fps_result = get_val("ResultingFrameRate")
                if fps_result is not None: fps_result = round(fps_result, 2)
                
                summary = {
                    "Program_Version": "Monitor",  # 💡 [여기에 추가] 프로그램 버전 기록
                    "Record_Duration_sec": actual_duration, 
                    "Resolution": {
                        "Width": get_val("Width"),
                        "Height": get_val("Height"),
                        "OffsetX": get_val("OffsetX"),
                        "OffsetY": get_val("OffsetY")
                    },
                    "PixelFormat": get_val("PixelFormat"),
                    "FPS_Target": get_val("AcquisitionFrameRate"),
                    "FPS_Result": fps_result,
                    "Exposure_us": get_val("ExposureTime"),
                    "Color_Temp": get_val("LightSourcePreset")
                }
                
                with open(settings_path, 'w', encoding='utf-8') as f:
                    f.write("=== [Summary] Main Camera Settings ===\n")
                    f.write(json.dumps(summary, indent=4, ensure_ascii=False))
                    f.write("\n\n\n=== [Full Dump] Camera All Settings ===\n")
                    f.write(full_dump)
                    
                print(f"💾 [Camera] 요약 + 전체 설정 저장 완료 (실제 녹화 시간: {actual_duration}초)")
                
            except Exception as e:
                print(f"❌ 설정 파일 저장 실패: {e}")
                
        self.recording_finished_signal.emit(self.save_dir)


# [UI 클래스] 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG_Qt_LiveMonitor (JSON Summary Log)")
        self.resize(1350, 720) 
        
        self.recording_duration = 60 
        self.remaining_time = 0
        self.current_save_dir = "" 
        self.is_currently_recording = False
        
        self.display_window_sec = 2.0 
        self.sensor_x_range = 400
        self.cam_x_range = int(400 * (60 / 200)) 

        MAX_LEN_SEN = 200 * 10
        MAX_LEN_CAM = 60 * 10
        
        self.sensor_q = deque(maxlen=MAX_LEN_SEN)
        self.cam_q = deque(maxlen=MAX_LEN_CAM)

        self.init_ui()
        
        self.video_thread = CameraThread(self.cam_q)
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
        left_layout.addLayout(row_conn_monitor)

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

        area_group = QGroupBox("ROI Config (X, Y, Length)")
        area_layout = QVBoxLayout()
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("🟦 ROI:"))
        self.spin_roi_x = QSpinBox(); self.spin_roi_x.setRange(0, 4000); self.spin_roi_x.setValue(472)
        self.spin_roi_y = QSpinBox(); self.spin_roi_y.setRange(0, 4000); self.spin_roi_y.setValue(472)
        self.spin_roi_l = QSpinBox(); self.spin_roi_l.setRange(2, 1000); self.spin_roi_l.setValue(80)
        roi_layout.addWidget(self.spin_roi_x); roi_layout.addWidget(self.spin_roi_y); roi_layout.addWidget(self.spin_roi_l)
        
        area_layout.addLayout(roi_layout)
        area_group.setLayout(area_layout)
        left_layout.addWidget(area_group)
        
        self.spin_roi_x.valueChanged.connect(self.update_areas)
        self.spin_roi_y.valueChanged.connect(self.update_areas)
        self.spin_roi_l.valueChanged.connect(self.update_areas)

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
        # ➡️ [우측 패널]
        # =========================================================
        graphs_group = QGroupBox("Real-time Signals (Time-Synced Raw)")
        graphs_layout = QVBoxLayout()
        
        self.plot_sensor = pg.PlotWidget(title="Hardware PPG Sensor (Raw)")
        self.plot_sensor.setBackground('k')
        self.plot_sensor.setMinimumHeight(150) 
        self.curve_sensor = self.plot_sensor.plot(pen=pg.mkPen(color='#f43f5e', width=2))
        
        self.plot_roi = pg.PlotWidget(title="Camera ROI (Raw)")
        self.plot_roi.setBackground('k')
        self.plot_roi.setMinimumHeight(150)
        self.curve_roi = self.plot_roi.plot(pen=pg.mkPen(color='#3b82f6', width=2)) 
        
        graphs_layout.addWidget(self.plot_sensor)
        graphs_layout.addWidget(self.plot_roi)
        graphs_group.setLayout(graphs_layout)
        right_layout.addWidget(graphs_group)

        bottom_options_group = QGroupBox("Additional Options")
        bottom_options_layout = QHBoxLayout()
        
        bottom_options_layout.addWidget(QLabel("X-Axis:"))
        self.combo_x_range = QComboBox()
        self.combo_x_range.addItems(["200", "300", "400", "800", "1200"])
        self.combo_x_range.setCurrentText("400")
        self.combo_x_range.currentTextChanged.connect(self.update_x_range)
        bottom_options_layout.addWidget(self.combo_x_range)

        bottom_options_layout.addWidget(QLabel(" | rPPG Ch:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(["R", "G (Def)", "B"])
        self.combo_channel.setCurrentIndex(1) 
        self.combo_channel.currentTextChanged.connect(self.clear_cam_buffers)
        bottom_options_layout.addWidget(self.combo_channel)
        
        self.check_invert = QCheckBox("Invert(-G)")
        bottom_options_layout.addWidget(self.check_invert)

        bottom_options_layout.addWidget(QLabel(" | IR LED:"))
        self.spin_brightness = QSpinBox()
        self.spin_brightness.setRange(0, 255)
        self.spin_brightness.setValue(60) 
        bottom_options_layout.addWidget(self.spin_brightness)
        
        self.btn_set_brightness = QPushButton("💡 적용")
        self.btn_set_brightness.clicked.connect(self.set_brightness)
        bottom_options_layout.addWidget(self.btn_set_brightness)

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

    def update_areas(self):
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            self.video_thread.roi_config = (self.spin_roi_x.value(), self.spin_roi_y.value(), self.spin_roi_l.value())

    def update_x_range(self, text):
        try:
            sensor_len = int(text)
            self.sensor_x_range = sensor_len
            self.cam_x_range = int(sensor_len * (60 / 200))
            self.display_window_sec = sensor_len / 200.0
        except ValueError: pass

    def clear_cam_buffers(self):
        self.cam_q.clear()

    def redraw_graphs(self):
        sen_list = list(self.sensor_q)
        cam_list = list(self.cam_q)
        
        if len(sen_list) > 10 and len(cam_list) > 10:
            latest_t = max(sen_list[-1][0], cam_list[-1][0])
            
            s_idx = max(0, len(sen_list) - self.sensor_x_range)
            c_idx = max(0, len(cam_list) - self.cam_x_range)

            t_sen = np.array([item[0] - latest_t for item in sen_list[s_idx:]])
            y_sen = np.array([item[1] for item in sen_list[s_idx:]])
            
            t_cam = np.array([item[0] - latest_t for item in cam_list[c_idx:]])
            
            ch_idx = self.combo_channel.currentIndex()
            y_roi = np.array([item[1][ch_idx] for item in cam_list[c_idx:]])
            
            if self.check_invert.isChecked():
                y_roi = -y_roi

            self.plot_sensor.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot_roi.enableAutoRange(axis=pg.ViewBox.YAxis)

            self.curve_sensor.setData(x=t_sen, y=y_sen)
            self.curve_roi.setData(x=t_cam, y=y_roi)
            
            self.plot_sensor.setXRange(-self.display_window_sec, 0, padding=0)
            self.plot_roi.setXRange(-self.display_window_sec, 0, padding=0)

    def update_sensor_stats(self, hz, drops):
        self.lbl_hz.setText(f"Sampling Rate: {hz} Hz")
        self.lbl_drop.setText(f"Packet Drops: {drops}")

    def set_brightness(self):
        val = self.spin_brightness.value()
        if self.sensor_thread.isRunning(): self.sensor_thread.send_brightness_command(val)

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
        self.frames_save_dir = os.path.join(self.current_save_dir, "frames")
        if not os.path.exists(self.frames_save_dir): os.makedirs(self.frames_save_dir)
        
        self.recording_duration = self.spin_duration.value()
        self.remaining_time = self.recording_duration
        
        self.is_currently_recording = True
        self.btn_start.setText("⏹ 녹화 조기 종료")
        self.btn_start.setStyleSheet("background-color: #64748b; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        self.spin_duration.setEnabled(False)
        self.lbl_timer.setText(f"REC: {self.remaining_time}s")
        
        self.sensor_thread.start_recording()
        self.video_thread.start_recording(self.frames_save_dir) 
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
            self.video_thread.is_running = False
            self.video_thread.wait(1000)
            QApplication.quit() 

    def save_csv_data(self, buffer_data):
        csv_path = os.path.join(self.current_save_dir, "ppg_sensor.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Frame_Index', 'IR_Value_Raw'])
                writer.writerows(buffer_data)
        except Exception: pass

    def closeEvent(self, event):
        self.video_thread.is_running = False
        self.video_thread.wait()
        self.sensor_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())