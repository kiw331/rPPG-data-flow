# 파일명: camera.py
# 목적: Basler pypylon을 사용하는 카메라 Base 클래스 정의

import os
import time
import queue
import threading
import gc
import numpy as np
from pypylon import pylon
from PyQt5.QtCore import QThread, pyqtSignal
from .storage import save_camera_settings # 공통된 카메라 설정 세이브 함수 임포트

class BaseCameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    queue_status_signal = pyqtSignal(int)
    recording_finished_signal = pyqtSignal(str) 
    initial_settings_signal = pyqtSignal(dict)
    
    def __init__(self, gui_q=None, user_set="UserSet1", program_version="Base"):
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.img_queue = None 
        self.writer_thread = None
        self.save_dir = "" 
        self.camera = None
        self.gui_q = gui_q
        self.cmd_queue = queue.Queue()
        self.current_format = "BayerRG12" 
        self.rec_start_time = 0
        self.user_set = user_set
        self.program_version = program_version
        self.frame_count = 0
        
    def run(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            
            self.camera.UserSetSelector.SetValue(self.user_set)
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
            self.frame_count = 0
            
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
                        if self.frame_count % 30 == 0: 
                            self.queue_status_signal.emit(self.img_queue.qsize())
                    
                    # 서브클래스에서 개별 구현할 프레임 처리 (ROI 연산 등)
                    self.process_frame(raw_data, capture_time)
                    
                    self.frame_count += 1
                        
                grabResult.Release()
            self.camera.StopGrabbing(); self.camera.Close()
        except Exception as e:
            print(f"❌ [Camera] 오류: {e}")
            
    def process_frame(self, raw_data, capture_time):
        """서브클래스에서 오버라이드하여 데이터 추출 및 렌더링을 처리하세요."""
        pass
        
    def render_base_pixmap(self, raw_data):
        """기본 BGR8 / Bayer 변환 후 스케일 축소된 view_8bit 이미지를 반환합니다."""
        import cv2
        if self.current_format == "BGR8":
            view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            view_rgb = cv2.resize(view_rgb, (0, 0), fx=0.5, fy=0.5) 
            view_8bit = view_rgb
        else:
            # BayerRG12: QImage.Format_RGB888로 전달하므로 RGB 출력
            view_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
            view_rgb = cv2.resize(view_rgb, (0, 0), fx=0.5, fy=0.5) 
            if self.current_format in ("BayerRG12", "BayerRG8"):
                view_8bit = (view_rgb >> 4).astype(np.uint8)  # 12bit → 8bit
            else:
                view_8bit = view_rgb.astype(np.uint8)
        return view_8bit

    def start_recording_with_worker(self, save_dir, worker_target):
        gc.collect()
        self.save_dir = save_dir
        self.img_queue = queue.Queue(maxsize=3000)
        self.writer_thread = threading.Thread(target=worker_target, args=(self.img_queue, self.save_dir), daemon=True)
        self.writer_thread.start()
        self.is_recording = True
        self.rec_start_time = time.time()
        
    def stop_recording_and_save_settings(self, settings_parent_dir=None):
        """녹화를 종료하고 큐를 닫은 뒤, camera_all_settings.txt를 저장합니다."""
        self.is_recording = False
        if self.img_queue: self.img_queue.put(None) 
        
        actual_duration = 0
        if hasattr(self, 'rec_start_time') and self.rec_start_time > 0:
            actual_duration = round(time.time() - self.rec_start_time, 2)
            
        if self.camera is not None:
            save_dest = settings_parent_dir if settings_parent_dir else os.path.dirname(self.save_dir)
            save_camera_settings(self.camera, self.program_version, actual_duration, save_dest)
                
        self.recording_finished_signal.emit(self.save_dir)
