import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tifffile

# modules 디렉터리를 sys.path에 추가하여 modules/sam3를 직접 import할 수 있도록 설정
modules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                             QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
                             QFileDialog, QMessageBox, QGroupBox, QProgressBar, QSplitter,
                             QScrollArea, QDialog, QCheckBox, QScrollBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer

# =====================================================================
# SAM3 선택적 로딩 함수
# =====================================================================
HAS_SAM3 = False
sam3_model_builder = None
Sam3Processor = None

def try_import_sam3(force_no=False):
    global HAS_SAM3, sam3_model_builder, Sam3Processor
    if force_no:
        HAS_SAM3 = False
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            print("[SAM3] CUDA is not available. Disabling SAM3 mode.")
            HAS_SAM3 = False
            return False
        
        # sam3 임포트 시도
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as Processor
        
        sam3_model_builder = build_sam3_image_model
        Sam3Processor = Processor
        HAS_SAM3 = True
        print("[SAM3] SAM3 successfully imported with CUDA support.")
        return True
    except Exception as e:
        print(f"[SAM3] SAM3 Import failed or disabled: {e}")
        HAS_SAM3 = False
        return False


# =====================================================================
# Unet++ 선택적 로딩 함수
# =====================================================================
HAS_UNETPP = False

def try_import_unetpp(force_no=False):
    global HAS_UNETPP
    if force_no:
        HAS_UNETPP = False
        return False
    try:
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import segmentation_models_pytorch as smp
        
        HAS_UNETPP = True
        print("[Unet++] Unet++ dependencies successfully imported.")
        return True
    except Exception as e:
        print(f"[Unet++] Unet++ Import failed or disabled: {e}")
        HAS_UNETPP = False
        return False



# =====================================================================
# 예외 클래스 정의
# =====================================================================
class HFAuthError(Exception):
    pass


# ====================================
class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict) # 결과 데이터 딕셔너리 전달
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path, start_frame_idx, duration_sec, rois_info, bg_roi_info, bpm_min, bpm_max,
                 roi_mode="manual", sam3_model=None, active_sam3_rois=None, sam_conf=0.3, sam_min_area=100,
                 unetpp_model=None, unetpp_transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.start_frame_idx = start_frame_idx
        self.duration_sec = duration_sec
        self.rois_info = rois_info # list of (x, y, w, h) for ROI 1~3 (manual 모드용)
        self.bg_roi_info = bg_roi_info # (x, y, w, h) or None (manual 모드용)
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.is_running = True
        
        # SAM3 및 Unet++ 모드 관련 설정
        self.roi_mode = roi_mode
        self.sam3_model = sam3_model
        self.active_sam3_rois = active_sam3_rois if active_sam3_rois is not None else []
        self.sam_conf = sam_conf
        self.sam_min_area = sam_min_area
        self.unetpp_model = unetpp_model
        self.unetpp_transform = unetpp_transform

    def detect_sam3_regions(self, img_rgb, processor, autocast_ctx) -> dict:
        """
        주어진 RGB 이미지에서 SAM3를 통해 'tail', 'foot1', 'foot2' 마스크를 반환.
        """
        from PIL import Image
        if img_rgb.dtype == np.uint16:
            img_8u = (img_rgb / 16.0).clip(0, 255).astype(np.uint8)
        else:
            img_8u = img_rgb.astype(np.uint8)
        pil_image = Image.fromarray(img_8u)
        
        import torch
        res_masks = {}
        
        # 1. Image Encoding
        with autocast_ctx:
            state = processor.set_image(pil_image)
            
        # 2. 'tail' 검출 (최대 1개 blob)
        if 'tail' in self.active_sam3_rois:
            with autocast_ctx:
                state = processor.set_text_prompt(prompt="tail", state=state)
            masks = state.get("masks")
            if masks is not None and masks.shape[0] > 0:
                combined_mask = np.any(masks.cpu().numpy(), axis=0)[0].astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
                
                valid_blobs = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= self.sam_min_area:
                        valid_blobs.append((i, area))
                
                if valid_blobs:
                    valid_blobs.sort(key=lambda x: x[1], reverse=True)
                    best_label = valid_blobs[0][0]
                    res_masks['tail'] = (labels == best_label).astype(np.uint8)
            
            processor.reset_all_prompts(state)
            with autocast_ctx:
                state = processor.set_image(pil_image)

        # 3. 'foot' 검출 (최대 2개 blob, x좌표 순 정렬)
        if 'foot1' in self.active_sam3_rois or 'foot2' in self.active_sam3_rois:
            with autocast_ctx:
                state = processor.set_text_prompt(prompt="foot", state=state)
            masks = state.get("masks")
            if masks is not None and masks.shape[0] > 0:
                combined_mask = np.any(masks.cpu().numpy(), axis=0)[0].astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
                
                valid_blobs = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= self.sam_min_area:
                        cx = centroids[i][0]
                        valid_blobs.append((i, area, cx))
                        
                if valid_blobs:
                    valid_blobs.sort(key=lambda x: x[1], reverse=True)
                    top_blobs = valid_blobs[:2]
                    
                    if len(top_blobs) == 2:
                        top_blobs.sort(key=lambda x: x[2]) # x가 작은 순 (왼쪽이 먼저)
                        res_masks['foot1'] = (labels == top_blobs[0][0]).astype(np.uint8)
                        res_masks['foot2'] = (labels == top_blobs[1][0]).astype(np.uint8)
                    else:
                        # 1개만 검출된 경우 foot1으로 지정
                        res_masks['foot1'] = (labels == top_blobs[0][0]).astype(np.uint8)
                        
        return res_masks

    def detect_unetpp_regions(self, img_rgb, model, transform, device) -> dict:
        """
        주어진 RGB 이미지에서 Unet++를 통해 'tail', 'foot1', 'foot2' 마스크를 반환.
        """
        import torch
        import cv2
        import numpy as np

        if img_rgb.dtype == np.uint16:
            img_8u = (img_rgb / 16.0).clip(0, 255).astype(np.uint8)
        else:
            img_8u = img_rgb.astype(np.uint8)

        tensor = transform(image=img_8u)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor) # (1, C, H, W)
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy() # (H, W)

        res_masks = {}

        # 2: tail 검출 (최대 1개 blob)
        if 'tail' in self.active_sam3_rois:
            tail_mask = (pred_mask == 2).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tail_mask)
            
            valid_blobs = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.sam_min_area:
                    valid_blobs.append((i, area))
            
            if valid_blobs:
                valid_blobs.sort(key=lambda x: x[1], reverse=True)
                best_label = valid_blobs[0][0]
                res_masks['tail'] = (labels == best_label).astype(np.uint8)

        # 1: foot 검출 (최대 2개 blob, x좌표 순 정렬)
        if 'foot1' in self.active_sam3_rois or 'foot2' in self.active_sam3_rois:
            foot_mask = (pred_mask == 1).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foot_mask)
            
            valid_blobs = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.sam_min_area:
                    cx = centroids[i][0]
                    valid_blobs.append((i, area, cx))
                    
            if valid_blobs:
                valid_blobs.sort(key=lambda x: x[1], reverse=True)
                top_blobs = valid_blobs[:2]
                
                if len(top_blobs) == 2:
                    top_blobs.sort(key=lambda x: x[2]) # x가 작은 순 (왼쪽이 먼저)
                    res_masks['foot1'] = (labels == top_blobs[0][0]).astype(np.uint8)
                    res_masks['foot2'] = (labels == top_blobs[1][0]).astype(np.uint8)
                else:
                    # 1개만 검출된 경우 foot1으로 지정
                    res_masks['foot1'] = (labels == top_blobs[0][0]).astype(np.uint8)

        return res_masks

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
            y_sensor_ir = np.array([])
            y_sensor_red = np.array([])
            if has_sensor:
                sensor_mask = (df_sensor['Timestamp'] - t0 >= start_time_offset) & (df_sensor['Timestamp'] - t0 <= end_time_offset)
                df_sensor_cut = df_sensor[sensor_mask]
                t_sensor = df_sensor_cut['Timestamp'].values - t0
                
                # IR
                if 'IR_Value_Raw' in df_sensor_cut.columns:
                    y_sensor_ir = df_sensor_cut['IR_Value_Raw'].values
                elif 'IR_Value' in df_sensor_cut.columns:
                    y_sensor_ir = df_sensor_cut['IR_Value'].values
                
                # RED
                if 'RED_Value_Raw' in df_sensor_cut.columns:
                    y_sensor_red = df_sensor_cut['RED_Value_Raw'].values
                elif 'RED_Value' in df_sensor_cut.columns:
                    y_sensor_red = df_sensor_cut['RED_Value'].values
                    
                # Default y_sensor is IR, fallback to RED
                if len(y_sensor_ir) > 0:
                    y_sensor = y_sensor_ir
                elif len(y_sensor_red) > 0:
                    y_sensor = y_sensor_red

            # 데이터 추출 루프 준비
            total_frames = len(df_cam_cut)
            t_cam_list = []
            frame_indices = []
            
            # ROI 설정에 따른 변수들 초기화
            if self.roi_mode == "sam3":
                # SAM3 전용 변수들
                roi_names = list(self.active_sam3_rois)
                roi_count = len(roi_names)
                has_bg = self.bg_roi_info is not None
                total_regions = roi_count + (1 if has_bg else 0)
                
                # SAM3 Processor 스레드 내 로딩
                self.progress_signal.emit(5, "SAM3 세그먼테이션 엔진 로드 중...")
                from sam3.model.sam3_image_processor import Sam3Processor
                import torch
                
                device = "cuda"
                processor = Sam3Processor(self.sam3_model, device=device, confidence_threshold=self.sam_conf)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                
                last_masks = {}
                warning_messages = []
            elif self.roi_mode == "unetpp":
                # Unet++ 전용 변수들
                roi_names = list(self.active_sam3_rois)
                roi_count = len(roi_names)
                has_bg = self.bg_roi_info is not None
                total_regions = roi_count + (1 if has_bg else 0)
                
                self.progress_signal.emit(5, "Unet++ 세그먼테이션 엔진 준비 중...")
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                last_masks = {}
                warning_messages = []
            else:
                # 수동 ROI 전용 변수들
                roi_names = [f"ROI {i+1}" for i in range(len(self.rois_info))]
                roi_count = len(self.rois_info)
                has_bg = self.bg_roi_info is not None
                total_regions = roi_count + (1 if has_bg else 0)
                
                all_rois = list(self.rois_info)
                if has_bg:
                    all_rois.append(self.bg_roi_info)
                warning_messages = []

            # raw_rgb_data shape: [region_idx][frame_idx][channel(0:R, 1:G, 2:B)]
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
                
                # 컬러 변환 (BayerRG12 픽셀 포맷 버그 수정 반영: cv2.COLOR_BayerBG2RGB 사용)
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.roi_mode == "sam3":
                    # SAM3 각 프레임 세그먼테이션 실행
                    current_masks = self.detect_sam3_regions(img_rgb, processor, autocast_ctx)
                    
                    for r_idx, name in enumerate(roi_names):
                        mask = current_masks.get(name)
                        if mask is None or np.sum(mask) == 0:
                            # 검출 실패 시 직전 프레임 마스크 재사용
                            if name in last_masks:
                                mask = last_masks[name]
                                warning_messages.append(f"Frame {f_idx}: '{name}' 객체가 검출되지 않아 직전 프레임의 영역을 사용했습니다.")
                            else:
                                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                        
                        # 검출 성공시 마스크 캐싱
                        if np.sum(mask) > 0:
                            last_masks[name] = mask
                            
                        # 마스크 영역 내 평균 RGB 추출
                        if np.sum(mask) > 0:
                            crop_pixels = img_rgb[mask > 0]
                            r_mean = np.mean(crop_pixels[:, 0])
                            g_mean = np.mean(crop_pixels[:, 1])
                            b_mean = np.mean(crop_pixels[:, 2])
                            raw_rgb_data[r_idx, idx, 0] = r_mean
                            raw_rgb_data[r_idx, idx, 1] = g_mean
                            raw_rgb_data[r_idx, idx, 2] = b_mean
                            
                    # 수동 배경 영역이 추가 설정되었을 경우 추출
                    if has_bg:
                        bx, by, bw, bh = self.bg_roi_info
                        y_start = max(0, int(by))
                        y_end = min(img_rgb.shape[0], int(by+bh))
                        x_start = max(0, int(bx))
                        x_end = min(img_rgb.shape[1], int(bx+bw))
                        crop = img_rgb[y_start:y_end, x_start:x_end]
                        if crop.size > 0:
                            raw_rgb_data[roi_count, idx, 0] = np.mean(crop[:, :, 0])
                            raw_rgb_data[roi_count, idx, 1] = np.mean(crop[:, :, 1])
                            raw_rgb_data[roi_count, idx, 2] = np.mean(crop[:, :, 2])
                            
                elif self.roi_mode == "unetpp":
                    # Unet++ 각 프레임 세그먼테이션 실행
                    current_masks = self.detect_unetpp_regions(img_rgb, self.unetpp_model, self.unetpp_transform, device)
                    
                    for r_idx, name in enumerate(roi_names):
                        mask = current_masks.get(name)
                        if mask is None or np.sum(mask) == 0:
                            # 검출 실패 시 직전 프레임 마스크 재사용
                            if name in last_masks:
                                mask = last_masks[name]
                                warning_messages.append(f"Frame {f_idx}: '{name}' 객체가 검출되지 않아 직전 프레임의 영역을 사용했습니다.")
                            else:
                                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                        
                        # 검출 성공시 마스크 캐싱
                        if np.sum(mask) > 0:
                            last_masks[name] = mask
                            
                        # 마스크 영역 내 평균 RGB 추출
                        if np.sum(mask) > 0:
                            crop_pixels = img_rgb[mask > 0]
                            r_mean = np.mean(crop_pixels[:, 0])
                            g_mean = np.mean(crop_pixels[:, 1])
                            b_mean = np.mean(crop_pixels[:, 2])
                            raw_rgb_data[r_idx, idx, 0] = r_mean
                            raw_rgb_data[r_idx, idx, 1] = g_mean
                            raw_rgb_data[r_idx, idx, 2] = b_mean
                            
                    # 수동 배경 영역이 추가 설정되었을 경우 추출
                    if has_bg:
                        bx, by, bw, bh = self.bg_roi_info
                        y_start = max(0, int(by))
                        y_end = min(img_rgb.shape[0], int(by+bh))
                        x_start = max(0, int(bx))
                        x_end = min(img_rgb.shape[1], int(bx+bw))
                        crop = img_rgb[y_start:y_end, x_start:x_end]
                        if crop.size > 0:
                            raw_rgb_data[roi_count, idx, 0] = np.mean(crop[:, :, 0])
                            raw_rgb_data[roi_count, idx, 1] = np.mean(crop[:, :, 1])
                            raw_rgb_data[roi_count, idx, 2] = np.mean(crop[:, :, 2])
                else:
                    # manual 모드 ROI 평균 RGB 추출
                    for r_idx, (x, y, w, h) in enumerate(all_rois):
                        # 경계 처리
                        y_start = max(0, int(y))
                        y_end = min(img_rgb.shape[0], int(y+h))
                        x_start = max(0, int(x))
                        x_end = min(img_rgb.shape[1], int(x+w))
                        
                        crop = img_rgb[y_start:y_end, x_start:x_end]
                        if crop.size > 0:
                            r_mean = np.mean(crop[:, :, 0])
                            g_mean = np.mean(crop[:, :, 1])
                            b_mean = np.mean(crop[:, :, 2])
                            raw_rgb_data[r_idx, idx, 0] = r_mean
                            raw_rgb_data[r_idx, idx, 1] = g_mean
                            raw_rgb_data[r_idx, idx, 2] = b_mean

                t_cam_list.append(f_time)
                frame_indices.append(f_idx)

                if idx % 5 == 0 or idx == total_frames - 1:
                    prog = 10 + int(85 * (idx / total_frames))
                    self.progress_signal.emit(prog, f"추출 및 분석 중... ({idx+1}/{total_frames})")

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
                'roi_names': roi_names,
                'warning_messages': warning_messages,
                'has_bg': has_bg,
                'has_sensor': has_sensor,
                't_sensor': t_sensor,
                'y_sensor': y_sensor,
                'y_sensor_ir': y_sensor_ir,
                'y_sensor_red': y_sensor_red,
                't0': t0,
                't0_sensor': df_sensor['Timestamp'].iloc[0] if (has_sensor and df_sensor is not None and len(df_sensor) > 0) else t0,
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
# PPG 보간 재건 함수 (standalone, GraphPopup 및 MainWindow 공용)
# =====================================================================
def reconstruct_ppg_signal(t_sensor, y_sensor, fs, bpf_lo, bpf_hi,
                            df_segments, t0, t0_sensor):
    """ppg_gap_viewer.py 와 동일한 보간 알고리즘 적용.
    
    Args:
        t_sensor  : 분석 구간의 센서 타임스탬프 배열 (절대 기준)
        y_sensor  : 대응하는 원시 신호 배열
        fs        : 센서 샘플링 주파수 (Hz)
        bpf_lo    : 밴드패스 하한 (Hz)
        bpf_hi    : 밴드패스 상한 (Hz)
        df_segments : hr_segments.csv DataFrame
        t0        : 분석 기준 시간 원점 (카메라/센서 공통)
        t0_sensor : 센서 CSV 원점 타임스탬프
    
    Returns:
        sig_rp   : 보간 포함 재건 신호 (NaN=미채움)
        sig_sp   : 합성 구간만 추출한 신호 (NaN=실신호)
        filled   : bool 배열 – 채워진 인덱스
        is_syn   : bool 배열 – 합성(보간) 인덱스
        valid    : 유효 세그먼트 리스트
        gaps     : 갭 구간 리스트
    """
    PAD_SEC    = 0.30
    MA_WIN_SEC = 0.15
    BPF_ORDER  = 4
    PEAK_PROM  = 0.3

    if df_segments is None or len(df_segments) == 0:
        n = len(t_sensor)
        return (np.full(n, np.nan), np.full(n, np.nan),
                np.zeros(n, bool), np.zeros(n, bool), [], [])

    # 센서 타임스탬프를 hr_segments.csv 기준으로 변환
    t_rel = t_sensor + t0 - t0_sensor
    t        = t_rel
    sig_raw  = y_sensor.astype(float)

    def _bpf(sig):
        nyq = fs / 2
        hi  = min(bpf_hi, nyq * 0.98)
        lo  = max(0.01,   bpf_lo)
        b, a = butter(BPF_ORDER, [lo / nyq, hi / nyq], btype='band')
        return filtfilt(b, a, sig)

    def proc_iso(s, e):
        m = (t >= s) & (t <= e)
        t_i, s_i = t[m], sig_raw[m].copy()
        if len(s_i) < 3:
            return t_i, s_i
        pad = int(PAD_SEC * fs)
        s_pad = np.pad(s_i, pad, mode='edge')
        ma_n  = max(3, int(MA_WIN_SEC * fs))
        s_pad = s_pad - uniform_filter1d(s_pad, ma_n)
        s_pad = _bpf(s_pad)
        return t_i, s_pad[pad:-pad]

    def detect(ts, ss, seg_start, seg_end):
        prom      = ss.std() * PEAK_PROM
        prom_edge = ss.std() * 0.12
        mind_boot = max(2, int(0.05 * fs))
        pk0, _ = find_peaks(ss, distance=mind_boot, prominence=prom)
        if len(pk0) < 2:
            vl0, _ = find_peaks(-ss, distance=mind_boot, prominence=prom)
            return pk0, vl0, False, False, []
        diffs = np.diff(ts[pk0])
        min_period = 1.0 / bpf_hi
        valid_d = diffs[diffs >= min_period]
        avg_T   = np.median(valid_d) if len(valid_d) >= 2 else np.median(diffs)
        mind    = max(mind_boot, round(0.6 * avg_T * fs))
        pk, _  = find_peaks( ss, distance=mind, prominence=prom)
        vl, _  = find_peaks(-ss, distance=mind, prominence=prom)
        if len(pk) < 2:
            return pk, vl, False, False, []
        diffs_p  = np.diff(ts[pk])
        valid_dp = diffs_p[diffs_p >= min_period]
        avg_T    = np.median(valid_dp) if len(valid_dp) >= 2 else np.median(diffs_p)

        def _fill(arr, inv=False):
            nonlocal avg_T
            changed = True
            while changed:
                changed = False
                if len(arr) < 2: break
                ivls = np.diff(ts[arr])
                for i in np.where(ivls > 1.4 * avg_T)[0]:
                    t_mid = (ts[arr[i]] + ts[arr[i+1]]) / 2
                    win   = 0.35 * avg_T
                    m_w   = (ts >= t_mid - win) & (ts <= t_mid + win)
                    if m_w.sum() < 3: continue
                    sw = -ss[m_w] if inv else ss[m_w]
                    ep, _ = find_peaks(sw, distance=mind_boot, prominence=prom * 0.15)
                    ni = (np.where(m_w)[0][ep[np.argmax(sw[ep])]] if len(ep)
                          else np.where(m_w)[0][np.argmax(sw)])
                    arr   = np.sort(np.unique(np.append(arr, ni)))
                    avg_T = np.median(np.diff(ts[arr]))
                    changed = True; break
            return arr

        interior_syn = []
        for i in np.where(np.diff(ts[pk]) > 1.4 * avg_T)[0]:
            interior_syn.append((ts[pk[i]], ts[pk[i+1]]))
        pk = _fill(pk); vl = _fill(vl, inv=True)

        avg_T  = np.median(np.diff(ts[pk]))
        ref_pk = np.median(ss[pk])

        def _edge(t_lo, t_hi):
            m = (ts >= t_lo) & (ts <= t_hi)
            if m.sum() <= 3: return np.array([], dtype=int)
            ep, _ = find_peaks(ss[m], distance=mind_boot, prominence=prom_edge)
            if not len(ep): return np.array([], dtype=int)
            ai = np.where(m)[0][ep]
            ok = (ss[ai] > 0) & (ss[ai] <= ref_pk * 1.8)
            if not ok.any(): return np.array([], dtype=int)
            ai = ai[ok]
            return np.array([ai[np.argmax(ss[ai])]])

        if ts[pk[0]] > seg_start + avg_T:
            new = _edge(seg_start, seg_start + avg_T)
            if len(new): pk = np.sort(np.unique(np.append(pk, new)))
        if ts[pk[-1]] < seg_end - avg_T:
            new = _edge(seg_end - avg_T, seg_end)
            if len(new): pk = np.sort(np.unique(np.append(pk, new)))

        events, last_type, trunc = [], None, False
        for p in pk: events.append((ts[p],  1, p))
        for v in vl: events.append((ts[v], -1, v))
        events.sort(key=lambda x: x[0])
        valid_ev = []
        for ev in events:
            _, tp, _ = ev
            if last_type is not None and tp == last_type: trunc = True; break
            valid_ev.append(ev); last_type = tp
        if trunc:
            pk = np.array([e[2] for e in valid_ev if e[1] ==  1], dtype=int)
            vl = np.array([e[2] for e in valid_ev if e[1] == -1], dtype=int)

        added_head = added_tail = False
        if len(pk) >= 2:
            avg_T = np.median(np.diff(ts[pk]))
            frt   = ts[pk[0]]
            while ts[pk[0]] - seg_start > avg_T:
                tn  = max(seg_start + 0.5 * avg_T, ts[0], ts[pk[0]] - avg_T)
                idx = int(np.argmin(np.abs(ts - tn)))
                if idx >= pk[0] or idx <= 0: break
                pk = np.sort(np.unique(np.append(pk, idx))); added_head = True
            if added_head: interior_syn.append((ts[pk[0]], frt))
            lrt = ts[pk[-1]]
            while seg_end - ts[pk[-1]] > avg_T:
                tn  = min(seg_end - 0.5 * avg_T, ts[-1], ts[pk[-1]] + avg_T)
                idx = int(np.argmin(np.abs(ts - tn)))
                if idx <= pk[-1] or idx >= len(ts) - 1: break
                pk = np.sort(np.unique(np.append(pk, idx))); added_tail = True
            if added_tail: interior_syn.append((lrt, ts[pk[-1]]))

        return pk, vl, added_head, added_tail, interior_syn

    def env_norm(tr, sr, tpk, spk, tvl, svl):
        upper = np.interp(tr, tpk, spk)
        lower = np.interp(tr, tvl, svl)
        d = np.where(upper - lower < 1e-9, 1e-9, upper - lower)
        return np.clip(2.0 * (sr - lower) / d - 1.0, -1.0, 1.0)

    def gen_synth(tg, T_start, T_end=None, start_type='peak', end_type='peak', valley_ref=-1.0):
        if T_end is None: T_end = T_start
        dur = tg[-1] - tg[0]
        if dur <= 0 or len(tg) < 2:
            return np.full(len(tg), 1.0 if start_type == 'peak' else valley_ref)
        T_hm = 2 * T_start * T_end / (T_start + T_end)
        if start_type == end_type:
            delta = 2 * np.pi * max(1, round(dur / T_hm))
        else:
            nh = max(1, round(2 * (dur / T_hm) - 1))
            if nh % 2 == 0: nh += 1
            delta = np.pi * nh
        w0, w1  = 2 * np.pi / T_start, 2 * np.pi / T_end
        a       = (tg - tg[0]) / dur
        phi_raw = dur * (w0 * a + (w1 - w0) * a ** 2 / 2)
        phi     = phi_raw * (delta / phi_raw[-1]) if phi_raw[-1] > 0 else phi_raw
        y       = np.cos(phi) if start_type == 'peak' else -np.cos(phi)
        mid, amp = (1.0 + valley_ref) / 2, (1.0 - valley_ref) / 2
        return mid + amp * y

    segs = []
    for _, row in df_segments.iterrows():
        s, e       = row["start_time"], row["end_time"]
        ts_s, ss_s = proc_iso(s, e)
        if len(ts_s) < 10: segs.append(None); continue
        pk, vl, fh, ft, int_syn = detect(ts_s, ss_s, ts_s[0], ts_s[-1])
        if len(pk) < 2 or len(vl) < 1: segs.append(None); continue
        tlo, thi = ts_s[pk[0]], ts_s[pk[-1]]
        mpp = (ts_s >= tlo) & (ts_s <= thi)
        if mpp.sum() < 5: segs.append(None); continue
        tr, sr   = ts_s[mpp], ss_s[mpp]
        pkr      = pk[(ts_s[pk] >= tlo) & (ts_s[pk] <= thi)]
        vlr      = vl[(ts_s[vl] >= tlo) & (ts_s[vl] <= thi)]
        if len(pkr) < 2 or len(vlr) < 1: segs.append(None); continue
        try:
            sn = env_norm(tr, sr, ts_s[pkr], ss_s[pkr], ts_s[vlr], ss_s[vlr])
        except Exception: segs.append(None); continue

        def _rep(t0l, t1l):
            ms = (tr >= t0l) & (tr <= t1l)
            if ms.sum() < 2: return
            mT = float(np.median(np.diff(ts_s[pkr]))) if len(pkr) > 1 else (t1l - t0l)
            sn[ms] = gen_synth(tr[ms], mT)

        for t0l, t1l in int_syn: _rep(t0l, t1l)
        if len(pkr) >= 2:
            amp = ss_s[pkr]
            for i in range(len(pkr) - 1):
                lo, hi2 = min(abs(amp[i]), abs(amp[i+1])), max(abs(amp[i]), abs(amp[i+1]))
                if lo > 0 and hi2 / lo > 1.8: _rep(ts_s[pkr[i]], ts_s[pkr[i+1]])
        for i in range(len(pkr) - 1):
            t0l, t1l = ts_s[pkr[i]], ts_s[pkr[i+1]]
            mc = (tr >= t0l) & (tr <= t1l)
            if mc.sum() < 4: continue
            alpha = (tr[mc] - tr[mc][0]) / (tr[mc][-1] - tr[mc][0] + 1e-12)
            if np.sqrt(np.mean((sn[mc] - np.cos(2*np.pi*alpha))**2)) > 0.25:
                _rep(t0l, t1l)

        pk_ivl = np.diff(ts_s[pkr])
        T_med  = float(np.median(pk_ivl)) if len(pk_ivl) else 0.15
        N_loc  = min(5, len(pk_ivl))
        p_head = float(np.median(pk_ivl[:N_loc])) if len(pk_ivl) else T_med
        p_tail = float(np.median(pk_ivl[-N_loc:])) if len(pk_ivl) else T_med
        p_head = np.clip(p_head, 0.85*T_med, 1.15*T_med)
        p_tail = np.clip(p_tail, 0.85*T_med, 1.15*T_med)
        s_corr = tlo - 0.5*T_med if (tlo-s < 0.25*T_med or tlo-s > 0.75*T_med) else s
        e_corr = thi + 0.5*T_med if (e-thi < 0.25*T_med or e-thi > 0.75*T_med) else e
        segs.append({"t": tr, "sig": sn, "period": T_med,
                     "period_head": p_head, "period_tail": p_tail,
                     "t_s": tlo, "t_e": thi,
                     "label_start": s_corr, "label_end": e_corr})

    valid = [s for s in segs if s is not None]
    n     = len(t)
    sig_rc, filled, is_syn = np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool)

    for seg in valid:
        m_real = (t >= seg["t_s"]) & (t <= seg["t_e"])
        sig_rc[m_real] = np.interp(t[m_real], seg["t"], seg["sig"])
        filled[m_real] = True
        if seg["t_s"] > seg["label_start"]:
            m = (t >= seg["label_start"]) & (t < seg["t_s"])
            if m.sum() >= 2:
                sig_rc[m] = gen_synth(t[m], seg["period_head"], start_type='valley', end_type='peak')
                filled[m] = is_syn[m] = True
        if seg["label_end"] > seg["t_e"]:
            m = (t > seg["t_e"]) & (t <= seg["label_end"])
            if m.sum() >= 2:
                sig_rc[m] = gen_synth(t[m], seg["period_tail"], start_type='peak', end_type='valley')
                filled[m] = is_syn[m] = True

    for i in range(len(valid) - 1):
        a, b   = valid[i], valid[i+1]
        gs, ge = a["label_end"], b["label_start"]
        if ge <= gs: continue
        m = (t > gs) & (t < ge)
        if m.sum() < 2: continue
        sig_rc[m] = gen_synth(t[m], a["period_tail"], b["period_head"],
                              start_type='valley', end_type='valley')
        filled[m] = is_syn[m] = True

    sig_rp = np.where(filled, sig_rc, np.nan)
    sig_sp = np.where(is_syn,  sig_rc, np.nan)
    gaps   = [(valid[i]["t_e"], valid[i+1]["t_s"], valid[i], valid[i+1])
              for i in range(len(valid)-1) if valid[i+1]["t_s"] > valid[i]["t_e"]]
    return sig_rp, sig_sp, filled, is_syn, valid, gaps


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
        self.init_ppg_selection()
        self.update_checkbox_labels()
        if self.analysis_result:
            self.update_graphs()

    def update_checkbox_labels(self):
        if self.analysis_result and 'roi_names' in self.analysis_result:
            roi_names = self.analysis_result['roi_names']
            for i, name in enumerate(roi_names):
                if i < 3:
                    self.chks[i].setText(name)

    def init_ppg_selection(self):
        if not self.analysis_result:
            self.group_ppg_select.setVisible(False)
            return

        res = self.analysis_result
        has_ir = len(res.get('y_sensor_ir', [])) > 0
        has_red = len(res.get('y_sensor_red', [])) > 0
        
        import os
        import pandas as pd
        hr_seg_path = os.path.join(self.folder_path, "hr_segments.csv")
        self.has_hr_segments = os.path.exists(hr_seg_path)
        
        self.radio_ppg_ir.setVisible(has_ir)
        self.radio_ppg_red.setVisible(has_red)
        self.radio_ppg_recon.setVisible(self.has_hr_segments)
        
        if not res.get('has_sensor', False) or (not has_ir and not has_red):
            self.group_ppg_select.setVisible(False)
            return
            
        self.group_ppg_select.setVisible(True)
        
        default_channel = "IR"
        if self.has_hr_segments:
            try:
                self.df_segments = pd.read_csv(hr_seg_path)
                if len(self.df_segments) > 0 and 'channel' in self.df_segments.columns:
                    first_chan = str(self.df_segments['channel'].iloc[0]).upper()
                    if "RED" in first_chan:
                        default_channel = "RED"
            except Exception as e:
                print(f"Error reading hr_segments.csv: {e}")
                self.df_segments = None
                self.has_hr_segments = False
                self.radio_ppg_recon.setVisible(False)
        else:
            self.df_segments = None
            
        if default_channel == "RED" and has_red:
            self.radio_ppg_red.setChecked(True)
        else:
            if has_ir:
                self.radio_ppg_ir.setChecked(True)
            elif has_red:
                self.radio_ppg_red.setChecked(True)

    def on_ppg_type_changed(self, btn_id, checked):
        if checked:
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

        # PPG 데이터 선택
        self.group_ppg_select = QGroupBox("PPG 데이터 선택")
        ppg_sel_layout = QHBoxLayout()
        self.radio_ppg_ir = QRadioButton("Raw IR")
        self.radio_ppg_red = QRadioButton("Raw RED")
        self.radio_ppg_recon = QRadioButton("보간 신호 (Recon)")
        
        self.bg_ppg_type = QButtonGroup()
        self.bg_ppg_type.addButton(self.radio_ppg_ir, 0)
        self.bg_ppg_type.addButton(self.radio_ppg_red, 1)
        self.bg_ppg_type.addButton(self.radio_ppg_recon, 2)
        
        ppg_sel_layout.addWidget(self.radio_ppg_ir)
        ppg_sel_layout.addWidget(self.radio_ppg_red)
        ppg_sel_layout.addWidget(self.radio_ppg_recon)
        self.group_ppg_select.setLayout(ppg_sel_layout)
        top_layout.addWidget(self.group_ppg_select)
        
        self.bg_ppg_type.idToggled.connect(self.on_ppg_type_changed)

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

    def reconstruct_ppg_signal(self, t_sensor, y_sensor, fs, bpf_lo, bpf_hi):
        """모듈 레벨 reconstruct_ppg_signal 을 호출하는 래퍼."""
        res = self.analysis_result
        return reconstruct_ppg_signal(
            t_sensor, y_sensor, fs, bpf_lo, bpf_hi,
            df_segments=self.df_segments,
            t0=res.get('t0', 0),
            t0_sensor=res.get('t0_sensor', 0)
        )


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

        # ROI 이름
        roi_names = res.get('roi_names', [f"ROI {x+1}" for x in range(roi_count)])

        color_roi = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_bg  = (200, 0, 200)
        color_ppg = (200, 200, 200)

        # 현재 X축 범위 저장 (채널 전환 시 뷰 유지)
        _saved_x_range = None
        for _p in self.plots:
            try:
                if _p.plotItem.scene() is not None:
                    _saved_x_range = _p.viewRange()[0]
                    break
            except Exception:
                pass

        self.graph_layout.clear()

        self.max_time = t_cam_plot[-1] if len(t_cam_plot) > 0 else 1.0

        active_plots = []

        # ── 1. ROI Raw (체크된 것만) ──────────────────────────────────
        for i in range(roi_count):
            if i < 3 and self.chks[i].isChecked():
                p = self.plots[i]
                p.clear()
                self.graph_layout.addItem(p)
                self.graph_layout.nextRow()
                p.plot(t_cam_plot, raw_rgb[i, :, ch_idx],
                       pen=pg.mkPen(color_roi[i], width=2))
                # 동적 라벨 사용
                p.setTitle(f"{roi_names[i]} Raw [{ch_name}]")
                p.enableAutoRange(axis=pg.ViewBox.YAxis)
                active_plots.append(p)

        # ── 2. PPG Raw / Recon (체크된 경우) ──────────────────────────
        if has_sensor and self.chk_ppg.isChecked():
            p_ppg = self.plots[4]
            p_ppg.clear()
            # Remove any previous region items
            for item in list(p_ppg.items):
                if isinstance(item, pg.LinearRegionItem):
                    p_ppg.removeItem(item)

            self.graph_layout.addItem(p_ppg)
            self.graph_layout.nextRow()
            
            t_sensor = res['t_sensor']
            t_sensor_plot = t_sensor - t_cam[0] if len(t_sensor) > 0 else t_sensor
            fs_sensor = (1.0 / np.mean(np.diff(t_sensor)) if len(t_sensor) > 1 else 60.0)
            
            ppg_type = self.bg_ppg_type.checkedId()
            
            if ppg_type == 0: # Raw IR
                y_data = res.get('y_sensor_ir', res.get('y_sensor', []))
                p_ppg.plot(t_sensor_plot, y_data, pen=pg.mkPen(color_ppg, width=2))
                p_ppg.setTitle("PPG Raw Sensor Data (IR)")
                p_ppg.enableAutoRange(axis=pg.ViewBox.YAxis)
                active_plots.append(p_ppg)
            elif ppg_type == 1: # Raw RED
                y_data = res.get('y_sensor_red', [])
                p_ppg.plot(t_sensor_plot, y_data, pen=pg.mkPen((255, 100, 100), width=2))
                p_ppg.setTitle("PPG Raw Sensor Data (RED)")
                p_ppg.enableAutoRange(axis=pg.ViewBox.YAxis)
                active_plots.append(p_ppg)
            elif ppg_type == 2 and self.has_hr_segments: # Reconstructed
                chan = "IR"
                if len(self.df_segments) > 0 and 'channel' in self.df_segments.columns:
                    chan = str(self.df_segments['channel'].iloc[0]).upper()
                
                if "RED" in chan:
                    y_raw = res.get('y_sensor_red', [])
                else:
                    y_raw = res.get('y_sensor_ir', res.get('y_sensor', []))
                
                if len(y_raw) > 0:
                    sig_rp, sig_sp, filled, is_syn, valid_segs, gaps = self.reconstruct_ppg_signal(
                        t_sensor, y_raw, fs_sensor, low_cut, high_cut
                    )
                    
                    p_ppg.plot(t_sensor_plot, sig_rp, pen=pg.mkPen('#1f77b4', width=2), connect='finite', name='실신호')
                    p_ppg.plot(t_sensor_plot, sig_sp, pen=pg.mkPen('#ff7f0e', width=2, style=Qt.PenStyle.DashLine), connect='finite', name='합성')
                    p_ppg.setTitle(f"PPG Reconstructed Signal ({chan})")
                    p_ppg.setYRange(-1.6, 1.6)
                    active_plots.append(p_ppg)
                    
                    t0 = res.get('t0', 0)
                    t0_sensor = res.get('t0_sensor', 0)
                    t_cam_0 = t_cam[0] if len(t_cam) > 0 else 0
                    dt_offset = t0_sensor - t0 - t_cam_0
                    
                    # 1. Real/Label areas: green/blue
                    for seg in valid_segs:
                        s_plot = seg['label_start'] + dt_offset
                        e_plot = seg['label_end'] + dt_offset
                        r_item = pg.LinearRegionItem(values=[s_plot, e_plot], movable=False,
                                                    brush=pg.mkBrush(30, 180, 60, 40), pen=None)
                        p_ppg.addItem(r_item)
                        
                    # 2. Gap/Interpolated areas: red
                    for gs, ge, *_ in gaps:
                        s_plot = gs + dt_offset
                        e_plot = ge + dt_offset
                        r_item = pg.LinearRegionItem(values=[s_plot, e_plot], movable=False,
                                                    brush=pg.mkBrush(220, 30, 30, 45), pen=None)
                        p_ppg.addItem(r_item)

        # ── 3. Background Raw (체크된 경우) ──────────────────────────
        if has_bg and self.chk_bg.isChecked():
            p_bg = self.plots[5]
            p_bg.clear()
            self.graph_layout.addItem(p_bg)
            self.graph_layout.nextRow()
            p_bg.plot(t_cam_plot, raw_rgb[roi_count, :, ch_idx],
                      pen=pg.mkPen(color_bg, width=2))
            p_bg.setTitle(f"Background Raw [{ch_name}]")
            p_bg.enableAutoRange(axis=pg.ViewBox.YAxis)
            active_plots.append(p_bg)

        # ── 4. Filtered All Signals (맨 아래, 체크된 신호 전부) ──────
        if self.chk_filt.isChecked():
            p_filt = self.plots[3]
            p_filt.clear()
            self.graph_layout.addItem(p_filt)
            self.graph_layout.nextRow()
            p_filt.setTitle(f"Filtered Signals [{ch_name}]  (z-score normalized)")
            p_filt.addLegend()

            def _plot_filtered(x_plot, y_raw, fs, color, label):
                """신호 평균 제거 → 밴드패스 → std 정규화 → 피크 검출 후 플롯.
                모든 신호를 동일 스케일(단위: std)로 정규화하여 진폭 비교 가능."""
                y_c = y_raw - np.mean(y_raw)
                try:
                    y_f = self.butter_bandpass_filter(y_c, low_cut, high_cut, fs)
                    # std 정규화: 모든 신호를 동일 진폭 스케일로
                    sigma = np.std(y_f)
                    y_norm = y_f / sigma if sigma > 0 else y_f
                    min_d = int(60.0 / bpm_max * fs)
                    peaks, _ = find_peaks(y_norm, distance=max(1, min_d),
                                          prominence=0.5)  # 정규화 후 고정 prominence
                    bpm = (60.0 / np.mean(np.diff(x_plot[peaks]))
                           if len(peaks) > 1 else 0)
                    p_filt.plot(x_plot, y_norm,
                                pen=pg.mkPen(color, width=2),
                                name=f"{label} (BPM: {bpm:.1f})")
                    if len(peaks) > 0:
                        sc = pg.ScatterPlotItem(
                            x=x_plot[peaks], y=y_norm[peaks],
                            pen=None, brush=pg.mkBrush(color), size=8, symbol='x')
                        p_filt.addItem(sc)
                except Exception:
                    pass

            # ROI 신호 (체크된 것만)
            for i in range(roi_count):
                if i < 3 and not self.chks[i].isChecked():
                    continue
                _plot_filtered(t_cam_plot, raw_rgb[i, :, ch_idx],
                               fs_cam, color_roi[i], roi_names[i])


            # 배경 신호 (체크된 경우)
            if has_bg and self.chk_bg.isChecked():
                _plot_filtered(t_cam_plot, raw_rgb[roi_count, :, ch_idx],
                               fs_cam, color_bg, "Background")

            # PPG 센서 신호 (체크된 경우, 센서 자체 샘플링 주파수 사용)
            if has_sensor and self.chk_ppg.isChecked():
                t_sensor = res['t_sensor']
                t_sensor_plot = t_sensor - t_cam[0] if len(t_sensor) > 0 else t_sensor
                fs_sensor = (1.0 / np.mean(np.diff(t_sensor))
                             if len(t_sensor) > 1 else fs_cam)
                
                ppg_type = self.bg_ppg_type.checkedId()
                if ppg_type == 2 and self.has_hr_segments:
                    # 보간 신호인 경우: 이미 필터링/보간 되었으므로 별도 필터링 없이 그대로 표시
                    chan = "IR"
                    if len(self.df_segments) > 0 and 'channel' in self.df_segments.columns:
                        chan = str(self.df_segments['channel'].iloc[0]).upper()
                    if "RED" in chan:
                        y_raw = res.get('y_sensor_red', [])
                    else:
                        y_raw = res.get('y_sensor_ir', res.get('y_sensor', []))
                    
                    if len(y_raw) > 0:
                        sig_rp, sig_sp, filled, is_syn, valid_segs, gaps = self.reconstruct_ppg_signal(
                            t_sensor, y_raw, fs_sensor, low_cut, high_cut
                        )
                        valid_mask = ~np.isnan(sig_rp)
                        if valid_mask.any():
                            y_norm = sig_rp.copy()
                            # std로 나누기 (다른 신호들과 z-score 스케일 정렬)
                            sigma = np.nanstd(y_norm)
                            if sigma > 0:
                                y_norm = y_norm / sigma
                            
                            min_d = int(60.0 / bpm_max * fs_sensor)
                            peaks, _ = find_peaks(np.nan_to_num(y_norm), distance=max(1, min_d), prominence=0.5)
                            bpm = (60.0 / np.mean(np.diff(t_sensor_plot[peaks]))
                                   if len(peaks) > 1 else 0)
                            
                            p_filt.plot(t_sensor_plot, y_norm,
                                        pen=pg.mkPen(color_ppg, width=2),
                                        name=f"PPG Sensor (Recon) (BPM: {bpm:.1f})",
                                        connect='finite')
                            if len(peaks) > 0:
                                sc = pg.ScatterPlotItem(
                                    x=t_sensor_plot[peaks], y=y_norm[peaks],
                                    pen=None, brush=pg.mkBrush(color_ppg), size=8, symbol='x')
                                p_filt.addItem(sc)
                else:
                    if ppg_type == 1:
                        y_val = res.get('y_sensor_red', [])
                        lbl = "PPG Sensor (RED)"
                    else:
                        y_val = res.get('y_sensor_ir', res.get('y_sensor', []))
                        lbl = "PPG Sensor (IR)"
                    _plot_filtered(t_sensor_plot, y_val,
                                   fs_sensor, color_ppg, lbl)

            p_filt.enableAutoRange(axis=pg.ViewBox.YAxis)
            active_plots.append(p_filt)

        # X축 링크
        for i, p in enumerate(active_plots):
            p.setXLink(active_plots[0] if i > 0 else None)

        # X축 슬라이더/스크롤바 설정 
        # 저장된 범위가 있으면 zoom/pan 선령 유지, 없으면 전체 보기
        if _saved_x_range is not None:
            # 슬라이더는 그대로두고 X범위만 돌려놈음
            if len(active_plots) > 0:
                active_plots[0].setXRange(_saved_x_range[0], _saved_x_range[1], padding=0)
        else:
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
        
        # Get the selected PPG type
        ppg_type = self.bg_ppg_type.checkedId()
        
        # Determine the signal to save
        if ppg_type == 2 and self.has_hr_segments:
            chan = "IR"
            if len(self.df_segments) > 0 and 'channel' in self.df_segments.columns:
                chan = str(self.df_segments['channel'].iloc[0]).upper()
            if "RED" in chan:
                y_raw = res.get('y_sensor_red', [])
            else:
                y_raw = res.get('y_sensor_ir', res.get('y_sensor', []))
            
            if len(y_raw) > 0:
                fs_sensor = (1.0 / np.mean(np.diff(t_sensor)) if len(t_sensor) > 1 else 60.0)
                low_cut = res['bpm_min'] / 60.0
                high_cut = res['bpm_max'] / 60.0
                sig_rp, sig_sp, filled, is_syn, valid_segs, gaps = self.reconstruct_ppg_signal(
                    t_sensor, y_raw, fs_sensor, low_cut, high_cut
                )
                y_sensor_to_save = sig_rp
            else:
                y_sensor_to_save = res['y_sensor']
        elif ppg_type == 1: # RED
            y_sensor_to_save = res.get('y_sensor_red', [])
        else: # IR / Default
            y_sensor_to_save = res.get('y_sensor_ir', res.get('y_sensor', []))
        
        mapped_ppg = []
        if has_sensor and len(t_sensor) > 0 and len(y_sensor_to_save) > 0:
            for tc in t_cam:
                idx = (np.abs(t_sensor - tc)).argmin()
                mapped_ppg.append(y_sensor_to_save[idx])
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

        # pyqtgraph는 BGR 해석 (Fix Log: 코드 48 = BayerRG2BGR)
        # BGR8만 tifffile이 RGB로 태깅하여 읽을 때 채널이 뒤집히므로 변환 필요
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)  # 코드 48
        else:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR8 전용

        # 히스토그램 계산용 RGB (img_display와 동일)
        self.current_img_rgb = img_display

        img_pg = np.transpose(img_display, (1, 0, 2))
        # autoLevels=False: 매 프레임 자동 정규화 비활성화 → 플리커 방지
        levels = (0, 255) if img_display.dtype == np.uint8 else (0, 4095)
        self.image_view.setImage(img_pg, autoRange=False, autoLevels=False, levels=levels)

        if not hasattr(self, 'x_range_initialized'):
            self.spin_x_min.blockSignals(True)
            self.spin_x_max.blockSignals(True)

            self.spin_x_min.setValue(0)
            if self.current_img_rgb.dtype == np.uint8:
                self.spin_x_max.setValue(256)
            else:
                img_max = self.current_img_rgb.max()
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
    def __init__(self, no_sam3=False):
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
        
        self.analysis_result = None  # Worker에서 받은 결과 저장
        self.graph_popup = None       # 상세 그래프 팝업 인스턴스

        # SAM3 관련 속성
        self.no_sam3_arg = no_sam3
        self.has_sam3 = try_import_sam3(force_no=no_sam3)
        self.sam3_model = None
        self.sam3_processor = None
        self.autocast_ctx = None

        # Unet++ 관련 속성
        self.has_unetpp = try_import_unetpp()
        self.unetpp_model = None
        self.unetpp_transform = None

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

        # ROI 설정 모드
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("ROI 모드:"))
        self.combo_roi_mode = QComboBox()
        self.combo_roi_mode.addItems(["ROI 직접설정", "SAM3 세그먼테이션", "Unet++ 세그먼테이션"])
        mode_layout.addWidget(self.combo_roi_mode)
        set_layout.addLayout(mode_layout)
        self.combo_roi_mode.currentIndexChanged.connect(self.on_roi_mode_changed)

        # --- 4-1. ROI 직접설정 설정 위젯 ---
        self.widget_manual_settings = QWidget()
        manual_layout = QVBoxLayout(self.widget_manual_settings)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI 개수:"))
        self.spin_roi_count = QSpinBox()
        self.spin_roi_count.setRange(1, 3)
        self.spin_roi_count.setValue(3)
        roi_layout.addWidget(self.spin_roi_count)
        manual_layout.addLayout(roi_layout)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("배경 영역:"))
        self.spin_bg_count = QSpinBox()
        self.spin_bg_count.setRange(0, 1)
        self.spin_bg_count.setValue(1)
        bg_layout.addWidget(self.spin_bg_count)
        manual_layout.addLayout(bg_layout)

        self.btn_setup = QPushButton("설정 완료")
        self.btn_setup.setCheckable(True)
        self.btn_setup.clicked.connect(self.toggle_setup)
        manual_layout.addWidget(self.btn_setup)
        
        set_layout.addWidget(self.widget_manual_settings)

        # --- 4-2. SAM3 및 Unet++ 세그먼테이션 설정 위젯 ---
        self.widget_sam3_settings = QWidget()
        sam3_layout = QVBoxLayout(self.widget_sam3_settings)
        sam3_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_segment = QPushButton("세그먼테이션")
        self.btn_segment.clicked.connect(self.run_segmentation_preview)
        sam3_layout.addWidget(self.btn_segment)

        chk_layout = QHBoxLayout()
        self.chk_sam_tail = QCheckBox("tail")
        self.chk_sam_foot1 = QCheckBox("foot1")
        self.chk_sam_foot2 = QCheckBox("foot2")
        self.chk_sam_tail.setChecked(True)
        self.chk_sam_foot1.setChecked(True)
        self.chk_sam_foot2.setChecked(True)
        chk_layout.addWidget(self.chk_sam_tail)
        chk_layout.addWidget(self.chk_sam_foot1)
        chk_layout.addWidget(self.chk_sam_foot2)
        sam3_layout.addLayout(chk_layout)

        self.widget_sam_conf = QWidget()
        conf_layout = QHBoxLayout(self.widget_sam_conf)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.addWidget(QLabel("임계값:"))
        self.spin_sam_conf = QDoubleSpinBox()
        self.spin_sam_conf.setRange(0.1, 1.0)
        self.spin_sam_conf.setValue(0.3)
        self.spin_sam_conf.setSingleStep(0.05)
        conf_layout.addWidget(self.spin_sam_conf)
        sam3_layout.addWidget(self.widget_sam_conf)

        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("최소 크기:"))
        self.spin_sam_min_area = QSpinBox()
        self.spin_sam_min_area.setRange(5, 10000)
        self.spin_sam_min_area.setValue(100)
        area_layout.addWidget(self.spin_sam_min_area)
        sam3_layout.addLayout(area_layout)

        sam_bg_layout = QHBoxLayout()
        sam_bg_layout.addWidget(QLabel("배경 영역:"))
        self.spin_sam_bg_count = QSpinBox()
        self.spin_sam_bg_count.setRange(0, 1)
        self.spin_sam_bg_count.setValue(0)
        self.spin_sam_bg_count.valueChanged.connect(self.update_auto_seg_bg_roi)
        sam_bg_layout.addWidget(self.spin_sam_bg_count)
        sam3_layout.addLayout(sam_bg_layout)

        set_layout.addWidget(self.widget_sam3_settings)
        self.widget_sam3_settings.setVisible(False) # 기본값은 숨김

        # SAM3 및 Unet++ 사용 불가할 시 대응
        from PyQt6.QtGui import QStandardItemModel
        model = self.combo_roi_mode.model()
        if isinstance(model, QStandardItemModel):
            if not self.has_sam3:
                item = model.item(1)
                if item:
                    item.setEnabled(False)
            if not self.has_unetpp:
                item = model.item(2)
                if item:
                    item.setEnabled(False)

        if not self.has_sam3 and not self.has_unetpp:
            self.combo_roi_mode.setEnabled(False)
            self.combo_roi_mode.setToolTip("GPU 환경이 구축되지 않았거나 모델 패키지가 설치되지 않았습니다.")

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
        self.spin_bpm_max.setValue(500)
        self.spin_bpm_max.setMinimumWidth(60)
        bpm_layout.addWidget(self.spin_bpm_min)
        bpm_layout.addWidget(QLabel("~"))
        bpm_layout.addWidget(self.spin_bpm_max)
        set_layout.addLayout(bpm_layout)
        
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

        # PPG 데이터 선택 (RAW IR / RAW RED / 보간 신호)
        self.group_ppg_select = QGroupBox("PPG 데이터 선택")
        ppg_sel_layout = QHBoxLayout()
        self.radio_ppg_ir    = QRadioButton("Raw IR")
        self.radio_ppg_red   = QRadioButton("Raw RED")
        self.radio_ppg_recon = QRadioButton("보간 신호")
        self.radio_ppg_ir.setChecked(True)
        self.bg_ppg_type = QButtonGroup()
        self.bg_ppg_type.addButton(self.radio_ppg_ir,    0)
        self.bg_ppg_type.addButton(self.radio_ppg_red,   1)
        self.bg_ppg_type.addButton(self.radio_ppg_recon, 2)
        ppg_sel_layout.addWidget(self.radio_ppg_ir)
        ppg_sel_layout.addWidget(self.radio_ppg_red)
        ppg_sel_layout.addWidget(self.radio_ppg_recon)
        self.group_ppg_select.setLayout(ppg_sel_layout)
        self.group_ppg_select.setEnabled(False)
        center_layout.addWidget(self.group_ppg_select)
        self.bg_ppg_type.idToggled.connect(self._on_main_ppg_type_changed)
        self._main_df_segments = None  # hr_segments.csv DataFrame 캐시

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

        # pyqtgraph BGR 해석, Fix Log 코드 48 = BayerRG2BGR
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
        else:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR8 테스트로 확인

        img_pg = np.transpose(img_display, (1, 0, 2))
        # autoLevels=False: 매 프레임 자동 정규화 비활성화 → 플리커 방지
        levels = (0, 255) if img_display.dtype == np.uint8 else (0, 4095)
        self.image_view.setImage(img_pg, autoRange=False, autoLevels=False, levels=levels)

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
        
        # SAM3 또는 Unet++ 모드인지 확인하여 분석 시작 버튼 활성화 여부 조절
        is_auto_seg = (self.combo_roi_mode.currentIndex() in (1, 2))
        self.btn_analyze.setEnabled(is_auto_seg)
        if is_auto_seg:
            self.update_auto_seg_bg_roi()
        
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
    # SAM3 및 Unet++ 세그먼테이션 지원 메서드
    # -----------------------------------------------------------------
    def update_auto_seg_bg_roi(self):
        is_auto_seg = (self.combo_roi_mode.currentIndex() in (1, 2))
        has_bg = self.spin_sam_bg_count.value() > 0 if is_auto_seg else False
        
        if is_auto_seg and has_bg:
            if self.bg_roi_item is None:
                x_start, y_start = 100, 100
                default_size = 100
                if self.saved_bg_state:
                    x, y, w, h = self.saved_bg_state
                else:
                    x, y, w, h = x_start, y_start, default_size, default_size
                
                self.bg_roi_item = pg.RectROI([x, y], [w, h], pen=pg.mkPen('m', width=3))
                text_bg = pg.TextItem(f"Background\n({int(w)}x{int(h)})", color='m', anchor=(0, 1))
                text_bg.setParentItem(self.bg_roi_item)
                self.bg_roi_item.sigRegionChanged.connect(lambda r, t=text_bg: t.setText(f"Background\n({int(r.size()[0])}x{int(r.size()[1])})"))
                self.image_view.addItem(self.bg_roi_item)
        else:
            if self.bg_roi_item is not None:
                pos = self.bg_roi_item.pos()
                size = self.bg_roi_item.size()
                self.saved_bg_state = (pos[0], pos[1], size[0], size[1])
                self.image_view.removeItem(self.bg_roi_item)
                self.bg_roi_item = None

    def on_roi_mode_changed(self, index):
        is_auto_seg = (index == 1 or index == 2)
        self.widget_manual_settings.setVisible(not is_auto_seg)
        self.widget_sam3_settings.setVisible(is_auto_seg)
        
        if is_auto_seg:
            # 수동 ROI 박스 제거
            self.remove_roi_boxes()
            self.btn_setup.setChecked(False)
            self.btn_setup.setText("설정 완료")
            self.btn_analyze.setEnabled(True)
            # SAM3 모드(index 1)에서만 임계값 표시
            self.widget_sam_conf.setVisible(index == 1)
            # 세그먼테이션 모드용 배경 ROI 업데이트
            self.update_auto_seg_bg_roi()
        else:
            # 자동 세그먼테이션 모드용 배경 ROI 제거
            if self.bg_roi_item is not None:
                pos = self.bg_roi_item.pos()
                size = self.bg_roi_item.size()
                self.saved_bg_state = (pos[0], pos[1], size[0], size[1])
                self.image_view.removeItem(self.bg_roi_item)
                self.bg_roi_item = None
            self.btn_analyze.setEnabled(self.btn_setup.isChecked())

    def get_sam3_processor(self):
        if self.sam3_processor is None:
            self.lbl_status.setText("SAM3 모델 로딩 중...")
            QApplication.processEvents()
            
            import torch
            self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            
            try:
                self.sam3_model = sam3_model_builder(
                    device="cuda",
                    eval_mode=True,
                    load_from_HF=True,
                )
                self.sam3_processor = Sam3Processor(self.sam3_model, device="cuda", confidence_threshold=0.3)
                self.lbl_status.setText("SAM3 로드 완료")
            except Exception as e:
                err_msg = str(e)
                # 허깅페이스 토큰/인증/권한 에러 감지
                is_hf_auth_error = any(keyword in err_msg.lower() for keyword in ["401", "403", "gated", "token", "unauthorized", "login", "credential"])
                if is_hf_auth_error:
                    msg = (
                        "허깅페이스(Hugging Face) 인증 에러가 발생했습니다.\n\n"
                        "SAM3 모델을 처음 사용하려면 다음 단계를 완료해야 합니다:\n"
                        "1. 웹 브라우저에서 아래 페이지에 로그인하여 모델 이용 약관에 동의하세요:\n"
                        "   https://huggingface.co/facebook/sam3\n"
                        "2. 터미널(가상환경)에서 'huggingface-cli login'을 실행해 액세스 토큰으로 로그인하거나,\n"
                        "   환경 변수 HF_TOKEN에 토큰을 설정하세요.\n\n"
                        "이 문제가 해결되기 전까지는 'ROI 직접설정' 또는 'Unet++ 세그먼테이션' 모드를 사용해 주세요."
                    )
                    QMessageBox.warning(self, "허깅페이스 인증 필요", msg)
                    self.lbl_status.setText("SAM3 로드 실패 (인증 필요)")
                    raise HFAuthError("허깅페이스 인증 필요")
                else:
                    raise e
        return self.sam3_processor

    def get_unetpp_model(self):
        if self.unetpp_model is None:
            self.lbl_status.setText("Unet++ 모델 로딩 중...")
            QApplication.processEvents()
            
            import torch
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            import segmentation_models_pytorch as smp
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 1. 모델 아키텍처 빌드
            self.unetpp_model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b3",
                encoder_weights=None,
                in_channels=3,
                classes=3,
            )
            
            # 2. 모델 가중치 로드
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "modules", "unetpp", "best_unetpp_an_ft_202606081136.pth"
            )
            self.unetpp_model.load_state_dict(torch.load(model_path, map_location=device))
            self.unetpp_model.to(device).eval()
            
            # 3. 전처리 transform 정의
            self.unetpp_transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            self.lbl_status.setText("Unet++ 로드 완료")
        return self.unetpp_model, self.unetpp_transform

    def run_segmentation_preview(self):
        if not self.folder_path:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            return

        f_idx = self.current_frame_idx
        f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tiff")
        if not os.path.exists(f_path):
            f_path = os.path.join(self.frames_dir, f"frame_{f_idx:04d}.tif")
            if not os.path.exists(f_path):
                QMessageBox.warning(self, "경고", f"프레임 {f_idx} 파일을 찾을 수 없습니다.")
                return

        current_mode = self.combo_roi_mode.currentIndex() # 1: SAM3, 2: Unet++
        
        if current_mode == 1:
            self.lbl_status.setText("SAM3 세그먼테이션 추론 중...")
        elif current_mode == 2:
            self.lbl_status.setText("Unet++ 세그먼테이션 추론 중...")
        QApplication.processEvents()

        # 이미지 읽기 (BayerRG12 버그 수정 반영: cv2.COLOR_BayerBG2RGB)
        img = tifffile.imread(f_path) if f_path.lower().endswith(('.tiff', '.tif')) else cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img_rgb.dtype == np.uint16:
            img_rgb = (img_rgb / 16.0).clip(0, 255).astype(np.uint8)

        try:
            tail_mask = None
            foot1_mask = None
            foot2_mask = None
            min_area_val = self.spin_sam_min_area.value()

            if current_mode == 1:
                processor = self.get_sam3_processor()
                if processor is None:
                    raise Exception("SAM3 모델 로드 실패 (CUDA 장치 또는 Hugging Face 상태 확인 필요)")

                conf_val = self.spin_sam_conf.value()
                processor.confidence_threshold = conf_val

                from PIL import Image
                pil_image = Image.fromarray(img_rgb)

                # state 초기화 및 인코딩
                with self.autocast_ctx:
                    state = processor.set_image(pil_image)

                # tail
                if self.chk_sam_tail.isChecked():
                    with self.autocast_ctx:
                        state = processor.set_text_prompt(prompt="tail", state=state)
                    masks = state.get("masks")
                    if masks is not None and masks.shape[0] > 0:
                        combined_mask = np.any(masks.cpu().numpy(), axis=0)[0].astype(np.uint8)
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
                        valid_blobs = []
                        for i in range(1, num_labels):
                            area = stats[i, cv2.CC_STAT_AREA]
                            if area >= min_area_val:
                                valid_blobs.append((i, area))
                        if valid_blobs:
                            valid_blobs.sort(key=lambda x: x[1], reverse=True)
                            best_label = valid_blobs[0][0]
                            tail_mask = (labels == best_label).astype(np.uint8)

                processor.reset_all_prompts(state)
                with self.autocast_ctx:
                    state = processor.set_image(pil_image)

                # foot (최대 2개)
                if self.chk_sam_foot1.isChecked() or self.chk_sam_foot2.isChecked():
                    with self.autocast_ctx:
                        state = processor.set_text_prompt(prompt="foot", state=state)
                    masks = state.get("masks")
                    if masks is not None and masks.shape[0] > 0:
                        combined_mask = np.any(masks.cpu().numpy(), axis=0)[0].astype(np.uint8)
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
                        valid_blobs = []
                        for i in range(1, num_labels):
                            area = stats[i, cv2.CC_STAT_AREA]
                            if area >= min_area_val:
                                cx = centroids[i][0]
                                valid_blobs.append((i, area, cx))
                        if valid_blobs:
                            valid_blobs.sort(key=lambda x: x[1], reverse=True)
                            top_blobs = valid_blobs[:2]
                            if len(top_blobs) == 2:
                                top_blobs.sort(key=lambda x: x[2])
                                foot1_mask = (labels == top_blobs[0][0]).astype(np.uint8)
                                foot2_mask = (labels == top_blobs[1][0]).astype(np.uint8)
                            else:
                                foot1_mask = (labels == top_blobs[0][0]).astype(np.uint8)
            elif current_mode == 2:
                model, transform = self.get_unetpp_model()
                if model is None:
                    raise Exception("Unet++ 모델 로드 실패")

                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Albumentations transform 및 tensor 변환
                tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(tensor) # (1, C, H, W)
                    pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy() # (H, W)

                # tail (class 2)
                if self.chk_sam_tail.isChecked():
                    tail_raw_mask = (pred_mask == 2).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tail_raw_mask)
                    valid_blobs = []
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area >= min_area_val:
                            valid_blobs.append((i, area))
                    if valid_blobs:
                        valid_blobs.sort(key=lambda x: x[1], reverse=True)
                        best_label = valid_blobs[0][0]
                        tail_mask = (labels == best_label).astype(np.uint8)

                # foot (class 1)
                if self.chk_sam_foot1.isChecked() or self.chk_sam_foot2.isChecked():
                    foot_raw_mask = (pred_mask == 1).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foot_raw_mask)
                    valid_blobs = []
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area >= min_area_val:
                            cx = centroids[i][0]
                            valid_blobs.append((i, area, cx))
                    if valid_blobs:
                        valid_blobs.sort(key=lambda x: x[1], reverse=True)
                        top_blobs = valid_blobs[:2]
                        if len(top_blobs) == 2:
                            top_blobs.sort(key=lambda x: x[2])
                            foot1_mask = (labels == top_blobs[0][0]).astype(np.uint8)
                            foot2_mask = (labels == top_blobs[1][0]).astype(np.uint8)
                        else:
                            foot1_mask = (labels == top_blobs[0][0]).astype(np.uint8)

            # 오버레이 이미지 합성
            overlay_img = img_rgb.copy()
            alpha = 0.40
            labels_info = []

            # tail: 초록색 (60, 200, 60)
            if tail_mask is not None and np.sum(tail_mask) > 0 and self.chk_sam_tail.isChecked():
                overlay_img[tail_mask > 0] = (60, 200, 60)
                cy, cx = np.where(tail_mask > 0)
                labels_info.append(("tail", (int(np.mean(cx)), int(np.mean(cy)))))

            # foot1: 빨간색 (255, 60, 60)
            if foot1_mask is not None and np.sum(foot1_mask) > 0 and self.chk_sam_foot1.isChecked():
                overlay_img[foot1_mask > 0] = (255, 60, 60)
                cy, cx = np.where(foot1_mask > 0)
                labels_info.append(("foot1", (int(np.mean(cx)), int(np.mean(cy)))))

            # foot2: 주황색 (255, 150, 50)
            if foot2_mask is not None and np.sum(foot2_mask) > 0 and self.chk_sam_foot2.isChecked():
                overlay_img[foot2_mask > 0] = (255, 150, 50)
                cy, cx = np.where(foot2_mask > 0)
                labels_info.append(("foot2", (int(np.mean(cx)), int(np.mean(cy)))))

            preview_rgb = cv2.addWeighted(overlay_img, alpha, img_rgb, 1 - alpha, 0)

            # 텍스트 라벨 렌더링
            for text, pos in labels_info:
                tx, ty = pos
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(preview_rgb, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
                cv2.putText(preview_rgb, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            img_pg = np.transpose(preview_rgb, (1, 0, 2))
            levels = (0, 255) if preview_rgb.dtype == np.uint8 else (0, 4095)
            self.image_view.setImage(img_pg, autoRange=False, autoLevels=False, levels=levels)

            if current_mode == 1:
                self.lbl_status.setText("SAM3 세그먼테이션 미리보기 완료")
            else:
                self.lbl_status.setText("Unet++ 세그먼테이션 미리보기 완료")

        except HFAuthError:
            pass
        except Exception as e:
            QMessageBox.critical(self, "오류", f"세그먼테이션 중 오류 발생:\n{str(e)}")
            self.lbl_status.setText("세그먼테이션 오류")

    # -----------------------------------------------------------------
    # 분석 시작
    # -----------------------------------------------------------------
    def start_analysis(self):
        if self.btn_analyze.text() == "분석중":
            return # 이미 실행중

        roi_mode = "manual"
        active_sam3_rois = []
        rois_info = []
        bg_roi_info = None

        if self.combo_roi_mode.currentIndex() == 1:
            # SAM3 세그먼테이션 모드
            roi_mode = "sam3"
            if self.chk_sam_tail.isChecked():
                active_sam3_rois.append('tail')
            if self.chk_sam_foot1.isChecked():
                active_sam3_rois.append('foot1')
            if self.chk_sam_foot2.isChecked():
                active_sam3_rois.append('foot2')

            if not active_sam3_rois:
                QMessageBox.warning(self, "경고", "분석할 SAM3 ROI 영역을 적어도 하나 이상 체크하세요.")
                return

            if self.bg_roi_item:
                pos = self.bg_roi_item.pos()
                size = self.bg_roi_item.size()
                bg_roi_info = (pos[0], pos[1], size[0], size[1])

            # 분석 전에 SAM3 모델 로드 보장
            try:
                self.get_sam3_processor()
            except HFAuthError:
                return
            except Exception as e:
                QMessageBox.critical(self, "오류", f"SAM3 모델 로드 중 오류 발생:\n{str(e)}")
                self.lbl_status.setText("SAM3 로드 실패")
                return
        elif self.combo_roi_mode.currentIndex() == 2:
            # Unet++ 세그먼테이션 모드
            roi_mode = "unetpp"
            if self.chk_sam_tail.isChecked():
                active_sam3_rois.append('tail')
            if self.chk_sam_foot1.isChecked():
                active_sam3_rois.append('foot1')
            if self.chk_sam_foot2.isChecked():
                active_sam3_rois.append('foot2')

            if not active_sam3_rois:
                QMessageBox.warning(self, "경고", "분석할 Unet++ ROI 영역을 적어도 하나 이상 체크하세요.")
                return

            if self.bg_roi_item:
                pos = self.bg_roi_item.pos()
                size = self.bg_roi_item.size()
                bg_roi_info = (pos[0], pos[1], size[0], size[1])

            # 분석 전에 Unet++ 모델 로드 보장
            try:
                self.get_unetpp_model()
            except Exception as e:
                QMessageBox.critical(self, "오류", f"Unet++ 모델 로드 중 오류 발생:\n{str(e)}")
                self.lbl_status.setText("Unet++ 로드 실패")
                return
        else:
            # 수동 ROI 직접설정 모드
            roi_mode = "manual"
            for roi in self.roi_items:
                pos = roi.pos()
                size = roi.size()
                rois_info.append((pos[0], pos[1], size[0], size[1]))
                
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
            rois_info, bg_roi_info, bpm_min, bpm_max,
            roi_mode=roi_mode, sam3_model=self.sam3_model,
            active_sam3_rois=active_sam3_rois,
            sam_conf=self.spin_sam_conf.value(),
            sam_min_area=self.spin_sam_min_area.value(),
            unetpp_model=self.unetpp_model,
            unetpp_transform=self.unetpp_transform
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

        # PPG 선택 UI 활성화 및 hr_segments.csv / 채널 초기화
        has_ir  = len(result.get('y_sensor_ir', [])) > 0
        has_red = len(result.get('y_sensor_red', [])) > 0
        hr_seg_path = os.path.join(self.folder_path, "hr_segments.csv")
        has_seg = os.path.exists(hr_seg_path)
        if has_ir or has_red:
            self.group_ppg_select.setEnabled(True)
            self.radio_ppg_ir.setVisible(has_ir)
            self.radio_ppg_red.setVisible(has_red)
            self.radio_ppg_recon.setVisible(has_seg)
            if has_seg:
                try:
                    self._main_df_segments = pd.read_csv(hr_seg_path)
                    if 'channel' in self._main_df_segments.columns:
                        ch = str(self._main_df_segments['channel'].iloc[0]).upper()
                        if "RED" in ch and has_red:
                            self.radio_ppg_red.setChecked(True)
                        elif has_ir:
                            self.radio_ppg_ir.setChecked(True)
                except Exception as e:
                    print(f"[PPG] hr_segments.csv load error: {e}")
                    self._main_df_segments = None
            else:
                self._main_df_segments = None
                if has_ir:
                    self.radio_ppg_ir.setChecked(True)
                elif has_red:
                    self.radio_ppg_red.setChecked(True)
        else:
            self.group_ppg_select.setEnabled(False)
            self._main_df_segments = None

        # 경고 문구가 있으면 QMessageBox로 요약하여 안내
        warnings = result.get('warning_messages', [])
        if warnings:
            unique_warnings = {}
            for w in warnings:
                parts = w.split(":")
                if len(parts) >= 2:
                    msg = parts[1].strip()
                    unique_warnings[msg] = unique_warnings.get(msg, 0) + 1
            
            warning_text = "\n".join([f"- {msg} (총 {count}회)" for msg, count in unique_warnings.items()])
            QMessageBox.warning(self, "검출 실패 경고", f"분석 중 일부 프레임에서 객체 검출에 실패하여 직전 영역을 사용했습니다:\n\n{warning_text}")

        # 팝업이 열려 있으면 새 결과로 자동 갱신
        if self.graph_popup is not None and self.graph_popup.isVisible():
            self.graph_popup.analysis_result = result
            self.graph_popup.update_checkbox_labels()
            self.graph_popup.update_graphs()

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

        if self.graph_popup is not None and self.graph_popup.isVisible():
            # 이미 열린 팝업이 있으면 최신 데이터로 갱신 후 포커스
            self.graph_popup.analysis_result = self.analysis_result
            self.graph_popup.update_graphs()
            self.graph_popup.raise_()
            self.graph_popup.activateWindow()
        else:
            # 새 팝업 생성
            self.graph_popup = GraphPopup(
                self,
                analysis_result=self.analysis_result,
                folder_path=self.folder_path,
                spin_duration_val=duration_val,
                start_frame_idx=self.current_frame_idx
            )
            self.graph_popup.exec()
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

    def _on_main_ppg_type_changed(self, btn_id, checked):
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

        # ROI 이름
        roi_names = res.get('roi_names', [f"ROI {x+1}" for x in range(roi_count)])

        plot_colors = [(255,0,0), (0,255,0), (0,0,255)]

        # 현재 X축 범위 저장 (채널 전환 시 뷰 유지)
        _saved_x_range = None
        for _p in self.plots:
            try:
                if _p.plotItem.scene() is not None:
                    _saved_x_range = _p.viewRange()[0]
                    break
            except Exception:
                pass

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
                # 동적 ROI 타이틀 반영
                p.setTitle(f"{roi_names[i]} Raw [{ch_name}]")
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
                
                # 동적 ROI 범례 반영
                p_filt.plot(t_cam_plot, y_filt, pen=pg.mkPen(plot_colors[i], width=2), name=f"{roi_names[i]} (BPM: {bpm:.1f})")
                
                # Peaks scatter
                if len(peaks) > 0:
                    scatter = pg.ScatterPlotItem(x=t_cam_plot[peaks], y=y_filt[peaks], pen=None, brush=pg.mkBrush(plot_colors[i]), size=8, symbol='x')
                    p_filt.addItem(scatter)
                    
            except Exception as e:
                print(f"Filter error ROI {i}:", e)
        p_filt.enableAutoRange()
        active_plots.append(p_filt)

        # 5: PPG Data (Raw IR / Raw RED / 보간 신호)
        if has_sensor:
            p_ppg = self.plots[4]
            p_ppg.clear()
            for item in list(p_ppg.items):
                if isinstance(item, pg.LinearRegionItem):
                    p_ppg.removeItem(item)
            self.graph_layout.addItem(p_ppg)
            self.graph_layout.nextRow()
            t_sensor = res['t_sensor']
            t_sensor_plot = t_sensor - t_cam[0] if len(t_sensor) > 0 else t_sensor
            fs_sensor = (1.0 / np.mean(np.diff(t_sensor)) if len(t_sensor) > 1 else 60.0)

            ppg_type = self.bg_ppg_type.checkedId()

            if ppg_type == 1:  # Raw RED
                y_data = res.get('y_sensor_red', [])
                p_ppg.plot(t_sensor_plot, y_data, pen=pg.mkPen((255, 100, 100), width=2))
                p_ppg.setTitle("PPG Raw Sensor Data (RED)")
                p_ppg.enableAutoRange()
                active_plots.append(p_ppg)
            elif ppg_type == 2 and self._main_df_segments is not None:  # 보간 신호
                chan = "IR"
                if 'channel' in self._main_df_segments.columns:
                    chan = str(self._main_df_segments['channel'].iloc[0]).upper()
                y_raw = res.get('y_sensor_red', []) if "RED" in chan else res.get('y_sensor_ir', res.get('y_sensor', []))
                if len(y_raw) > 0:
                    sig_rp, sig_sp, filled, is_syn, valid_segs, gaps = reconstruct_ppg_signal(
                        t_sensor, y_raw, fs_sensor,
                        bpm_min / 60.0, bpm_max / 60.0,
                        df_segments=self._main_df_segments,
                        t0=res.get('t0', 0),
                        t0_sensor=res.get('t0_sensor', 0)
                    )
                    p_ppg.plot(t_sensor_plot, sig_rp, pen=pg.mkPen('#1f77b4', width=2), connect='finite')
                    p_ppg.plot(t_sensor_plot, sig_sp, pen=pg.mkPen('#ff7f0e', width=2, style=Qt.PenStyle.DashLine), connect='finite')
                    p_ppg.setTitle(f"PPG Reconstructed Signal ({chan})")
                    p_ppg.setYRange(-1.6, 1.6)
                    active_plots.append(p_ppg)

                    # 타임스탬프 오프셋
                    t0_v      = res.get('t0', 0)
                    t0_s_v    = res.get('t0_sensor', 0)
                    dt_offset = t0_s_v - t0_v - (t_cam[0] if len(t_cam) > 0 else 0)

                    # 라벨 영역: 초록
                    for seg in valid_segs:
                        s_pl = seg['label_start'] + dt_offset
                        e_pl = seg['label_end']   + dt_offset
                        p_ppg.addItem(pg.LinearRegionItem([s_pl, e_pl], movable=False,
                                                          brush=pg.mkBrush(30, 180, 60, 40), pen=None))
                    # 갭 영역: 빨강
                    for gs, ge, *_ in gaps:
                        s_pl = gs + dt_offset
                        e_pl = ge + dt_offset
                        p_ppg.addItem(pg.LinearRegionItem([s_pl, e_pl], movable=False,
                                                          brush=pg.mkBrush(220, 30, 30, 45), pen=None))
            else:  # Raw IR (default)
                y_data = res.get('y_sensor_ir', res.get('y_sensor', []))
                p_ppg.plot(t_sensor_plot, y_data, pen=pg.mkPen(200, 200, 200, width=2))
                p_ppg.setTitle("PPG Raw Sensor Data (IR)")
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

        # 저장된 X범위 복원 (채널 전환 시 등 재렌더링 후도 동일 범위 유지)
        if _saved_x_range is not None and len(active_plots) > 0:
            active_plots[0].setXRange(_saved_x_range[0], _saved_x_range[1], padding=0)

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

        fps = round(res['fs_cam'], 2)

        # ROI 이름 목록 가져오기
        roi_names = res.get('roi_names', [f"roi{x+1}" for x in range(roi_count)])

        rows = []
        for i, f_idx in enumerate(frame_indices):
            save_f_idx = i + 1
            # ROI 1~3 (동적 이름 사용)
            for r in range(roi_count):
                rows.append({
                    "Frame": save_f_idx,
                    "Type": roi_names[r],
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
    no_sam3 = "--no-sam3" in sys.argv
    if no_sam3:
        sys.argv.remove("--no-sam3")
    
    app = QApplication(sys.argv)
    window = MainWindow(no_sam3=no_sam3)
    window.show()
    sys.exit(app.exec())
