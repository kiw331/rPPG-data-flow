"""
파일명: extract_from_frames.py
설명: Basler 카메라 프레임 시퀀스를 PNG 이미지 데이터셋으로 가공 및 추출하는 GUI 도구.
      - frames 폴더의 TIFF 파일을 로드하여 미리보기, 재생, 탐색.
      - camera_summary.json 기반으로 픽셀 포맷(BayerRG8, BayerRG12, BGR8) 자동 감지 및 12비트 복원 처리.
      - 형제 폴더 이동 지원 (이전 폴더, 다음 폴더).
      - 전체 프레임 저장 및 크기조절·회전이 가능한 크롭 박스 모드 지원 (기본 128x128).
      - 샘플링 간격 설정 (기본 60프레임).
      - 현재 활성화된 단일 프레임 저장 기능.
      - 출력 비트 깊이(8비트/12비트) 설정 및 8비트->12비트 경고 메시지 지원.
      - 데이터셋 경로: data/mouse_segmentation/[폴더명]/[파일명]0001.png 형태로 저장 및 기존 번호 이어가기.
"""

import sys
import os
import json
import glob
import math
import re
import numpy as np
import cv2
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox, QDoubleSpinBox,
    QSizePolicy, QStatusBar, QRadioButton, QButtonGroup, QScrollArea,
    QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont


# ─────────────────────────── 카메라 요약 파일 및 디코더 ───────────────────────────

def load_camera_summary(folder):
    """지정 폴더 내 camera_summary.json 파일을 파싱"""
    path = os.path.join(folder, "camera_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"camera_summary.json 로드 실패: {e}")
        return None


def decode_frame(raw_img, pixel_format):
    """
    TIFF raw 이미지를 GUI 표시용 8비트 RGB numpy 배열로 변환.
    Bayer Pixel Format Fix Log.md 규칙 준수:
      - BayerRG12 / BayerRG8: 2D 배열 -> cv2.COLOR_BayerBG2RGB (코드 48, QImage RGB888 용)
      - BGR8: 3채널 배열 -> cv2.COLOR_BGR2RGB
    """
    pf = pixel_format.upper() if pixel_format else ""

    if raw_img.ndim == 2:
        # Bayer 이미지 (BayerRG8 또는 BayerRG12)
        if "12" in pf:
            # BayerRG12: uint16 12비트 -> 8비트로 스케일링
            img8 = (raw_img >> 4).astype(np.uint8)
        else:
            img8 = raw_img.astype(np.uint8)
        # 코드 48: COLOR_BayerBG2RGB (QImage RGB888 용)
        rgb = cv2.cvtColor(img8, cv2.COLOR_BayerBG2RGB)
        return rgb
    elif raw_img.ndim == 3:
        # BGR8: tifffile은 BGR 순서 그대로 읽어오므로 RGB로 변환
        rgb = cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb
    else:
        return raw_img.astype(np.uint8)


def rotate_crop(img, cx, cy, w, h, angle_deg):
    """이미지에서 (cx, cy) 중심, w×h 크기, angle_deg 회전된 영역을 회전 크롭"""
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
    img_h, img_w = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (img_w, img_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h

    # 경계 클램프
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(img_w, x2)
    y2c = min(img_h, y2)

    if x2c <= x1c or y2c <= y1c:
        return None

    crop = rotated[y1c:y2c, x1c:x2c]

    # 크롭 영역이 이미지 영역을 벗어나 크기가 다를 경우 검은색 패딩 적용
    if crop.shape[0] != h or crop.shape[1] != w:
        canvas = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        ox = x1c - x1
        oy = y1c - y1
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        return canvas

    return crop


def get_processed_frame_for_save(raw_img, pixel_format, bit_depth, crop_enabled, crop_params):
    """
    저장할 이미지를 포맷 규격에 맞게 최종 가공(디베이어링, 비트 레이트 스케일링, 크롭 포함)하여 반환.
    OpenCV imwrite() 저장을 위해 최종 출력은 BGR 순서로 생성됨.
    """
    pf = pixel_format.upper() if pixel_format else ""

    # 1. 픽셀 포맷 처리
    if raw_img.ndim == 2:
        # Bayer 이미지
        if "12" in pf:
            if bit_depth == 12:
                # 12비트 그대로 변환 (uint16 유지, imwrite로 16비트 PNG 저장)
                # 코드 46: COLOR_BayerBG2BGR (imwrite BGR 저장용)
                bgr = cv2.cvtColor(raw_img, cv2.COLOR_BayerBG2BGR)
            else:
                # 8비트 다운스케일링 변환
                img8 = (raw_img >> 4).astype(np.uint8)
                bgr = cv2.cvtColor(img8, cv2.COLOR_BayerBG2BGR)
        else:
            # 8비트 Bayer
            img8 = raw_img.astype(np.uint8)
            bgr = cv2.cvtColor(img8, cv2.COLOR_BayerBG2BGR)
    elif raw_img.ndim == 3:
        # BGR8 이미지 (이미 3채널 BGR 순서)
        bgr = raw_img.astype(np.uint8)
    else:
        bgr = raw_img.copy()

    # 2. 크롭 모드 적용
    if crop_enabled:
        cx = crop_params['cx']
        cy = crop_params['cy']
        w = crop_params['w']
        h = crop_params['h']
        angle = crop_params['angle']
        bgr = rotate_crop(bgr, cx, cy, w, h, angle)

    return bgr


def get_next_file_number(base_dir, file_prefix):
    """지정 디렉터리 내 [prefix][번호].png 형식의 파일 목록 중 최댓값을 가져와 다음 번호를 반환"""
    existing_files = glob.glob(os.path.join(base_dir, f"{file_prefix}*.png"))
    max_num = 0
    pattern = re.compile(rf"^{re.escape(file_prefix)}(\d+)\.png$")
    for f in existing_files:
        name = os.path.basename(f)
        match = pattern.match(name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


# ─────────────────────────── 이미지 뷰어 위젯 (크롭 오버레이) ───────────────────────────

class CropOverlayLabel(QLabel):
    """
    이미지 및 크롭 선택 박스를 오버레이하여 표시하는 QLabel 확장 클래스.
    - 드래그 앤 드롭으로 크롭 박스 이동.
    - Ctrl + 마우스 휠: 크롭 박스 크기 조절.
    - Alt + 마우스 휠: 크롭 박스 회전 각도 조절.
    - 마우스 휠: 프레임 탐색.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #11111b; border: 2px solid #313244; border-radius: 8px;")

        # 크롭 박스 상태 (이미지 좌표계 기준)
        self.crop_cx = 512.0
        self.crop_cy = 512.0
        self.crop_w = 128
        self.crop_h = 128
        self.crop_angle = 0.0  # degrees
        self.crop_mode_enabled = False

        self._pixmap = None
        self._img_w = 0
        self._img_h = 0

        # 드래그 관련 상태
        self._dragging = False
        self._drag_offset_x = 0.0
        self._drag_offset_y = 0.0

        # 콜백 함수들
        self.crop_changed_callback = None
        self.wheel_callback = None

    def set_image(self, rgb_array):
        """RGB Numpy array 데이터를 받아서 QLabel에 그릴 수 있도록 설정"""
        h, w, ch = rgb_array.shape
        self._img_w = w
        self._img_h = h
        bytes_per_line = ch * w
        qimg = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _get_display_rect(self):
        """위젯 안에서 pixmap이 비율을 유지하며 실제로 그려지는 영역과 스케일 팩터 반환"""
        if self._pixmap is None:
            return QRectF(0, 0, self.width(), self.height()), 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        dw, dh = pw * scale, ph * scale
        dx = (ww - dw) / 2
        dy = (wh - dh) / 2
        return QRectF(dx, dy, dw, dh), scale

    def _widget_to_img(self, pos):
        """위젯 로컬 좌표 -> 원본 이미지 좌표 변환"""
        rect, scale = self._get_display_rect()
        ix = (pos.x() - rect.x()) / scale
        iy = (pos.y() - rect.y()) / scale
        return ix, iy

    def _img_to_widget(self, ix, iy):
        """원본 이미지 좌표 -> 위젯 로컬 좌표 변환"""
        rect, scale = self._get_display_rect()
        wx = rect.x() + ix * scale
        wy = rect.y() + iy * scale
        return wx, wy

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e1e2e"))

        if self._pixmap:
            rect, scale = self._get_display_rect()
            painter.drawPixmap(int(rect.x()), int(rect.y()),
                               self._pixmap.scaled(int(rect.width()), int(rect.height()),
                                                   Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 크롭 모드일 때만 크롭 상자 그리기
            if self.crop_mode_enabled:
                cx_w, cy_w = self._img_to_widget(self.crop_cx, self.crop_cy)
                painter.save()
                painter.translate(cx_w, cy_w)
                painter.rotate(self.crop_angle)
                hw = self.crop_w * scale / 2
                hh = self.crop_h * scale / 2

                # 크롭 테두리선 그리기 (형광 그린)
                pen = QPen(QColor("#a6e3a1"), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(QRectF(-hw, -hh, hw * 2, hh * 2))

                # 중심 조준점 (빨간 점선)
                painter.setPen(QPen(QColor("#f38ba8"), 1, Qt.DashLine))
                painter.drawLine(QPointF(-hw, 0), QPointF(hw, 0))
                painter.drawLine(QPointF(0, -hh), QPointF(0, hh))

                # 정보 라벨링 텍스트
                painter.setPen(QColor("#a6e3a1"))
                font = QFont("Consolas", 10)
                painter.setFont(font)
                painter.drawText(QPointF(-hw + 4, -hh - 6),
                                 f"{self.crop_w}×{self.crop_h}  {self.crop_angle:.1f}°")
                painter.restore()

        painter.end()

    def mousePressEvent(self, event):
        if not self.crop_mode_enabled or self._pixmap is None:
            return

        if event.button() == Qt.LeftButton:
            ix, iy = self._widget_to_img(event.pos())
            # 박스 내부 또는 근처 클릭 여부 판별 (회전을 복원하여 원본 축정렬 영역에서 거리 판별)
            rad = math.radians(-self.crop_angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            dx = ix - self.crop_cx
            dy = iy - self.crop_cy
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a

            if abs(rx) <= self.crop_w / 2 + 10 and abs(ry) <= self.crop_h / 2 + 10:
                self._dragging = True
                self._drag_offset_x = dx
                self._drag_offset_y = dy
                self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self._pixmap:
            ix, iy = self._widget_to_img(event.pos())
            self.crop_cx = max(0, min(self._img_w, ix - self._drag_offset_x))
            self.crop_cy = max(0, min(self._img_h, iy - self._drag_offset_y))
            self.update()
            if self.crop_changed_callback:
                self.crop_changed_callback(self.crop_cx, self.crop_cy)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        delta = event.angleDelta().y()

        if modifiers == Qt.ControlModifier:
            # Ctrl + Wheel: 크기 조절 (8px 단위 동시 조절)
            step = 8 if delta > 0 else -8
            self.crop_w = max(16, min(self._img_w, self.crop_w + step))
            self.crop_h = max(16, min(self._img_h, self.crop_h + step))
            self.update()
            if self.crop_changed_callback:
                self.crop_changed_callback(self.crop_cx, self.crop_cy)
            event.accept()
        elif modifiers == Qt.AltModifier:
            # Alt + Wheel: 각도 조절 (5도 단위 회전)
            step = 5.0 if delta > 0 else -5.0
            self.crop_angle = (self.crop_angle + step) % 360.0
            if self.crop_angle > 180.0:
                self.crop_angle -= 360.0
            self.update()
            if self.crop_changed_callback:
                self.crop_changed_callback(self.crop_cx, self.crop_cy)
            event.accept()
        else:
            # 단독 마우스 휠: 프레임 탐색
            if self.wheel_callback:
                self.wheel_callback(delta)
                event.accept()


# ─────────────────────────── 메인 윈도우 애플리케이션 ───────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basler Frame Extractor & Cropper (rPPG 데이터 전처리 도구)")
        self.setMinimumSize(1200, 800)

        # 상태 변수
        self.current_folder = ""
        self.frames_list = []
        self.pixel_format = ""
        self.camera_info = {}
        self.current_idx = 0
        self.playing = False

        # UI 생성 및 연결
        self._build_ui()
        self._apply_theme()
        self._connect_signals()

        # 타이머 (재생 제어용)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        # 기본 예시 폴더 자동 로드 시도
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(script_dir, "data", "basler_0507", "20260507_155148")
        if os.path.exists(default_dir):
            self.load_folder(default_dir)

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # ─── 1. 왼쪽 영역 (프레임 뷰어 및 탐색 패널) ───
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # 프레임 뷰어
        self.viewer = CropOverlayLabel()
        self.viewer.crop_changed_callback = self._on_crop_changed_in_viewer
        self.viewer.wheel_callback = self._on_wheel_in_viewer
        left_panel.addWidget(self.viewer, 1)

        # 탐색 슬라이더 및 프레임 카운트 레이아웃
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.lbl_frame_idx = QLabel("0 / 0")
        self.lbl_frame_idx.setStyleSheet("color: #cdd6f4; font-family: 'Consolas', sans-serif; font-weight: bold;")
        self.lbl_frame_idx.setMinimumWidth(80)
        self.lbl_frame_idx.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_layout.addWidget(self.slider, 1)
        slider_layout.addWidget(self.lbl_frame_idx)
        left_panel.addLayout(slider_layout)

        # 이동 버튼 툴바
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)

        self.btn_open = QPushButton("📂 폴더 선택...")
        self.btn_prev_folder = QPushButton("◀ 이전 폴더")
        self.btn_next_folder = QPushButton("다음 폴더 ▶")

        self.btn_first = QPushButton("◀◀")
        self.btn_prev_frame = QPushButton("◀")
        self.btn_play = QPushButton("▶ 재생")
        self.btn_next_frame = QPushButton("▶")
        self.btn_last = QPushButton("▶▶")

        # 툴바 정렬
        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addSpacing(10)
        toolbar_layout.addWidget(self.btn_prev_folder)
        toolbar_layout.addWidget(self.btn_next_folder)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.btn_first)
        toolbar_layout.addWidget(self.btn_prev_frame)
        toolbar_layout.addWidget(self.btn_play)
        toolbar_layout.addWidget(self.btn_next_frame)
        toolbar_layout.addWidget(self.btn_last)
        toolbar_layout.addStretch(1)

        left_panel.addLayout(toolbar_layout)
        root_layout.addLayout(left_panel, 3)

        # ─── 2. 오른쪽 영역 (설정 패널) ───
        right_widget = QWidget()
        right_widget.setFixedWidth(340)
        right_panel = QVBoxLayout(right_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(10)

        # 스크롤 가능 구역으로 설정 패널 감싸기
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_widget = QWidget()
        scroll_widget.setObjectName("scroll_widget")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        # 영상 정보 Group
        grp_info = QGroupBox("영상 정보")
        info_form = QFormLayout()
        info_form.setSpacing(8)
        self.lbl_pixel_fmt = QLabel("-")
        self.lbl_resolution = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_total_frames = QLabel("-")
        self.lbl_duration = QLabel("-")
        
        # 값 라벨 고대비 코딩 스타일 적용
        for lbl in [self.lbl_pixel_fmt, self.lbl_resolution, self.lbl_fps, self.lbl_total_frames, self.lbl_duration]:
            lbl.setStyleSheet("color: #89b4fa; font-weight: bold; font-family: 'Consolas', sans-serif;")
            
        info_form.addRow("픽셀 포맷:", self.lbl_pixel_fmt)
        info_form.addRow("해상도:", self.lbl_resolution)
        info_form.addRow("프레임 레이트:", self.lbl_fps)
        info_form.addRow("총 프레임 수:", self.lbl_total_frames)
        info_form.addRow("녹화 시간:", self.lbl_duration)
        grp_info.setLayout(info_form)
        scroll_layout.addWidget(grp_info)

        # 저장 모드 설정 Group
        grp_mode = QGroupBox("저장 모드 설정")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rad_full_frame = QRadioButton("전체 프레임 저장 (전체 픽셀 사용)")
        self.rad_crop = QRadioButton("지정 영역 크롭 저장 (CROP 모드)")
        self.rad_full_frame.setChecked(True)

        self.mode_group.addButton(self.rad_full_frame)
        self.mode_group.addButton(self.rad_crop)
        mode_layout.addWidget(self.rad_full_frame)
        mode_layout.addWidget(self.rad_crop)
        grp_mode.setLayout(mode_layout)
        scroll_layout.addWidget(grp_mode)

        # 크롭 박스 크기 및 회전 설정 Group
        self.grp_crop = QGroupBox("크롭 영역 세부 설정")
        crop_form = QFormLayout()
        crop_form.setSpacing(8)

        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(16, 4096)
        self.spin_crop_w.setValue(128)
        self.spin_crop_w.setSuffix(" px")

        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(16, 4096)
        self.spin_crop_h.setValue(128)
        self.spin_crop_h.setSuffix(" px")

        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-180.0, 180.0)
        self.spin_angle.setSingleStep(1.0)
        self.spin_angle.setValue(0.0)
        self.spin_angle.setSuffix(" °")

        crop_form.addRow("크롭 너비:", self.spin_crop_w)
        crop_form.addRow("크롭 높이:", self.spin_crop_h)
        crop_form.addRow("회전 각도:", self.spin_angle)
        self.grp_crop.setLayout(crop_form)
        scroll_layout.addWidget(self.grp_crop)
        self.grp_crop.setEnabled(False)  # 초기 상태 (Full Frame 모드) 비활성화

        # 내보내기/샘플링 설정 Group
        grp_export = QGroupBox("가공 및 내보내기 설정")
        export_form = QFormLayout()
        export_form.setSpacing(8)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 9999)
        self.spin_sample.setValue(60)
        self.spin_sample.setSuffix(" 프레임")

        self.combo_bit_depth = QComboBox()
        self.combo_bit_depth.addItems(["8비트 (8-bit PNG)", "12비트 (12-bit / 16-bit PNG)"])
        self.combo_bit_depth.setCurrentIndex(0)

        export_form.addRow("샘플링 간격 설정:", self.spin_sample)
        export_form.addRow("저장 비트 깊이:", self.combo_bit_depth)
        grp_export.setLayout(export_form)
        scroll_layout.addWidget(grp_export)

        # 데이터셋 경로 설정 Group
        grp_dataset = QGroupBox("세그먼테이션 데이터셋 설정")
        dataset_form = QFormLayout()
        dataset_form.setSpacing(8)

        self.edit_folder = QLineEdit("folder_name")
        self.edit_folder.setPlaceholderText("예: 20260507_155148")
        self.edit_prefix = QLineEdit("frame_")

        dataset_form.addRow("저장 폴더명:", self.edit_folder)
        dataset_form.addRow("파일명 접두사:", self.edit_prefix)

        # 실시간 경로 가이드
        self.lbl_path_preview = QLabel("경로: data/mouse_segmentation/...")
        self.lbl_path_preview.setWordWrap(True)
        self.lbl_path_preview.setStyleSheet("color: #a6adc8; font-size: 11px;")
        dataset_form.addRow(self.lbl_path_preview)

        grp_dataset.setLayout(dataset_form)
        scroll_layout.addWidget(grp_dataset)

        scroll.setWidget(scroll_widget)
        right_panel.addWidget(scroll, 1)

        # 액션 버튼 영역
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)

        self.btn_save_current = QPushButton("📸 현재 프레임 저장")
        self.btn_save_current.setObjectName("btn_save_current")
        self.btn_save_current.setMinimumHeight(40)

        self.btn_export = QPushButton("💾 설정 조건으로 전체 시퀀스 추출")
        self.btn_export.setObjectName("btn_export")
        self.btn_export.setMinimumHeight(45)

        # 진행 표시줄
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background-color: #11111b;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
                width: 10px;
            }
        """)

        actions_layout.addWidget(self.btn_save_current)
        actions_layout.addWidget(self.btn_export)
        actions_layout.addWidget(self.progress_bar)
        right_panel.addLayout(actions_layout)

        root_layout.addWidget(right_widget)

        # 상태 바
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("준비 완료. 폴더를 열어주세요.")

    def _apply_theme(self):
        """Catppuccin Mocha 테마 스타일 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget#central {
                background-color: #1e1e2e;
            }
            QWidget#scroll_widget {
                background-color: #1e1e2e;
            }
            QScrollArea {
                background-color: #1e1e2e;
                border: none;
            }
            QWidget {
                color: #cdd6f4;
                font-family: "Segoe UI", "Malgun Gothic", sans-serif;
                font-size: 13px;
            }
            /* Card design for GroupBoxes */
            QGroupBox {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background-color: #181825;
                color: #89b4fa;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox:disabled {
                background-color: #11111b;
                border-color: #1e1e2e;
            }
            
            /* 설정 항목 (Labels): 밝고 선명한 라벤더 회색 볼드로 입력값과 뚜렷이 구분 */
            QLabel {
                color: #bac2de;
                font-weight: bold;
            }
            /* 비활성화된 그룹의 라벨 스타일 */
            QLabel:disabled {
                color: #585b70;
            }
            
            /* 입력창 (Inputs) 기본 스타일: 테두리를 좀 더 밝게(#585b70) 하여 활성화 상태 표시 */
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #313244;
                color: #ffffff; /* 활성화 시 밝은 흰색 */
                border: 1px solid #585b70;
                border-radius: 6px;
                padding: 5px 8px;
            }
            
            /* 스핀박스 내부 텍스트창 색상 지정 */
            QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QAbstractSpinBox QLineEdit, QAbstractSpinBox::lineEdit {
                color: #ffffff;
                background-color: transparent;
                border: none;
            }
            
            /* 비활성화(Disabled) 상태일 때의 스타일 지정 */
            QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled,
            QSpinBox:disabled QLineEdit, QDoubleSpinBox:disabled QLineEdit, QAbstractSpinBox:disabled QLineEdit {
                color: #585b70; /* 어두운 회색 */
                background-color: #181825;
                border-color: #313244;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
                border-color: #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #ffffff;
                selection-background-color: #45475a;
                border: 1px solid #45475a;
            }
            /* Radio buttons custom styling */
            QRadioButton {
                color: #cdd6f4;
                padding: 4px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid #45475a;
                background-color: #313244;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #a6e3a1;
                background-color: #a6e3a1;
            }
            QRadioButton::indicator:hover {
                border-color: #89b4fa;
            }
            /* Buttons */
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
            QPushButton:disabled {
                background-color: #181825;
                color: #585b70;
                border-color: #313244;
            }
            QPushButton#btn_save_current {
                background-color: #f9e2af;
                color: #1e1e2e;
                font-weight: bold;
                border: none;
            }
            QPushButton#btn_save_current:hover {
                background-color: #ffe082;
            }
            QPushButton#btn_export {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                border: none;
            }
            QPushButton#btn_export:hover {
                background-color: #94e2d5;
            }
            /* Slider & ScrollBar */
            QSlider::groove:horizontal {
                background-color: #313244;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #89b4fa;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #b4befe;
            }
            QStatusBar {
                background-color: #11111b;
                color: #a6adc8;
            }
            QScrollBar:vertical {
                background-color: #11111b;
                width: 12px;
                margin: 15px 3px 15px 3px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #45475a;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #585b70;
            }
        """)

    def _connect_signals(self):
        # 뷰어 관련 컨트롤
        self.btn_open.clicked.connect(self._open_folder_dialog)
        self.btn_prev_folder.clicked.connect(self.prev_folder)
        self.btn_next_folder.clicked.connect(self.next_folder)

        # 미디어 플레이어 관련
        self.btn_first.clicked.connect(lambda: self.slider.setValue(0))
        self.btn_last.clicked.connect(lambda: self.slider.setValue(len(self.frames_list) - 1))
        self.btn_prev_frame.clicked.connect(lambda: self._step_frame(-1))
        self.btn_next_frame.clicked.connect(lambda: self._step_frame(1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._show_frame)

        # 저장 모드 선택 전환에 따른 설정 상자 활성화/비활성화 처리
        self.rad_full_frame.toggled.connect(self._on_mode_toggled)
        self.rad_crop.toggled.connect(self._on_mode_toggled)

        # 크롭 세부 매개변수 실시간 동기화
        self.spin_crop_w.valueChanged.connect(self._update_crop_params_from_ui)
        self.spin_crop_h.valueChanged.connect(self._update_crop_params_from_ui)
        self.spin_angle.valueChanged.connect(self._update_crop_params_from_ui)

        # 경로 표시 가이드 업데이트
        self.edit_folder.textChanged.connect(self._update_path_preview)
        self.edit_prefix.textChanged.connect(self._update_path_preview)

        # 저장 동작 연동
        self.btn_save_current.clicked.connect(self._save_current_frame)
        self.btn_export.clicked.connect(self._export_crops)

    # ─── 탐색 슬라이더 및 재생 제어 ───

    def _step_frame(self, step):
        nxt = self.current_idx + step
        if 0 <= nxt < len(self.frames_list):
            self.slider.setValue(nxt)

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            self.btn_play.setText("▶ 재생")
        else:
            if not self.frames_list:
                return
            self.playing = True
            self.btn_play.setText("⏸ 정지")
            # camera_summary.json 에 명시된 FPS 기반으로 타이머 속도 설정 (기본값 30)
            fps = 30
            if self.camera_info:
                val = self.camera_info.get("FPS_Result", self.camera_info.get("FPS_Target", 30))
                if isinstance(val, (int, float)) and 0 < val <= 120:
                    fps = val
            self.play_timer.start(int(1000 / fps))

    def _on_play_tick(self):
        nxt = self.current_idx + 1
        if nxt >= len(self.frames_list):
            nxt = 0
        self.slider.setValue(nxt)

    def _show_frame(self, idx):
        if idx < 0 or idx >= len(self.frames_list):
            return
        self.current_idx = idx
        try:
            # tiff 프레임 로드
            raw = tifffile.imread(self.frames_list[idx])
            # 디비어 및 표시용 RGB 8비트 디코딩
            rgb = decode_frame(raw, self.pixel_format)
            self.viewer.set_image(rgb)
            self.lbl_frame_idx.setText(f"{idx} / {len(self.frames_list) - 1}")
        except Exception as e:
            print(f"프레임 {idx} 읽기 실패: {e}")

    # ─── 마우스 조작을 통한 크롭 상태 동기화 ───

    def _on_crop_changed_in_viewer(self, cx, cy):
        """뷰어 위젯에서 마우스 및 휠 드래그 등으로 크롭 박스 변경 시 UI 수치 동기화"""
        self.spin_crop_w.blockSignals(True)
        self.spin_crop_h.blockSignals(True)
        self.spin_angle.blockSignals(True)

        self.spin_crop_w.setValue(self.viewer.crop_w)
        self.spin_crop_h.setValue(self.viewer.crop_h)
        self.spin_angle.setValue(self.viewer.crop_angle)

        self.spin_crop_w.blockSignals(False)
        self.spin_crop_h.blockSignals(False)
        self.spin_angle.blockSignals(False)

        self.statusBar().showMessage(
            f"크롭 설정 변경 - 중심: ({cx:.1f}, {cy:.1f}) | 크기: {self.viewer.crop_w}×{self.viewer.crop_h} | 각도: {self.viewer.crop_angle:.1f}°"
        )

    def _on_wheel_in_viewer(self, delta):
        """Ctrl/Alt 없는 휠 단독 조작 시 이전/다음 프레임 탐색"""
        step = 1 if delta > 0 else -1
        self._step_frame(step)

    def _on_mode_toggled(self):
        """저장 모드 라디오 버튼 변경에 따른 크롭 박스 드로잉 제어 및 UI 활성화 변경"""
        crop_enabled = self.rad_crop.isChecked()
        self.grp_crop.setEnabled(crop_enabled)
        self.viewer.crop_mode_enabled = crop_enabled
        self.viewer.update()
        self._update_path_preview()

    def _update_crop_params_from_ui(self):
        """UI의 스핀 박스 조정 시 뷰어 크롭 파라미터 업데이트"""
        self.viewer.crop_w = self.spin_crop_w.value()
        self.viewer.crop_h = self.spin_crop_h.value()
        self.viewer.crop_angle = self.spin_angle.value()
        self.viewer.update()

    def _update_path_preview(self):
        """저장될 경로 미리보기 실시간 텍스트 업데이트"""
        folder = self.edit_folder.text().strip() or "[폴더명]"
        prefix = self.edit_prefix.text().strip() or "[파일명]"
        self.lbl_path_preview.setText(f"예상 경로: data/mouse_segmentation/<b>{folder}</b>/{prefix}0001.png")

    # ─── 폴더 탐색 및 로딩 ───

    def _open_folder_dialog(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        start_dir = os.path.join(script_dir, "data")
        if not os.path.exists(start_dir):
            start_dir = script_dir

        folder = QFileDialog.getExistingDirectory(self, "Basler 촬영 폴더 또는 frames 폴더 선택", start_dir)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder_path):
        folder_path = os.path.normpath(folder_path)
        if not os.path.isdir(folder_path):
            return False

        # 사용자가 frames 폴더를 직접 선택한 경우, 한 단계 상위 세션 폴더로 보정
        if os.path.basename(folder_path).lower() == "frames":
            session_dir = os.path.dirname(folder_path)
            frames_dir = folder_path
        else:
            session_dir = folder_path
            frames_dir = os.path.join(session_dir, "frames")

        if not os.path.isdir(frames_dir):
            QMessageBox.warning(self, "오류", f"해당 폴더에 frames 하위 폴더가 없습니다:\n{frames_dir}")
            return False

        # tiff/tif 이미지 찾기
        tiffs = sorted(glob.glob(os.path.join(frames_dir, "*.tiff")))
        if not tiffs:
            tiffs = sorted(glob.glob(os.path.join(frames_dir, "*.tif")))

        if not tiffs:
            QMessageBox.warning(self, "오류", "frames 폴더 내에 TIFF 포맷 파일이 존재하지 않습니다.")
            return False

        self.frames_list = tiffs
        self.current_folder = session_dir

        # camera_summary.json 파싱 및 정보 설정
        summary = load_camera_summary(session_dir)
        if summary:
            self.camera_info = summary
            self.pixel_format = summary.get("PixelFormat", "")
            res = summary.get("Resolution", {})
            w, h = res.get("Width", "?"), res.get("Height", "?")
            fps = summary.get("FPS_Result", summary.get("FPS_Target", "?"))
            dur = summary.get("Record_Duration_sec", "?")
            self.lbl_pixel_fmt.setText(self.pixel_format)
            self.lbl_resolution.setText(f"{w} × {h}")
            self.lbl_fps.setText(f"{fps} fps" if isinstance(fps, (int, float)) else str(fps))
            self.lbl_total_frames.setText(f"{len(tiffs)} frames")
            self.lbl_duration.setText(f"{dur} s" if isinstance(dur, (int, float)) else str(dur))
        else:
            self.camera_info = {}
            self.pixel_format = ""
            # JSON 요약 파일이 없을 경우 첫 번째 이미지 형상 조회 시도
            try:
                first_img = tifffile.imread(tiffs[0])
                h, w = first_img.shape[:2]
                self.lbl_pixel_fmt.setText("미지정 (camera_summary.json 없음)")
                self.lbl_resolution.setText(f"{w} × {h}")
                self.lbl_fps.setText("-")
                self.lbl_total_frames.setText(f"{len(tiffs)} frames")
                self.lbl_duration.setText("-")
            except Exception as e:
                print(f"이미지 차원 분석 실패: {e}")
        # 기본 저장 폴더명을 세션 이름으로 자동 업데이트하지 않고 이전 값을 유지하도록 주석 처리
        # self.edit_folder.setText(os.path.basename(session_dir))

        # 크롭 박스 드래그 시작 좌표 자동 리센터링
        if len(tiffs) > 0:
            try:
                raw_shape = tifffile.imread(tiffs[0]).shape
                h_img, w_img = raw_shape[:2]
                self.viewer.crop_cx = w_img / 2
                self.viewer.crop_cy = h_img / 2
            except:
                pass

        # 슬라이더 및 상태 업데이트
        self.slider.setRange(0, len(self.frames_list) - 1)
        self.slider.setValue(0)
        self.current_idx = 0
        self._show_frame(0)

        self._update_sibling_buttons()
        self._update_path_preview()

        self.statusBar().showMessage(f"폴더 로드 완료: {session_dir} ({len(tiffs)} 프레임)")
        return True

    # ─── 형제 세션 폴더 탐색 이동 ───

    def _get_sibling_folders(self):
        """현재 폴더의 상위 폴더 내에서 frames를 포함하고 있는 형제 폴더 목록을 조회"""
        if not self.current_folder:
            return []
        parent = os.path.dirname(self.current_folder)
        if not os.path.exists(parent):
            return []
        siblings = []
        try:
            for name in sorted(os.listdir(parent)):
                full_path = os.path.join(parent, name)
                if os.path.isdir(full_path):
                    if os.path.isdir(os.path.join(full_path, "frames")):
                        siblings.append(os.path.normpath(full_path))
            return siblings
        except Exception as e:
            print(f"형제 폴더 탐색 오류: {e}")
            return []

    def _update_sibling_buttons(self):
        siblings = self._get_sibling_folders()
        if not siblings or len(siblings) <= 1:
            self.btn_prev_folder.setEnabled(False)
            self.btn_next_folder.setEnabled(False)
            return

        try:
            cur_nc = os.path.normcase(self.current_folder)
            idx = -1
            for i, s in enumerate(siblings):
                if os.path.normcase(s) == cur_nc:
                    idx = i
                    break

            self.btn_prev_folder.setEnabled(idx > 0)
            self.btn_next_folder.setEnabled(idx >= 0 and idx < len(siblings) - 1)
        except Exception:
            self.btn_prev_folder.setEnabled(False)
            self.btn_next_folder.setEnabled(False)

    def prev_folder(self):
        siblings = self._get_sibling_folders()
        if not siblings or not self.current_folder:
            return
        try:
            cur_nc = os.path.normcase(self.current_folder)
            idx = -1
            for i, s in enumerate(siblings):
                if os.path.normcase(s) == cur_nc:
                    idx = i
                    break
            if idx > 0:
                self.load_folder(siblings[idx - 1])
        except Exception as e:
            QMessageBox.warning(self, "오류", f"이전 폴더 로드에 실패했습니다:\n{e}")

    def next_folder(self):
        siblings = self._get_sibling_folders()
        if not siblings or not self.current_folder:
            return
        try:
            cur_nc = os.path.normcase(self.current_folder)
            idx = -1
            for i, s in enumerate(siblings):
                if os.path.normcase(s) == cur_nc:
                    idx = i
                    break
            if idx >= 0 and idx < len(siblings) - 1:
                self.load_folder(siblings[idx + 1])
        except Exception as e:
            QMessageBox.warning(self, "오류", f"다음 폴더 로드에 실패했습니다:\n{e}")

    # ─── 저장 유효성 검증 및 내보내기 ───

    def validate_bit_depth(self):
        """8비트 원본을 12비트(16비트 PNG)로 부적절하게 상향 저장 시 경고 알림"""
        bit_depth = 8 if self.combo_bit_depth.currentIndex() == 0 else 12
        is_source_12bit = "12" in self.pixel_format.upper() or "16" in self.pixel_format.upper()

        if bit_depth == 12 and not is_source_12bit:
            reply = QMessageBox.warning(
                self, "저장 비트 깊이 설정 경고",
                "경고: 8비트 원본 데이터를 12비트(16비트 PNG)로 변경하여 저장하는 것은 부적절합니다.\n"
                "그래도 계속 저장하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return False
        return True

    def _save_current_frame(self):
        """현재 미리보기 화면에 열려있는 단일 프레임 저장"""
        if not self.frames_list:
            QMessageBox.warning(self, "저장 불가", "로딩된 프레임이 없습니다.")
            return

        if not self.validate_bit_depth():
            return

        folder_name = self.edit_folder.text().strip()
        file_prefix = self.edit_prefix.text().strip()
        if not folder_name or not file_prefix:
            QMessageBox.warning(self, "설정 미입력", "저장할 폴더명과 파일명 접두사를 입력해주세요.")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "data", "mouse_segmentation", folder_name)
        os.makedirs(base_dir, exist_ok=True)

        # 번호 이어받기
        max_num = get_next_file_number(base_dir, file_prefix)
        counter = max_num + 1

        fidx = self.current_idx
        try:
            raw = tifffile.imread(self.frames_list[fidx])
            bit_depth = 8 if self.combo_bit_depth.currentIndex() == 0 else 12
            crop_enabled = self.rad_crop.isChecked()
            crop_params = {
                'cx': self.viewer.crop_cx,
                'cy': self.viewer.crop_cy,
                'w': self.viewer.crop_w,
                'h': self.viewer.crop_h,
                'angle': self.viewer.crop_angle
            }

            processed = get_processed_frame_for_save(raw, self.pixel_format, bit_depth, crop_enabled, crop_params)
            if processed is None:
                QMessageBox.critical(self, "오류", "프레임 처리에 실패했습니다. 영역 바깥 침범 여부를 확인하세요.")
                return

            fname = f"{file_prefix}{counter:04d}.png"
            out_path = os.path.join(base_dir, fname)

            # cv2.imwrite는 BGR 순서로 된 3채널 numpy array를 받아서 correct PNG로 출력함
            success = cv2.imwrite(out_path, processed)

            if success:
                self.statusBar().showMessage(f"단일 프레임 저장 성공: {fname} -> {folder_name}")
                QMessageBox.information(self, "완료", f"단일 프레임이 저장되었습니다.\n저장 경로: {out_path}")
            else:
                QMessageBox.critical(self, "실패", f"파일 쓰기 실패: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 중 예외 발생:\n{e}")

    def _export_crops(self):
        """샘플링 간격 조건에 따라 전체 탐색하여 가공 배치 내보내기 수행"""
        if not self.frames_list:
            QMessageBox.warning(self, "내보내기 불가", "로딩된 프레임이 없습니다.")
            return

        if not self.validate_bit_depth():
            return

        folder_name = self.edit_folder.text().strip()
        file_prefix = self.edit_prefix.text().strip()
        if not folder_name or not file_prefix:
            QMessageBox.warning(self, "설정 미입력", "저장할 폴더명과 파일명 접두사를 입력해주세요.")
            return

        sample_interval = self.spin_sample.value()
        bit_depth = 8 if self.combo_bit_depth.currentIndex() == 0 else 12
        crop_enabled = self.rad_crop.isChecked()
        crop_params = {
            'cx': self.viewer.crop_cx,
            'cy': self.viewer.crop_cy,
            'w': self.viewer.crop_w,
            'h': self.viewer.crop_h,
            'angle': self.viewer.crop_angle
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "data", "mouse_segmentation", folder_name)
        os.makedirs(base_dir, exist_ok=True)

        # 번호 이어받기
        max_num = get_next_file_number(base_dir, file_prefix)
        counter = max_num

        # 샘플링 프레임 목록 작성
        indices = list(range(0, len(self.frames_list), sample_interval))
        total = len(indices)

        # UI 비활성화
        self.btn_export.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)

        saved_count = 0
        try:
            for idx, fidx in enumerate(indices):
                raw = tifffile.imread(self.frames_list[fidx])
                processed = get_processed_frame_for_save(raw, self.pixel_format, bit_depth, crop_enabled, crop_params)

                if processed is not None:
                    counter += 1
                    fname = f"{file_prefix}{counter:04d}.png"
                    out_path = os.path.join(base_dir, fname)
                    cv2.imwrite(out_path, processed)
                    saved_count += 1

                self.progress_bar.setValue(idx + 1)
                self.statusBar().showMessage(f"추출 배치 처리 중... ({idx + 1}/{total})")
                QApplication.processEvents()

            self.statusBar().showMessage(f"추출 완료. {saved_count}개 파일 저장됨 -> {folder_name}")
            QMessageBox.information(
                self, "내보내기 완료",
                f"총 {saved_count}개의 프레임이 데이터셋 폴더에 성공적으로 저장되었습니다.\n\n"
                f"경로: {base_dir}\n"
                f"파일명 범위: {file_prefix}{max_num + 1:04d}.png ~ {file_prefix}{counter:04d}.png"
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"배치 내보내기 처리 중 오류 발생:\n{e}")
        finally:
            self.btn_export.setEnabled(True)
            self.btn_save_current.setEnabled(True)
            self.progress_bar.setVisible(False)


# ─────────────────────────── 애플리케이션 진입점 ───────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 기본 시스템 폰트 크기 보정
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
