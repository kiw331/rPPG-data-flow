"""
세그먼테이션 용 이미지 Crop GUI 도구
- 폴더 선택 → frames/*.tiff 로드, camera_summary.json 기반 픽셀포맷 처리
- BayerRG8 / BayerRG12 / BGR8 자동 디베이어링 (코드 48/46 규칙 준수)
- 프레임 미리보기, 재생, 탐색 (슬라이더 + 재생/정지)
- 크기조절·회전 가능한 crop 사각형 (기본 128×128)
- 샘플링 간격 설정 (기본 60)
- data/mouse_segmentation/[폴더명]/[파일명]NNNN.png 로 저장, 번호 이어짐
"""

import sys, os, json, glob, math, re
import numpy as np
import cv2
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox, QProgressDialog, QDoubleSpinBox,
    QSizePolicy, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QTransform


# ─────────────────────────── 프레임 로더 ───────────────────────────

def load_camera_summary(folder):
    path = os.path.join(folder, "camera_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_frame(raw_img, pixel_format):
    """
    TIFF 프레임을 RGB numpy 배열로 변환.
    Bayer Pixel Format Fix Log.md 규칙 준수:
      - BayerRG12 / BayerRG8: 2D 배열 → cv2.COLOR_BayerBG2RGB (코드 48, QImage RGB888 용)
      - BGR8: 3채널 배열 → cv2.cvtColor BGR2RGB
    """
    pf = pixel_format.upper() if pixel_format else ""

    if raw_img.ndim == 2:
        # Bayer 패턴 (BayerRG8 또는 BayerRG12)
        if "12" in pf:
            # BayerRG12: uint16, 12비트 → 8비트로 스케일
            img8 = (raw_img >> 4).astype(np.uint8)
        else:
            img8 = raw_img.astype(np.uint8)
        # 코드 48: COLOR_BayerBG2RGB (QImage RGB888 용)
        rgb = cv2.cvtColor(img8, cv2.COLOR_BayerBG2RGB)
        return rgb
    elif raw_img.ndim == 3:
        # BGR8: tifffile은 바이트 순서 그대로(BGR) 반환
        rgb = cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb
    else:
        return raw_img.astype(np.uint8)


# ─────────────────────────── 메인 윈도우 ───────────────────────────

class CropOverlayLabel(QLabel):
    """프레임 표시 + 크롭 박스 오버레이"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #444;")

        # crop box state (image 좌표)
        self.crop_cx = 512.0
        self.crop_cy = 512.0
        self.crop_w = 128
        self.crop_h = 128
        self.crop_angle = 0.0  # degrees

        self._pixmap = None
        self._img_w = 0
        self._img_h = 0

        # drag state
        self._dragging = False
        self._drag_start = None

    def set_image(self, rgb_array):
        h, w, ch = rgb_array.shape
        self._img_w = w
        self._img_h = h
        bytes_per_line = ch * w
        qimg = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _get_display_rect(self):
        """위젯 안에서 pixmap이 실제로 그려지는 영역 (aspect ratio 유지)"""
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
        rect, scale = self._get_display_rect()
        ix = (pos.x() - rect.x()) / scale
        iy = (pos.y() - rect.y()) / scale
        return ix, iy

    def _img_to_widget(self, ix, iy):
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

            # Draw crop box
            cx_w, cy_w = self._img_to_widget(self.crop_cx, self.crop_cy)
            painter.save()
            painter.translate(cx_w, cy_w)
            painter.rotate(self.crop_angle)
            hw = self.crop_w * scale / 2
            hh = self.crop_h * scale / 2

            pen = QPen(QColor("#00ff88"), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(-hw, -hh, hw * 2, hh * 2))

            # 중심 십자
            painter.setPen(QPen(QColor("#ff4444"), 1, Qt.DashLine))
            painter.drawLine(QPointF(-hw, 0), QPointF(hw, 0))
            painter.drawLine(QPointF(0, -hh), QPointF(0, hh))

            # 크기 텍스트
            painter.setPen(QColor("#00ff88"))
            font = QFont("Consolas", 9)
            painter.setFont(font)
            painter.drawText(QPointF(-hw + 4, -hh - 6),
                             f"{self.crop_w}×{self.crop_h}  {self.crop_angle:.1f}°")
            painter.restore()

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap:
            self._dragging = True
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self._dragging and self._pixmap:
            ix, iy = self._widget_to_img(event.pos())
            self.crop_cx = max(0, min(self._img_w, ix))
            self.crop_cy = max(0, min(self._img_h, iy))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def get_crop_corners_img(self):
        """이미지 좌표계에서 회전된 crop 박스의 4개 꼭짓점 반환"""
        cx, cy = self.crop_cx, self.crop_cy
        hw, hh = self.crop_w / 2, self.crop_h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        rad = math.radians(self.crop_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated = []
        for x, y in corners:
            rx = cx + x * cos_a - y * sin_a
            ry = cy + x * sin_a + y * cos_a
            rotated.append((rx, ry))
        return np.array(rotated, dtype=np.float32)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("세그먼테이션 Crop 도구")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet("""
            QMainWindow { background: #181825; }
            QGroupBox { color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px;
                         margin-top: 12px; padding-top: 14px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel { color: #cdd6f4; }
            QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                          border-radius: 4px; padding: 6px 14px; font-size: 13px; }
            QPushButton:hover { background: #45475a; }
            QPushButton:pressed { background: #585b70; }
            QPushButton#btnExport { background: #a6e3a1; color: #1e1e2e; font-weight: bold; }
            QPushButton#btnExport:hover { background: #94e2d5; }
            QSlider::groove:horizontal { background: #45475a; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 14px; margin: -5px 0;
                                          border-radius: 7px; }
            QSpinBox, QDoubleSpinBox, QLineEdit { background: #313244; color: #cdd6f4;
                          border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
            QStatusBar { color: #a6adc8; }
        """)

        self.frames_list = []
        self.pixel_format = ""
        self.camera_info = {}
        self.current_idx = 0
        self.cached_rgb = None
        self.playing = False

        self._build_ui()
        self._connect_signals()

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        # ─── Left: image viewer ───
        left = QVBoxLayout()
        self.viewer = CropOverlayLabel()
        left.addWidget(self.viewer, 1)

        # transport bar
        transport = QHBoxLayout()
        self.btn_open = QPushButton("📂 폴더 열기")
        self.btn_play = QPushButton("▶ 재생")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setMinimumWidth(80)
        transport.addWidget(self.btn_open)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.lbl_frame)
        left.addLayout(transport)

        root_layout.addLayout(left, 3)

        # ─── Right: controls ───
        right = QVBoxLayout()

        # info group
        grp_info = QGroupBox("영상 정보")
        info_form = QFormLayout()
        self.lbl_pixel_fmt = QLabel("-")
        self.lbl_resolution = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_total_frames = QLabel("-")
        self.lbl_duration = QLabel("-")
        self.lbl_exposure = QLabel("-")
        info_form.addRow("픽셀포맷:", self.lbl_pixel_fmt)
        info_form.addRow("해상도:", self.lbl_resolution)
        info_form.addRow("FPS:", self.lbl_fps)
        info_form.addRow("프레임 수:", self.lbl_total_frames)
        info_form.addRow("녹화시간:", self.lbl_duration)
        info_form.addRow("노출(μs):", self.lbl_exposure)
        grp_info.setLayout(info_form)
        right.addWidget(grp_info)

        # crop group
        grp_crop = QGroupBox("크롭 설정")
        crop_form = QFormLayout()
        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(16, 2048)
        self.spin_crop_w.setValue(128)
        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(16, 2048)
        self.spin_crop_h.setValue(128)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.spin_angle.setSingleStep(1.0)
        self.spin_angle.setValue(0.0)
        self.spin_angle.setSuffix("°")
        crop_form.addRow("너비:", self.spin_crop_w)
        crop_form.addRow("높이:", self.spin_crop_h)
        crop_form.addRow("회전:", self.spin_angle)
        grp_crop.setLayout(crop_form)
        right.addWidget(grp_crop)

        # sampling & export group
        grp_export = QGroupBox("저장 설정")
        export_form = QFormLayout()
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 9999)
        self.spin_sample.setValue(60)
        self.edit_folder = QLineEdit("session01")
        self.edit_prefix = QLineEdit("img")
        export_form.addRow("샘플링 간격:", self.spin_sample)
        export_form.addRow("폴더명:", self.edit_folder)
        export_form.addRow("파일명(접두사):", self.edit_prefix)
        grp_export.setLayout(export_form)
        right.addWidget(grp_export)

        self.btn_export = QPushButton("💾 크롭 저장 (Export)")
        self.btn_export.setObjectName("btnExport")
        self.btn_export.setMinimumHeight(40)
        right.addWidget(self.btn_export)

        right.addStretch()
        root_layout.addLayout(right, 1)

        self.statusBar().showMessage("폴더를 열어 프레임을 불러오세요.")

    def _connect_signals(self):
        self.btn_open.clicked.connect(self._open_folder)
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin_crop_w.valueChanged.connect(self._update_crop_params)
        self.spin_crop_h.valueChanged.connect(self._update_crop_params)
        self.spin_angle.valueChanged.connect(self._update_crop_params)
        self.btn_export.clicked.connect(self._export_crops)

    # ─── 폴더 열기 ───
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "세션 폴더 선택",
                                                   os.path.join(os.path.dirname(__file__), "data"))
        if not folder:
            return

        frames_dir = os.path.join(folder, "frames")
        if not os.path.isdir(frames_dir):
            QMessageBox.warning(self, "오류", "선택한 폴더에 frames 디렉터리가 없습니다.")
            return

        self.frames_list = sorted(glob.glob(os.path.join(frames_dir, "*.tiff")))
        if not self.frames_list:
            self.frames_list = sorted(glob.glob(os.path.join(frames_dir, "*.tif")))
        if not self.frames_list:
            QMessageBox.warning(self, "오류", "frames 폴더에 TIFF 파일이 없습니다.")
            return

        summary = load_camera_summary(folder)
        if summary:
            self.camera_info = summary
            self.pixel_format = summary.get("PixelFormat", "")
            res = summary.get("Resolution", {})
            w, h = res.get("Width", "?"), res.get("Height", "?")
            fps = summary.get("FPS_Result", summary.get("FPS_Target", "?"))
            dur = summary.get("Record_Duration_sec", "?")
            exp = summary.get("Exposure_us", "?")
            self.lbl_pixel_fmt.setText(self.pixel_format)
            self.lbl_resolution.setText(f"{w} × {h}")
            self.lbl_fps.setText(str(fps))
            self.lbl_total_frames.setText(str(len(self.frames_list)))
            self.lbl_duration.setText(f"{dur} s")
            self.lbl_exposure.setText(str(exp))
        else:
            self.pixel_format = ""
            self.lbl_pixel_fmt.setText("(camera_summary.json 없음)")

        self.slider.setMaximum(len(self.frames_list) - 1)
        self.slider.setValue(0)
        self.current_idx = 0
        self._show_frame(0)
        self.statusBar().showMessage(f"로드 완료: {folder}  ({len(self.frames_list)} 프레임)")

    # ─── 프레임 표시 ───
    def _show_frame(self, idx):
        if idx < 0 or idx >= len(self.frames_list):
            return
        self.current_idx = idx
        raw = tifffile.imread(self.frames_list[idx])
        rgb = decode_frame(raw, self.pixel_format)
        self.cached_rgb = rgb
        self.viewer.set_image(rgb)
        self.lbl_frame.setText(f"{idx} / {len(self.frames_list) - 1}")

    def _on_slider(self, val):
        self._show_frame(val)

    # ─── 재생 ───
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
            fps = 30  # 프리뷰 재생 속도
            self.play_timer.start(int(1000 / fps))

    def _on_play_tick(self):
        nxt = self.current_idx + 1
        if nxt >= len(self.frames_list):
            nxt = 0
        self.slider.setValue(nxt)

    # ─── 크롭 파라미터 동기화 ───
    def _update_crop_params(self):
        self.viewer.crop_w = self.spin_crop_w.value()
        self.viewer.crop_h = self.spin_crop_h.value()
        self.viewer.crop_angle = self.spin_angle.value()
        self.viewer.update()

    # ─── 크롭 & 저장 ───
    def _export_crops(self):
        if not self.frames_list:
            QMessageBox.warning(self, "오류", "먼저 폴더를 열어주세요.")
            return

        folder_name = self.edit_folder.text().strip()
        file_prefix = self.edit_prefix.text().strip()
        if not folder_name or not file_prefix:
            QMessageBox.warning(self, "오류", "폴더명과 파일명을 입력해주세요.")
            return

        sample_interval = self.spin_sample.value()
        crop_w = self.spin_crop_w.value()
        crop_h = self.spin_crop_h.value()
        angle = self.spin_angle.value()
        cx, cy = self.viewer.crop_cx, self.viewer.crop_cy

        # 저장 경로
        base_dir = os.path.join(os.path.dirname(__file__), "data", "mouse_segmentation", folder_name)
        os.makedirs(base_dir, exist_ok=True)

        # 기존 파일 번호 이어받기
        existing = glob.glob(os.path.join(base_dir, f"{file_prefix}*.png"))
        max_num = 0
        pattern = re.compile(re.escape(file_prefix) + r"(\d+)\.png$")
        for f in existing:
            m = pattern.search(os.path.basename(f))
            if m:
                max_num = max(max_num, int(m.group(1)))
        counter = max_num  # will start from max_num + 1

        # 샘플링할 프레임 인덱스
        indices = list(range(0, len(self.frames_list), sample_interval))
        total = len(indices)

        progress = QProgressDialog(f"크롭 저장 중... (0/{total})", "취소", 0, total, self)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        saved = 0
        for i, fidx in enumerate(indices):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"크롭 저장 중... ({i + 1}/{total})")
            QApplication.processEvents()

            raw = tifffile.imread(self.frames_list[fidx])
            rgb = decode_frame(raw, self.pixel_format)

            # 회전 크롭 수행
            cropped = self._rotate_crop(rgb, cx, cy, crop_w, crop_h, angle)
            if cropped is None:
                continue

            counter += 1
            fname = f"{file_prefix}{counter:04d}.png"
            out_path = os.path.join(base_dir, fname)
            # PNG 저장 (RGB → BGR for cv2)
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            saved += 1

        progress.setValue(total)
        QMessageBox.information(self, "완료",
                                f"저장 완료: {saved}장\n경로: {base_dir}\n"
                                f"파일: {file_prefix}{max_num + 1:04d}.png ~ {file_prefix}{counter:04d}.png")
        self.statusBar().showMessage(f"Export 완료: {saved}장 저장됨 → {base_dir}")

    @staticmethod
    def _rotate_crop(img, cx, cy, w, h, angle_deg):
        """이미지에서 (cx, cy) 중심, w×h 크기, angle_deg 회전된 영역을 크롭"""
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
        # 역회전 적용 → 축 정렬된 사각형으로 크롭
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
        # 크기가 다르면 패딩
        if crop.shape[0] != h or crop.shape[1] != w:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            ox = x1c - x1
            oy = y1c - y1
            canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
            return canvas
        return crop


# ─────────────────────────── 실행 ───────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
