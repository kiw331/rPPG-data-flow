# 파일명: modules/tiff_to_avi.py
# 목적: TIFF 프레임 시퀀스를 압축 없는 AVI 영상으로 변환
#
# 지원 포맷:
#   - BayerRG12 (uint16, 1ch) → 디베이어링 후 8bit 변환 → AVI
#   - BayerRG8  (uint8, 1ch)  → 디베이어링 → AVI
#   - BGR8      (uint8, 3ch)  → 그대로 AVI
#
# 사용 예시:
#   from modules.tiff_to_avi import tiff_sequence_to_avi
#   tiff_sequence_to_avi(frames_dir='data/.../frames', output_path='output/video.avi')

import os
import glob
import json
import cv2
import numpy as np
import tifffile


def _get_pixel_format(frames_dir: str) -> str:
    """frames_dir 상위 폴더의 camera_summary.json에서 PixelFormat을 읽어 반환."""
    parent = os.path.dirname(frames_dir)
    summary_path = os.path.join(parent, "camera_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        fmt = s.get("PixelFormat", "BayerRG12")
        fps = s.get("FPS_Result") or s.get("FPS_Target") or 60.0
        return fmt, float(fps)
    return "BayerRG12", 60.0


def _demosaic_to_bgr8(img: np.ndarray, pixel_format: str) -> np.ndarray:
    """
    raw Bayer 배열을 OpenCV용 BGR uint8로 변환.
    Fix Log: 코드 48 (BayerRG2BGR) 사용 — 이 카메라에서 실증된 올바른 연산.
    """
    if len(img.shape) == 3:
        # BGR8: tifffile로 읽으면 바이트 순서 그대로 (BGR)
        # VideoWriter는 BGR을 기대하므로 변환 없이 그대로 전달
        return img.astype(np.uint8)

    # 1채널 Bayer 배열 처리
    fmt = pixel_format.upper()

    # Bayer 패턴별 OpenCV 변환 코드 선택 (2BGR: 코드 48 계열)
    # VideoWriter는 표준 BGR 파이프라인 → 코드 46 (BayerBG2BGR = BayerRG2RGB) 사용
    # ※ pyqtgraph는 코드 48(BayerRG2BGR)이 올바름 (Fix Log 참조)
    #    VideoWriter/미디어플레이어는 코드 46이 올바름 (BGR 정방향 해석)
    bayer_map = {
        "BAYERRG": cv2.COLOR_BayerBG2BGR,   # 코드 46
        "BAYERGB": cv2.COLOR_BayerGR2BGR,   # 코드 46 계열
        "BAYERGR": cv2.COLOR_BayerGB2BGR,   # 코드 46 계열
        "BAYERBG": cv2.COLOR_BayerRG2BGR,   # 코드 46 계열
    }
    code = None
    for key, val in bayer_map.items():
        if fmt.startswith(key):
            code = val
            break
    if code is None:
        # Mono 또는 알 수 없는 포맷 → 그레이스케일로 처리
        code = None

    # 12bit → 8bit 정규화
    if "12" in fmt:
        img_8 = (img >> 4).astype(np.uint8)
    elif "16" in fmt:
        img_8 = (img >> 8).astype(np.uint8)
    else:
        img_8 = img.astype(np.uint8)

    if code is not None:
        bgr = cv2.cvtColor(img_8, code)
    else:
        # Mono: 3채널로 복사
        bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)

    return bgr


def tiff_sequence_to_avi(
    frames_dir: str,
    output_path: str = None,
    fps: float = None,
    pixel_format: str = None,
    codec: str = "MJPG",
    quality: int = None,
    verbose: bool = True,
) -> str:
    """
    TIFF 프레임 시퀀스를 AVI 파일로 변환합니다.

    Parameters
    ----------
    frames_dir   : TIFF 파일들이 있는 폴더 경로
    output_path  : 출력 AVI 경로 (None이면 frames_dir 상위에 자동 생성)
    fps          : 영상 FPS (None이면 camera_summary.json에서 자동 읽기)
    pixel_format : 픽셀 포맷 (None이면 camera_summary.json에서 자동 읽기)
    codec        : FourCC 코덱 문자열
                   'DIB '  → 압축 없음 (무손실, 파일 크기 매우 큼)
                   'MJPG'  → Motion JPEG (소용량, 손실 압축)
                   'XVID'  → Xvid (일반 AVI)
    quality      : MJPG 사용 시 품질 (미사용, OpenCV VideoWriter 기본)
    verbose      : 진행 상황 출력 여부

    Returns
    -------
    str : 저장된 AVI 파일 경로
    """
    # 프레임 파일 목록 수집
    tiff_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.tiff")) +
        glob.glob(os.path.join(frames_dir, "*.tif"))
    )
    if not tiff_files:
        raise FileNotFoundError(f"TIFF 파일이 없습니다: {frames_dir}")

    # camera_summary.json에서 설정 자동 읽기
    auto_fmt, auto_fps = _get_pixel_format(frames_dir)
    if pixel_format is None:
        pixel_format = auto_fmt
    if fps is None:
        fps = auto_fps

    # 첫 프레임으로 해상도 확인
    sample = tifffile.imread(tiff_files[0])
    bgr_sample = _demosaic_to_bgr8(sample, pixel_format)
    h, w = bgr_sample.shape[:2]

    # 출력 경로 자동 설정
    if output_path is None:
        parent = os.path.dirname(frames_dir)
        folder_name = os.path.basename(parent)
        output_path = os.path.join(parent, f"{folder_name}_{codec}.avi")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # VideoWriter 생성
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter를 열 수 없습니다. 코덱 '{codec}'을 확인하세요.")

    if verbose:
        print(f"[tiff_to_avi] 변환 시작")
        print(f"  입력 폴더  : {frames_dir}")
        print(f"  총 프레임  : {len(tiff_files)}")
        print(f"  해상도     : {w} x {h}")
        print(f"  FPS        : {fps}")
        print(f"  PixelFormat: {pixel_format}")
        print(f"  코덱       : {codec}")
        print(f"  출력 경로  : {output_path}")

    for i, fpath in enumerate(tiff_files):
        img = tifffile.imread(fpath)
        bgr = _demosaic_to_bgr8(img, pixel_format)
        writer.write(bgr)

        if verbose and (i + 1) % 100 == 0:
            print(f"  진행: {i+1}/{len(tiff_files)} ({(i+1)/len(tiff_files)*100:.1f}%)")

    writer.release()

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"[tiff_to_avi] 완료: {output_path} ({size_mb:.1f} MB)")

    return output_path


# =====================================================================
# 직접 실행 시 사용 예시
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TIFF 프레임 → AVI 변환")
    parser.add_argument("frames_dir", help="TIFF 파일이 있는 폴더")
    parser.add_argument("--output", default=None, help="출력 AVI 경로 (기본: 자동)")
    parser.add_argument("--fps", type=float, default=None, help="FPS (기본: camera_summary.json)")
    parser.add_argument("--codec", default="DIB ", help="코덱 FourCC (기본: DIB  = 무압축)")
    parser.add_argument("--format", default=None, dest="pixel_format", help="픽셀 포맷")
    args = parser.parse_args()

    tiff_sequence_to_avi(
        frames_dir=args.frames_dir,
        output_path=args.output,
        fps=args.fps,
        pixel_format=args.pixel_format,
        codec=args.codec,
    )
