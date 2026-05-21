"""
sam3_test.py
============
SAM 3 모델을 이용하여 첫 번째 프레임에서 foot / tail을 추론하고
원본 이미지에 반투명 오버레이로 팝업 표시합니다.

지원 픽셀 포맷 (camera_summary.json 자동 감지):
  - BayerRG12  (uint16, 1ch)  → 디베이어링(코드 48) → RGB
  - BayerRG8   (uint8,  1ch)  → 디베이어링(코드 48) → RGB
  - BGR8       (uint8,  3ch)  → BGR→RGB 변환
  - Mono8/Mono12 (1ch)        → 그레이→RGB 변환

디베이어링 기준:
  Fix Log §3 표 기준, **pyqtgraph / 일반 RGB 파이프라인** 에서는
  COLOR_BayerRG2BGR (코드 48) 이 올바른 연산입니다.
  OpenCV는 BGR 채널 순서이므로, SAM3/PIL 입력용 RGB는
  마지막에 cv2.cvtColor(..., COLOR_BGR2RGB)를 추가합니다.

의존 패키지 (현재 venv 기준):
  sam3, torch, torchvision, tifffile, opencv-python, Pillow, numpy, matplotlib
"""

import json
import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

# matplotlib 한글 폰트 설정 (Windows 기본 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False



# ──────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────
DATA_FOLDER = r"C:\Users\COMSOL_LSH\git_package\rPPG-data-flow\data\basler_0409\20260403_155430"
FRAMES_DIR = os.path.join(DATA_FOLDER, "frames")
SUMMARY_JSON = os.path.join(DATA_FOLDER, "camera_summary.json")

# SAM 3 추론 대상 텍스트 프롬프트
PROMPTS = ["foot", "tail"]

# 마스크 오버레이 색상 (RGB)
MASK_COLORS = {
    "foot": (255, 60, 60),   # 붉은색
    "tail": (60, 200, 60),   # 초록색
}
OVERLAY_ALPHA = 0.40         # 반투명 강도


# ──────────────────────────────────────────────────────────────
# 유틸: metadata 읽기
# ──────────────────────────────────────────────────────────────
def read_metadata(summary_path: str) -> dict:
    """camera_summary.json 에서 PixelFormat 등 메타데이터 반환."""
    if not os.path.exists(summary_path):
        print(f"[경고] camera_summary.json 없음 → BayerRG12 기본값 사용")
        return {"PixelFormat": "BayerRG12"}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# 유틸: TIFF → RGB ndarray
#   Fix Log §3, §8 기반 디베이어링 / 채널 변환 로직
# ──────────────────────────────────────────────────────────────
def load_frame_as_rgb(tiff_path: str, pixel_format: str) -> np.ndarray:
    """
    TIFF 프레임을 읽어 uint8 RGB ndarray (H, W, 3) 로 반환.

    픽셀 포맷별 처리:
      - Bayer 계열 (1ch, 2D):
          · 12bit → >> 4 → uint8
          · 16bit → >> 8 → uint8
          · Bayer 패턴 → OpenCV 디베이어링 (코드 48, RGB 파이프라인용)
      - BGR8 / BGR 계열 (3ch):
          · tifffile은 바이트 순서 그대로(BGR) 반환
          · BGR→RGB 명시 변환 (Fix Log §8)
      - Mono 계열 (1ch): 그레이→RGB
    """
    img = tifffile.imread(tiff_path)
    fmt = pixel_format.upper().replace("_", "").replace("-", "")

    # ── 1채널 ──────────────────────────────────────────────────
    if img.ndim == 2:
        # Bayer 패턴 여부 확인
        bayer_prefix_map = {
            "BAYERRG": cv2.COLOR_BayerBG2RGB,  # 코드 48 ✅ (RGB 직접 변환용)
            "BAYERBG": cv2.COLOR_BayerRG2RGB,  # 코드 46
            "BAYERGB": cv2.COLOR_BayerGR2RGB,
            "BAYERGR": cv2.COLOR_BayerGB2RGB,
        }
        bayer_code = None
        for prefix, code in bayer_prefix_map.items():
            if fmt.startswith(prefix):
                bayer_code = code
                break

        # 비트 깊이 정규화 → uint8
        if "12" in fmt:
            img8 = (img >> 4).astype(np.uint8)
        elif "16" in fmt:
            img8 = (img >> 8).astype(np.uint8)
        else:
            img8 = img.astype(np.uint8)

        if bayer_code is not None:
            # 디베이어링 → 바로 RGB
            rgb = cv2.cvtColor(img8, bayer_code)
        else:
            # Mono → RGB
            rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)

    # ── 3채널 ──────────────────────────────────────────────────
    else:
        if img.dtype != np.uint8:
            img = (img >> 8).astype(np.uint8)

        if "BGR" in fmt:
            # tifffile로 읽은 BGR8 → BGR→RGB (Fix Log §8)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # RGB 순서인 경우 그대로
            rgb = img.copy()

    print(f"  프레임 로드 완료: shape={rgb.shape}, dtype={rgb.dtype}, "
          f"PixelFormat={pixel_format}")
    return rgb


# ──────────────────────────────────────────────────────────────
# 유틸: 마스크 오버레이 생성
# ──────────────────────────────────────────────────────────────
def apply_mask_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    base_rgb 위에 mask 영역을 color_rgb 색으로 alpha 투명도로 합성.
    mask: bool or uint8 2D array (H, W)
    """
    overlay = base_rgb.copy()
    overlay[mask > 0] = color_rgb
    return cv2.addWeighted(overlay, alpha, base_rgb, 1 - alpha, 0)


# ──────────────────────────────────────────────────────────────
# 메인 추론 루틴
# ──────────────────────────────────────────────────────────────
def main():
    import torch

    # ── CLI 인수 파싱 ────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="SAM 3 foot/tail 세그멘테이션 테스트")
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace 액세스 토큰 (gated 모델 다운로드 필요). "
             "또는 환경변수 HF_TOKEN 사용."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="감지 신뢰도 임계값 (기본값: 0.3)"
    )
    args = parser.parse_args()

    # ── 0. HuggingFace 인증 확인 ────────────────────────────────
    # facebook/sam3 는 gated 모델로, 사전에 다음 중 하나가 필요합니다:
    #   1) hf auth login  (한 번만 실행하면 됨)
    #   2) 환경변수 HF_TOKEN=<your_token>
    try:
        from huggingface_hub import get_token
        token = (
            args.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token()
        )
        if token:
            os.environ["HF_TOKEN"] = token
            print("[OK] HuggingFace 토큰 확인됨.")
        else:
            print("[경고] HuggingFace 토큰 없음.")
            print("       facebook/sam3 는 gated 모델입니다.")
            print("       다음 방법 중 하나로 인증 후 실행하세요:")
            print("         python sam3_test.py --hf-token <YOUR_TOKEN>")
            print("         또는: $env:HF_TOKEN='<YOUR_TOKEN>'; python sam3_test.py")
    except Exception as e:
        print(f"[경고] HF 토큰 확인 실패: {e}")

    # ── 1. 메타데이터 & 첫 프레임 로드 ─────────────────────────
    print("\n[1/4] 메타데이터 및 프레임 로드...")
    meta = read_metadata(SUMMARY_JSON)
    pixel_format = meta.get("PixelFormat", "BayerRG12")
    print(f"  PixelFormat: {pixel_format}")

    tiff_files = sorted(
        f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".tiff", ".tif"))
    )
    if not tiff_files:
        sys.exit(f"[오류] TIFF 파일 없음: {FRAMES_DIR}")

    first_tiff = os.path.join(FRAMES_DIR, tiff_files[0])
    print(f"  첫 프레임: {tiff_files[0]}")
    rgb_frame = load_frame_as_rgb(first_tiff, pixel_format)

    # PIL Image 변환 (Sam3Processor 입력)
    pil_image = Image.fromarray(rgb_frame)

    # ── 2. SAM 3 모델 로드 ──────────────────────────────────────
    print("\n[2/4] SAM 3 모델 로드 (HuggingFace 자동 다운로드)...")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  사용 디바이스: {device}")

    # load_from_HF=True → facebook/sam3 에서 자동 다운로드
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence)
    print("  모델 로드 완료.")

    # ── 3. 이미지 인코딩 ────────────────────────────────────────
    print("\n[3/4] 이미지 인코딩 및 추론...")
    import contextlib
    if device == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        state = processor.set_image(pil_image)

    # ── 4. 텍스트 프롬프트 추론 ─────────────────────────────────
    result_image = rgb_frame.copy()
    found_any = False

    for prompt in PROMPTS:
        print(f"  프롬프트: '{prompt}' 추론 중...")
        with autocast_ctx:
            state = processor.set_text_prompt(prompt=prompt, state=state)

        masks = state.get("masks")    # shape: (N, 1, H, W) bool tensor
        scores = state.get("scores")  # shape: (N,)
        boxes = state.get("boxes")    # shape: (N, 4) xyxy

        if masks is None or masks.shape[0] == 0:
            print(f"    → '{prompt}' 감지 없음 (confidence {processor.confidence_threshold})")
            # 이미지 인코딩 state는 유지하고 텍스트 결과만 초기화
            processor.reset_all_prompts(state)
            with autocast_ctx:
                state = processor.set_image(pil_image)
            continue

        found_any = True
        color = MASK_COLORS.get(prompt, (200, 200, 60))
        num_det = masks.shape[0]
        print(f"    → {num_det}개 감지 (최고 score: {scores[0].item():.3f})")

        # 모든 감지된 인스턴스를 같은 색으로 오버레이
        for i in range(num_det):
            mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
            result_image = apply_mask_overlay(result_image, mask_np, color, OVERLAY_ALPHA)

            # 바운딩박스 및 라벨 표시
            box = boxes[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            label = f"{prompt} {scores[i].item():.2f}"
            # 박스
            cv2.rectangle(
                result_image, (x1, y1), (x2, y2),
                color=(color[2], color[1], color[0]),  # BGR
                thickness=2
            )
            # 라벨 배경
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                result_image, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                (color[2], color[1], color[0]), -1
            )
            # 라벨 텍스트 (흰색)
            cv2.putText(
                result_image, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        # 다음 프롬프트를 위해 이미지 인코딩 state 유지, 텍스트 결과만 초기화
        processor.reset_all_prompts(state)
        state = processor.set_image(pil_image)

    if not found_any:
        print("\n  [결과] 어떤 프롬프트에서도 감지된 객체가 없습니다.")
        print("  Tip: confidence_threshold를 낮추거나 프롬프트를 변경해 보세요.")

    # ── 5. 팝업 표시 ────────────────────────────────────────────
    print("\n[4/4] 결과 이미지 팝업 표시...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"SAM 3 추론 결과  |  PixelFormat: {pixel_format}  |  "
        f"프롬프트: {PROMPTS}",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(rgb_frame)
    axes[0].set_title("원본 프레임 (디베이어링 후)")
    axes[0].axis("off")

    axes[1].imshow(result_image)
    axes[1].set_title(
        f"SAM 3 오버레이  "
        f"(빨강=foot, 초록=tail, α={OVERLAY_ALPHA})"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("완료.")


if __name__ == "__main__":
    main()
