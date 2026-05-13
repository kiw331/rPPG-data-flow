"""
루트 폴더에서 실행
resample_ppg_timestamps.py

이전 버전 sensor.py의 타임스탬프 중복 문제가 있는 ppg_sensor.csv 파일을
일괄적으로 리샘플링하는 유틸리티.

사용법:
    python resample_ppg_timestamps.py                       # 기본 폴더 사용
    python resample_ppg_timestamps.py data\basler_0303      # 폴더 직접 지정
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd


def find_csv_folders(root: str) -> list[str]:
    """root 하위 전체 트리에서 ppg_sensor.csv 가 존재하는 폴더 경로를 반환."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        if "ppg_sensor.csv" in filenames:
            found.append(dirpath)
    return found


def resample_folder(folder: str) -> bool:
    csv_path = os.path.join(folder, "ppg_sensor.csv")
    backup_path = os.path.join(folder, "ppg_sensor_original.csv")

    # 원본 백업 (최초 1회만)
    if not os.path.exists(backup_path):
        shutil.copy2(csv_path, backup_path)
        print(f"  [백업] ppg_sensor_original.csv 생성")
    else:
        print(f"  [백업 스킵] ppg_sensor_original.csv 이미 존재")

    df = pd.read_csv(csv_path)

    if "Timestamp" not in df.columns:
        print(f"  [스킵] Timestamp 컬럼 없음")
        return False

    n = len(df)
    if n < 2:
        print(f"  [스킵] 데이터가 1행 이하 ({n}행)")
        return False

    t_start = df["Timestamp"].iloc[0]
    t_end   = df["Timestamp"].iloc[-1]

    # 중복 타임스탬프 개수 확인 (정보 출력용)
    dup_count = int(df["Timestamp"].duplicated().sum())

    df["Timestamp"] = np.linspace(t_start, t_end, n)
    df.to_csv(csv_path, index=False)

    print(f"  [완료] {n}행 / 중복 {dup_count}개 → linspace 재할당 "
          f"({t_start:.6f} ~ {t_end:.6f})")
    return True


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"data\basler_0507"

    if not os.path.isdir(root):
        print(f"[오류] 폴더가 존재하지 않습니다: {root}")
        sys.exit(1)

    folders = find_csv_folders(root)

    if not folders:
        print(f"[정보] ppg_sensor.csv 파일을 찾지 못했습니다: {root}")
        sys.exit(0)

    print(f"[정보] {len(folders)}개 폴더 발견\n")

    success = 0
    for folder in folders:
        rel = os.path.relpath(folder, root)
        print(f"[처리] {rel}")
        if resample_folder(folder):
            success += 1
        print()

    print(f"[종료] {success} / {len(folders)}개 파일 처리 완료.")


if __name__ == "__main__":
    main()
