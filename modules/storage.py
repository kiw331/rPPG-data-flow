# 파일명: storage.py
# 목적: 카메라 프레임 기록(TIFF/RAW) 등 파일 I/O 담당 스레드 워커 제공

import os
import csv
import json
import tifffile
from pypylon import pylon

def writer_worker(q, save_dir):
    """일반적인 (Basic, CameraOnly, LiveMonitor) TIFF 방식 저장 워커"""
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
    print("\n💾 [Writer] 이미지(TIFF) 저장 완료.")

def raw_writer_worker(q, save_dir):
    """RawSave 전용 .raw 바이너리 고속 저장 워커"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    video_filename = os.path.join(save_dir, "recording.raw")
    csv_path = os.path.join(save_dir, "camera_timestamps.csv")
    
    print(f"💾 [Writer] 고속 저장 준비: {video_filename}")
    
    with open(video_filename, 'wb') as f_vid, open(csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Frame_Index", "Timestamp"])
        
        frame_idx = 0
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            
            raw_data, timestamp = item
            
            # 배열 메모리를 디스크에 그대로 이어붙이기 (속도 최적화)
            raw_data.tofile(f_vid) 
            
            writer.writerow([frame_idx, timestamp])
            frame_idx += 1
            q.task_done()
            
    print(f"\n💾 [Writer] .raw 저장 완료. 총 {frame_idx} 프레임.")

def save_camera_settings(camera, program_version, actual_duration, parent_dir):
    """카메라 설정을 요약(JSON) 및 전체 형태(TXT)의 별개 파일로 분리 저장"""
    summary_path = os.path.join(parent_dir, "camera_summary.json")
    settings_path = os.path.join(parent_dir, "camera_all_settings.txt")
    temp_pfs_path = os.path.join(parent_dir, "temp_settings.pfs")
    
    try:
        pylon.FeaturePersistence.Save(temp_pfs_path, camera.GetNodeMap())
        with open(temp_pfs_path, 'r', encoding='utf-8') as f:
            full_dump = f.read()
        os.remove(temp_pfs_path) 
        
        def get_val(node):
            try: return getattr(camera, node).GetValue()
            except: return None
        
        fps_result = get_val("ResultingFrameRate")
        if fps_result is not None: fps_result = round(fps_result, 2)
        
        summary = {
            "Program_Version": program_version,
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
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(full_dump)
            
        print(f"💾 [Camera] 요약 JSON({os.path.basename(summary_path)}) 및 전체 설정 TXT 저장 완료 (실제 소요시간: {actual_duration}초)")
        
    except Exception as e:
        print(f"❌ 설정 파일 저장 실패: {e}")
