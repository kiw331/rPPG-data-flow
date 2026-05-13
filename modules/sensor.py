# 파일명: sensor.py
# 목적: 하드웨어 센서 데이터를 시리얼 통신으로 읽어오고 파싱하는 스레드 모듈

import time
import serial
import struct
import queue
from PyQt5.QtCore import QThread, pyqtSignal

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
        self.last_sample_time = 0
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
            self.last_sample_time = 0
            
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

                            interval = 1.0 / 200.0
                            actual_time = time.time()
                            current_time = max(actual_time, self.last_sample_time + interval)
                            self.last_sample_time = current_time

                            self.gui_q.append((current_time, ir_val))

                            if actual_time - self.last_check_time >= 1.0:
                                self.update_stats_signal.emit(self.packet_count_1s, self.drop_count)
                                self.packet_count_1s = 0
                                self.last_check_time = actual_time

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
