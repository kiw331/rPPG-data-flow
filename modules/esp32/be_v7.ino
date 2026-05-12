/*
 * ESP32 PPG Monitor (Dual Core + Binary Protocol)
 * - Protocol: [0xA5][0x5A][SEQ(1)][IR(4)] = Total 7 Bytes
 */

#include <Wire.h>
#include "MAX30105.h" 
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ===== [설정] =====
#define I2C_SPEED        400000  
#define SERIAL_BAUD      115200  // 바이너리라 115200으로도 충분함 (물론 921600 쓰면 더 좋음)
#define QUEUE_SIZE       256     

MAX30105 particleSensor;
QueueHandle_t ppgQueue;
volatile int commandBrightness = -1; 

// 큐 데이터 구조체
struct PpgPacket {
  uint32_t ir;
  uint8_t seq;
};

// ======================== [Core 0] Binary Tx Task ========================
void TaskSerialTx(void *pvParameters) {
  PpgPacket packet;
  // 바이너리 패킷 버퍼: [Header1][Header2][Seq][IR_4bytes]
  uint8_t txBuf[7]; 
  txBuf[0] = 0xA5; // Sync Byte 1
  txBuf[1] = 0x5A; // Sync Byte 2

  while (true) {
    if (xQueueReceive(ppgQueue, &packet, (TickType_t)10) == pdPASS) {
      // 데이터 채우기
      txBuf[2] = packet.seq;
      
      // IR값(4바이트) 쪼개서 넣기 (Little Endian)
      txBuf[3] = (uint8_t)(packet.ir & 0xFF);
      txBuf[4] = (uint8_t)((packet.ir >> 8) & 0xFF);
      txBuf[5] = (uint8_t)((packet.ir >> 16) & 0xFF);
      txBuf[6] = (uint8_t)((packet.ir >> 24) & 0xFF);

      // 7바이트 한 번에 전송 (가장 효율적)
      Serial.write(txBuf, 7);
      
    } else {
      vTaskDelay(1);
    }

    // [밝기 명령 수신] 기존과 동일
    if (Serial.available() > 0) {
      int val = Serial.parseInt();
      while(Serial.available()) Serial.read();
      if (val >= 0 && val <= 255) commandBrightness = val;
    }
  }
}

// ======================== [Setup & Core 1] ========================
void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();
  
  ppgQueue = xQueueCreate(QUEUE_SIZE, sizeof(PpgPacket));

  if (!particleSensor.begin(Wire, I2C_SPEED)) {
    // 에러 상황도 바이너리로 보내면 꼬일 수 있으니, LED로 표시하거나 일단 넘어감
    while (1);
  }

  byte ledBrightness = 60; 
  byte sampleAverage = 1; 
  byte ledMode = 2;        
  int sampleRate = 200;    
  int pulseWidth = 411;    
  int adcRange = 4096;     

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.enableFIFORollover();

  xTaskCreatePinnedToCore(TaskSerialTx, "SerialTx", 4096, NULL, 1, NULL, 0);
}

void loop() {
  static uint8_t sequence = 0; // 0~255 자동 순환

  if (commandBrightness != -1) {
    particleSensor.setPulseAmplitudeIR((byte)commandBrightness);
    commandBrightness = -1;
  }
  
  particleSensor.check();

  while (particleSensor.available()) {
    PpgPacket packet;
    packet.ir = particleSensor.getFIFOIR(); 
    packet.seq = sequence++; // 0~255, 255 다음엔 0으로
    
    particleSensor.nextSample();
    xQueueSend(ppgQueue, &packet, 0); 
  }
  delay(1); 
}