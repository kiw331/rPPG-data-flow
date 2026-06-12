/*
 * ESP32 PPG Monitor (Dual Core + Binary Protocol) - v7
 * - Protocol: [0xA5][0x5A][SEQ(1)][IR(4)][RED(4)] = Total 11 Bytes
 * - Brightness command (ASCII, '\n' terminated):
 *     "I<val>"  -> IR  LED amplitude (0~255)
 *     "R<val>"  -> RED LED amplitude (0~255)
 *     "<val>"   -> (legacy) IR LED amplitude
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
#define PACKET_SIZE      11

MAX30105 particleSensor;
QueueHandle_t ppgQueue;
volatile int commandBrightnessIR  = -1;
volatile int commandBrightnessRed = -1;

// 큐 데이터 구조체
struct PpgPacket {
  uint32_t ir;
  uint32_t red;
  uint8_t  seq;
};

// ======================== [Core 0] Binary Tx Task ========================
void TaskSerialTx(void *pvParameters) {
  PpgPacket packet;
  // 바이너리 패킷 버퍼: [Header1][Header2][Seq][IR_4bytes][RED_4bytes]
  uint8_t txBuf[PACKET_SIZE];
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

      // RED값(4바이트) 쪼개서 넣기 (Little Endian)
      txBuf[7]  = (uint8_t)(packet.red & 0xFF);
      txBuf[8]  = (uint8_t)((packet.red >> 8) & 0xFF);
      txBuf[9]  = (uint8_t)((packet.red >> 16) & 0xFF);
      txBuf[10] = (uint8_t)((packet.red >> 24) & 0xFF);

      // 11바이트 한 번에 전송 (가장 효율적)
      Serial.write(txBuf, PACKET_SIZE);

    } else {
      vTaskDelay(1);
    }

    // [밝기 명령 수신] "I<val>" / "R<val>" / "<val>"(legacy=IR)
    if (Serial.available() > 0) {
      char prefix = Serial.peek();
      if (prefix == 'I' || prefix == 'R') {
        Serial.read(); // prefix 소비
        int val = Serial.parseInt();
        while (Serial.available()) Serial.read();
        if (val >= 0 && val <= 255) {
          if (prefix == 'I') commandBrightnessIR = val;
          else               commandBrightnessRed = val;
        }
      } else {
        int val = Serial.parseInt();
        while (Serial.available()) Serial.read();
        if (val >= 0 && val <= 255) commandBrightnessIR = val;
      }
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
  byte ledMode = 2;        // 2 = Red + IR
  int sampleRate = 200;
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.enableFIFORollover();

  xTaskCreatePinnedToCore(TaskSerialTx, "SerialTx", 4096, NULL, 1, NULL, 0);
}

void loop() {
  static uint8_t sequence = 0; // 0~255 자동 순환

  if (commandBrightnessIR != -1) {
    particleSensor.setPulseAmplitudeIR((byte)commandBrightnessIR);
    commandBrightnessIR = -1;
  }
  if (commandBrightnessRed != -1) {
    particleSensor.setPulseAmplitudeRed((byte)commandBrightnessRed);
    commandBrightnessRed = -1;
  }

  particleSensor.check();

  while (particleSensor.available()) {
    PpgPacket packet;
    packet.ir  = particleSensor.getFIFOIR();
    packet.red = particleSensor.getFIFORed();
    packet.seq = sequence++; // 0~255, 255 다음엔 0으로

    particleSensor.nextSample();
    xQueueSend(ppgQueue, &packet, 0);
  }
  delay(1);
}
