
// #include <SPI.h>
#include <RH_RF95.h>
#include <Adafruit_BNO08x.h>

#define BNO08X_CS 10
#define BNO08X_INT 9

#define BNO08X_RESET -1

Adafruit_BNO08x  bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// First 3 here are boards w/radio BUILT-IN. Boards using FeatherWing follow.
#if defined (__AVR_ATmega32U4__)  // Feather 32u4 w/Radio
  #define RFM95_CS    8
  #define RFM95_INT   7
  #define RFM95_RST   4

#elif defined(ARDUINO_ADAFRUIT_FEATHER_RP2040_RFM)  // Feather RP2040 w/Radio
  #define RFM95_CS   16
  #define RFM95_INT  21
  #define RFM95_RST  17

#endif

#define RF95_FREQ 915.0

// Singleton instance of the radio driver
RH_RF95 rf95(RFM95_CS, RFM95_INT);

void setup(void) {
  Serial.begin(115200);
  while (!Serial) delay(10);

  int atmpt=0;
  while (!bno08x.begin_I2C()) {
    Serial.print("Failed to find BNO08x chip... tried ");
    Serial.print(++atmpt);
    Serial.println(" times.");
    // while (1) { delay(10); }
    delay(500);
  }
  Serial.println("BNO08x Found!");

for (int n = 0; n < bno08x.prodIds.numEntries; n++) {
    Serial.print("Part ");
    Serial.print(bno08x.prodIds.entry[n].swPartNumber);
    Serial.print(": Version :");
    Serial.print(bno08x.prodIds.entry[n].swVersionMajor);
    Serial.print(".");
    Serial.print(bno08x.prodIds.entry[n].swVersionMinor);
    Serial.print(".");
    Serial.print(bno08x.prodIds.entry[n].swVersionPatch);
    Serial.print(" Build ");
    Serial.println(bno08x.prodIds.entry[n].swBuildNumber);
  }

  setReports();

  Serial.println("Reading events");
  delay(100);



  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH);

  // while (!Serial) delay(1);
  // delay(100);

  Serial.println("Feather LoRa TX Test!");

  // manual reset
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);

  while (!rf95.init()) {
    Serial.println("LoRa radio init failed");
    Serial.println("Uncomment '#define SERIAL_DEBUG' in RH_RF95.cpp for detailed debug info");
    while (1);
  }
  Serial.println("LoRa radio init OK!");

  // Defaults after init are 434.0MHz, modulation GFSK_Rb250Fd250, +13dbM
  if (!rf95.setFrequency(RF95_FREQ)) {
    Serial.println("setFrequency failed");
    while (1);
  }
  Serial.print("Set Freq to: "); Serial.println(RF95_FREQ);

  rf95.setTxPower(23, false);
}

// int16_t packetnum = 0;  // packet counter, we increment per xmission

void setReports(void) {
  Serial.println("Setting desired reports");
  if (! bno08x.enableReport(SH2_GAME_ROTATION_VECTOR)) {
    Serial.println("Could not enable game vector");
  }
}

float data[4];

void loop() {

  if (! bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  // switch (sensorValue.sensorId) {
    
  //   case SH2_GAME_ROTATION_VECTOR:
  //     Serial.print("Game Rotation Vector - r: ");
  //     Serial.print(sensorValue.un.gameRotationVector.real);
  //     Serial.print(" i: ");
  //     Serial.print(sensorValue.un.gameRotationVector.i);
  //     Serial.print(" j: ");
  //     Serial.print(sensorValue.un.gameRotationVector.j);
  //     Serial.print(" k: ");
  //     Serial.println(sensorValue.un.gameRotationVector.k);
  //     break;
  // }

    // delay(500); // Wait 1 second between transmits, could also 'sleep' here!
  // Serial.println("Transmitting..."); // Send a message to rf95_server

  // char radiopacket[50];
  // // Format the imu data as a string with four decimal places for each value
  // sprintf(radiopacket, "%.4f %.4f %.4f %.4f %d", sensorValue.un.gameRotationVector.real, sensorValue.un.gameRotationVector.i, sensorValue.un.gameRotationVector.j, sensorValue.un.gameRotationVector.k, packetnum++);
  // rf95.send((uint8_t *)radiopacket, sizeof(radiopacket));
  data[0] = sensorValue.un.gameRotationVector.real;
  data[1] = sensorValue.un.gameRotationVector.i;
  data[2] = sensorValue.un.gameRotationVector.j;
  data[3] = sensorValue.un.gameRotationVector.k;
  // data[4] = packetnum++;

  rf95.send((uint8_t *) data, sizeof(data));
  
  // Serial.println("Sending...");
  // delay(1);

  // Serial.println("Waiting for packet to complete...");
  // delay(1);
  rf95.waitPacketSent();
  // Now wait for a reply
  // uint8_t buf[RH_RF95_MAX_MESSAGE_LEN];
  // uint8_t len = sizeof(buf);

  // Serial.println("Waiting for reply...");
  // if (rf95.waitAvailableTimeout(10)) {
  //   // Should be a reply message for us now
  //   if (rf95.recv(buf, &len)) {
  //     Serial.print("Got reply: ");
  //     Serial.println((char*)buf);
  //     Serial.print("RSSI: ");
  //     Serial.println(rf95.lastRssi(), DEC);
  //   } else {
  //     Serial.println("Receive failed");
  //   }
  // } else {
  //   Serial.println("No reply, is there a listener around?");
  // }

}
