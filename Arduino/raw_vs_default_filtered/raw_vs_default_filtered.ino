// Basic demo for readings from Adafruit BNO08x
#include <Adafruit_BNO08x.h>

// For SPI mode, we need a CS pin
#define BNO08X_CS 10
#define BNO08X_INT 9

// For SPI mode, we also need a RESET
//#define BNO08X_RESET 5
// but not for I2C or UART
#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit BNO08x test!");

  // Try to initialize!
  if (!bno08x.begin_I2C()) {
    // if (!bno08x.begin_UART(&Serial1)) {  // Requires a device with > 300 byte
    // UART buffer! if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) {
      delay(10);
    }
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
}


void setReports(void) {

  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(SH2_ACCELEROMETER)) {
    Serial.println("Could not enable accelerometer");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED)) {
    Serial.println("Could not enable gyroscope");
  }
  if (!bno08x.enableReport(SH2_RAW_ACCELEROMETER)) {
    Serial.println("Could not enable raw accelerometer");
  }
  if (!bno08x.enableReport(SH2_RAW_GYROSCOPE)) {
    Serial.println("Could not enable raw gyroscope");
  }
}
void loop() {
  delay(10);

    if (bno08x.wasReset()) {
    Serial.print("sensor was reset ");
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  switch (sensorValue.sensorId) {
    return;
  }

  switch (sensorValue.sensorId) {

case SH2_ACCELEROMETER:
    Serial.print("Accelerometer - x: ");
    Serial.print(sensorValue.un.accelerometer.x);
    Serial.print(" y: ");
    Serial.print(sensorValue.un.accelerometer.y);
    Serial.print(" z: ");
    Serial.println(sensorValue.un.accelerometer.z);
    break;
  case SH2_GYROSCOPE_CALIBRATED:
    Serial.print("Gyro - x: ");
    Serial.print(sensorValue.un.gyroscope.x);
    Serial.print(" y: ");
    Serial.print(sensorValue.un.gyroscope.y);
    Serial.print(" z: ");
    Serial.println(sensorValue.un.gyroscope.z);
    break;

case SH2_RAW_ACCELEROMETER:
    Serial.print("Raw Accelerometer - x: ");
    Serial.print(sensorValue.un.rawAccelerometer.x);
    Serial.print(" y: ");
    Serial.print(sensorValue.un.rawAccelerometer.y);
    Serial.print(" z: ");
    Serial.println(sensorValue.un.rawAccelerometer.z);
    break;
  case SH2_RAW_GYROSCOPE:
    Serial.print("Raw Gyro - x: ");
    Serial.print(sensorValue.un.rawGyroscope.x);
    Serial.print(" y: ");
    Serial.print(sensorValue.un.rawGyroscope.y);
    Serial.print(" z: ");
    Serial.println(sensorValue.un.rawGyroscope.z);
    break;

  }
  // put your main code here, to run repeatedly:

}
