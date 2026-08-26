/*
 * IMU data collector -- streams raw MPU6050 samples over Serial in the exact
 * CSV schema the training pipeline expects:
 *   recording_id,label,t_ms,ax,ay,az,gx,gy,gz
 *
 * How to record a real gesture:
 *   1. Set CURRENT_LABEL below to "idle" / "wave" / "circle" / "punch".
 *   2. Flash this sketch, open Serial Monitor (or Serial Plotter) at 115200 baud.
 *   3. Press the button on GPIO 0 (BOOT button on most ESP32 dev boards),
 *      perform the gesture for exactly WINDOW_MS milliseconds, then stop.
 *   4. Copy the printed CSV rows into ../../data/imu_gesture_log_real.csv
 *      (append -- don't overwrite the synthetic file).
 *   5. Repeat for ~20-50 recordings per class, ideally from more than one
 *      person, to get data that generalizes better than the synthetic set.
 *
 * Wiring: see ../../wiring/WIRING_GUIDE.md. MPU6050 on I2C:
 *   VCC -> 3V3, GND -> GND, SCL -> GPIO22, SDA -> GPIO21 (ESP32 DevKit default).
 */
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define BUTTON_PIN 0        // BOOT button, active LOW on most ESP32 dev boards
#define SAMPLE_RATE_HZ 50
#define WINDOW_MS 1000
#define SAMPLES_PER_WINDOW (SAMPLE_RATE_HZ * WINDOW_MS / 1000)

const char* CURRENT_LABEL = "idle";  // change before each recording batch

Adafruit_MPU6050 mpu;
int recording_id = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin();
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found -- check wiring (SDA=21, SCL=22, VCC=3V3)");
    while (1) { delay(1000); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);

  Serial.println("recording_id,label,t_ms,ax,ay,az,gx,gy,gz");
  Serial.println("# ready. hold BOOT button and perform the gesture.");
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(30);  // debounce
    unsigned long start = millis();
    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
      unsigned long t0 = millis();
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      // a.acceleration is m/s^2 -- convert to g to match the synthetic dataset
      float ax = a.acceleration.x / 9.80665;
      float ay = a.acceleration.y / 9.80665;
      float az = a.acceleration.z / 9.80665;
      // g.gyro is rad/s -- convert to deg/s to match the synthetic dataset
      float gx = g.gyro.x * 57.2958;
      float gy = g.gyro.y * 57.2958;
      float gz = g.gyro.z * 57.2958;

      Serial.print(recording_id); Serial.print(",");
      Serial.print(CURRENT_LABEL); Serial.print(",");
      Serial.print(i * (1000 / SAMPLE_RATE_HZ)); Serial.print(",");
      Serial.print(ax, 5); Serial.print(",");
      Serial.print(ay, 5); Serial.print(",");
      Serial.print(az, 5); Serial.print(",");
      Serial.print(gx, 3); Serial.print(",");
      Serial.print(gy, 3); Serial.print(",");
      Serial.println(gz, 3);

      long elapsed = millis() - t0;
      long remaining = (1000 / SAMPLE_RATE_HZ) - elapsed;
      if (remaining > 0) delay(remaining);
    }
    Serial.print("# recording "); Serial.print(recording_id);
    Serial.println(" done. release button, wait 1s, press again for the next one.");
    recording_id++;
    delay(1000);
    while (digitalRead(BUTTON_PIN) == LOW) { delay(10); }  // wait for release
  }
}
