# Wiring Guide — TinyML Gesture Classifier

Bill of materials (~$25-30):
- 1x ESP32 DevKitC (or any ESP32 dev board with exposed I2C pins)
- 1x MPU6050 accelerometer+gyro breakout (I2C)
- Breadboard + jumper wires
- USB cable (data-capable, not charge-only)
- A multimeter (for the power-measurement step — see below)

## Connections

| MPU6050 pin | ESP32 pin  | Notes                              |
|-------------|------------|-------------------------------------|
| VCC         | 3V3        | MPU6050 is 3.3V logic — do NOT use 5V |
| GND         | GND        |                                      |
| SCL         | GPIO 22    | ESP32 default I2C clock              |
| SDA         | GPIO 21    | ESP32 default I2C data               |
| XDA, XCL, AD0, INT | not connected | not needed for this project   |

```
        ESP32 DevKitC                      MPU6050
       ┌───────────────┐                 ┌───────────┐
       │            3V3 ├────────────────┤ VCC       │
       │            GND ├────────────────┤ GND       │
       │        GPIO 22 ├────────────────┤ SCL       │
       │        GPIO 21 ├────────────────┤ SDA       │
       │       (BOOT/G0)│  <- data-collector uses this button
       └───────────────┘                 └───────────┘
```

## Bring-up checklist (do this before trusting any code)

1. Wire it per the table above. Double check VCC goes to **3V3**, not 5V — the
   MPU6050 breakout has no level shifter on most cheap boards and 5V can damage it.
2. Plug in over USB. `arduino-cli board list` (or the Arduino IDE's Tools > Port
   menu) should show a new serial port appear.
3. Flash `firmware/data_collector/data_collector.ino` first, not the inference
   sketch — it's the simplest possible thing that proves the wiring works.
4. Open the serial monitor at 115200 baud. You should see the CSV header line
   and then a "ready" message. If instead you see `MPU6050 not found`, the two
   most common causes are: SDA/SCL swapped, or a breadboard row that isn't
   actually making contact (push the jumpers in further, or move to a different
   row) — check with the multimeter's continuity/beep mode across the two
   suspect pins before reflashing anything.
5. Hold the BOOT button and move the board. Confirm the ax/ay/az/gx/gy/gz
   numbers actually change and roughly make sense (az near 1.0 = 1g of gravity
   when the board is flat and still, others near 0).

This bring-up step is the one part of this project I (the assistant building
this repo) cannot do for you — I have no physical MPU6050 or ESP32 on this
machine. Everything in `../model/` and the firmware's C++ was written and, where
a toolchain was installable, actually compiled on this machine (see the root
README's "what's verified vs what needs your bench" section). The soldering,
the continuity check, and the "does it actually read sensible numbers when I
wave my arm" moment are yours to run and are genuinely how you'd debug this on
the job — not a step to skip.

## Power measurement (the "on-device inference" resume line)

The stack line in this project's brief explicitly calls out **power
measurement** — this is what turns "I ran TensorFlow Lite on a microcontroller"
into "I measured what running an ML model on a microcontroller actually costs
in power," which is the more senior claim.

1. Put your multimeter in **DC current** mode, in series with the ESP32's power
   input (cut the 5V/VIN line between your USB power source and the board, put
   the multimeter's leads across the cut — or use a USB power meter inline if
   you have one, which is easier and safer than cutting a wire).
2. Flash `firmware/tinyml_gesture/tinyml_gesture.ino`. Let it sit idle (no
   gesture) and record the average current draw over ~10 seconds. This is your
   **idle/baseline** current.
3. Perform a gesture repeatedly for ~10 seconds and record the average current
   draw again. This is your **inference-active** current.
4. Subtract: `inference_current - idle_current` is roughly the marginal cost of
   running the model, not just the WiFi radio / CPU idle loop / MPU6050 polling
   that would be there anyway.
5. Multiply by your supply voltage (5V from USB) to get power in watts, and by
   your battery's mAh rating to estimate runtime if you were to run this off a
   battery instead of USB.

Write your actual numbers down — `knowledge/09_power_and_the_resume_line.md`
has a worked example with placeholder numbers and explains why "I measured
X mA idle vs Y mA during inference" is a genuinely differentiating line in an
embedded-AI interview, and why almost no candidate actually has real numbers
here instead of a spec-sheet guess.
