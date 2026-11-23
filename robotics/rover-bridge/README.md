# Rover Bridge — ROS 2 drive link from the Jetson to the rover

Python on the Jetson that turns `/cmd_vel` messages into drive commands for the rover's STM32 controller over a UART.

## Wiring

Jetson 40-pin header to the Blue Pill, 3.3 V logic, common ground. Pins 2 and 4 are 5 V: leave them alone.

| Jetson pin | Signal | Blue Pill |
|------------|--------|-----------|
| 6  | GND     | G   |
| 8  | UART TX | PB7 |
| 10 | UART RX | PB6 |

The port is `/dev/ttyTHS1`. The `jetson` user is already in `dialout`, so no `sudo` is needed.

## Files

| File | Purpose |
|------|---------|
| `rover_link.py` | `RoverLink` class (`ping`, `drive`, `stop`, `configure`) and a small command-line test |
| `rover_bridge.py` | ROS 2 node: `/cmd_vel` to `V left right`, with a timeout stop and automatic reconnect |
| `rover_remote.py` | ROS 2 node: serves a phone web page and publishes the joystick as `/cmd_vel` |
| `remote.html` | the phone page: live camera or depth view with the touch joystick over it, speed slider, STOP button |
| `run_remote.sh` | starts the IMU node, the bridge and the phone remote together |
| `mpu6050.py` | reads the MPU-6050 (GY-521) over I2C with no extra libraries; `probe` and `axes` modes |
| `rover_imu.py` | ROS 2 node: publishes the IMU as `/imu/data_raw` at 100 Hz, gyro bias removed |
| `rover_autostart.sh` | supervisor: starts the IMU node (if present), bridge and phone remote and restarts them if one stops; also runs the phone sensor driver (SensorStream) on its own restart loop |
| `install_autostart.sh` | adds a user-cron `@reboot` line so the rover software starts at boot (`remove` undoes it) |
| `setup_ap.sh` | one-time: gives the Jetson its own Wi-Fi network (`rover-jetson`) as a last-resort fallback and sets the network order |
| `check_network.sh` | shows the Wi-Fi connection, address, whether the remote answers, and the phone URL |
| `calibrate_deadband.py` | ramps both sides until you stop it, to find the smallest command that moves the rover |
| `motor_test.py` | pulses each motor in turn to check wheel position and direction (stop the rover software first) |
| `turn_control.py` | pure-Python turn controller (gyro feedback, saturation, spin and arc modes) |

Copy both to the Jetson (for example `~/rover_bridge/`) and run from that folder. ROS 2 Humble and `pyserial` are needed.

## Run

```bash
cd ~/rover_bridge
python3 rover_link.py ping                         # PONG
python3 rover_link.py drive 60 60 --seconds 2      # wheels lifted

source /opt/ros/humble/setup.bash
python3 rover_bridge.py
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.15}}"
```

## Phone remote

Drive the rover from a phone browser on the same Wi-Fi as the Jetson.

```bash
bash run_remote.sh        # on the Jetson; prints the URL
```

Open `http://<jetson-ip>:8080/?key=<key>` (or `http://ubuntu.local:8080/?key=<key>`). The access key is created on first start and kept in `~/.rover_remote_key`, so the URL stays the same. In Safari, Share then Add to Home Screen gives a full-screen controller.

- The joystick only drives while a finger is on it. Letting go sends a stop; locking the phone, switching apps or losing Wi-Fi stops the rover too, and the node stops it after 0.4 s without drive messages.
- The page and the API (`GET /ping`, `POST /drive`, `POST /stop`) refuse requests without the key. There is no encryption, so anyone on the network who has the key can drive the rover: keep it private.
- **Live view:** the page shows the phone sensor's camera behind the stick, with buttons for Colour, Depth (a heat map, red near and blue far, 0.2 to 4 m, black where there is no reading) and Off, and the Jetson's battery voltage next to the link light. The Jetson decodes each JPEG at half size (1920x1440 to 960x720) and sends motion JPEG at 12 fps per browser, about 12 ms of one core per frame. If the phone stops sending, the page says "no video from the phone" and reopens the stream when frames return. Parameters `video_fps`, `video_reduce`, `video_quality`, `depth_max_mm`. The depth units are assumed to be millimetres (to be verified with a tape measure).
- The speed slider (10-100%) scales the stick. Defaults top out at 0.2 m/s and 1.5 rad/s, which the bridge turns into about 100/255 motor power.

| Remote parameter | Default | Meaning |
|------------------|---------|---------|
| `port` | 8080 | web server port |
| `max_linear` | 0.2 | m/s at full stick and 100% slider |
| `max_angular` | 1.5 | rad/s at full sideways stick and 100% slider |
| `video_fps` | 12 | frames per second sent to each browser |
| `video_reduce` | 2 | the JPEG is decoded at 1/N size (1, 2, 4 or 8) |
| `video_quality` | 70 | JPEG quality of the frames sent to the browser |
| `depth_max_mm` | 4000 | farthest depth shown in the Depth view |

For full motor power on the floor, tune both nodes so full stick equals full command:

```bash
python3 rover_bridge.py --ros-args -p speed_limit:=255 -p max_speed:=0.25
python3 rover_remote.py --ros-args -p max_linear:=0.25 -p max_angular:=2.5
```

Full power with the motors stalled against the floor draws a lot of current from the pack, which the Jetson shares: if the pack's protection trips, the Jetson loses power too.

## Node parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `port` | `/dev/ttyTHS1` | serial device |
| `track_width` | 0.24 | metres, centre of the left wheel to centre of the right wheel |
| `max_speed` | 0.5 | m/s at full command (255). **Placeholder, calibrate it** |
| `speed_limit` | 100 | cap sent to the firmware at start-up, out of 255 |
| `send_rate` | 20 | Hz, the turn loop runs at this rate; must stay well inside the firmware failsafe (500 ms) |
| `cmd_timeout` | 0.5 | seconds without `/cmd_vel` before the rover is stopped |

`linear.x` and `angular.z` follow the ROS convention: positive angular is a left turn (counter-clockwise).

## Closed-loop turning (IMU)

A GY-521 (MPU-6050) on the Jetson's I2C bus lets the bridge close the loop on the turn rate. There are no encoders, so the loop adds power until the gyro shows the requested rate, and the output saturates at the motor command range (±255), the PWM resolution. That is why no wheel-speed estimate is needed.

| Jetson pin | IMU |
|------------|-----|
| 1 (3.3 V) | VCC |
| 3 (SDA, `/dev/i2c-7`) | SDA |
| 5 (SCL) | SCL |
| 9 (GND) | GND |

Power it from 3.3 V: the board's pull-ups tie to VCC, and 5 V would reach the Jetson's pins. On this rover the board's axes are X forward, Y left, Z up (the ROS convention), so a positive gyro Z is a left turn, the same sign as `/cmd_vel`. `python3 mpu6050.py axes` finds the axes from a nose-up and a right-side-down tilt.

```bash
python3 rover_imu.py &        # keep the rover still for the first 2 s (rover_autostart.sh starts it only if the chip answers: USE_IMU=auto)
python3 rover_bridge.py       # closed loop when /imu/data_raw is flowing, open loop otherwise
```

The command is the kinematic feed-forward plus `kp * error + ki * integral(error)` on the gyro error, optionally minus a weighted accelerometer damping term (`acc_weight`, off by default). The integrator stops growing while the output is saturated. Turn modes (`turn_mode`):

| Mode | Behaviour |
|------|-----------|
| `open` | feed-forward only, no feedback |
| `spin` | the wheels counter-rotate for a turn on the spot |
| `arc` | the inner wheel never reverses, so the rover turns while rolling forward and needs far less torque |
| `auto` | spin first; if the output sits at full power for `stall_time` (1 s) without the gyro reaching half of the requested rate, switch to an arc until the turn request ends |

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `turn_mode` | `auto` | see above |
| `turn_kp` | 50 | command units per rad/s of error |
| `turn_ki` | 120 | command units per rad of accumulated error |
| `acc_weight` | 0 | weight of the accelerometer damping term |
| `stall_time` | 1.0 | seconds at full power without progress before `auto` switches |
| `arc_ratio` | 1.5 | forward speed of an arc relative to the pivot speed |
| `imu_offset` | 0.07 | metres the IMU sits ahead of the turning centre |
| `imu_timeout` | 0.3 | seconds without IMU data before falling back to open loop |

A feedback loop cannot add torque: if full power cannot turn the rover, the loop sits at saturation, which is what the `auto` fallback to an arc is for. The loop was simulated against model rovers with weak torque and checked on the rover with the wheels lifted (spin ramps to full power, then the fallback to an arc).

## Start at boot and Wi-Fi

`bash install_autostart.sh` once on the Jetson; from then on the software starts about 20 s after power-on (keep the rover still for the first few seconds, the gyro bias is measured then). Settings are at the top of `rover_autostart.sh`; logs are in `~/rover_logs/`. The Jetson joins any saved Wi-Fi, including a phone hotspot (`nmcli dev wifi connect "<name>" --ask`). On a hotspot the address changes, so open `http://ubuntu.local:8080/?key=<key>`.

### The Jetson's own Wi-Fi network

The Jetson hosts its own network, `rover-jetson`, so the rover works anywhere with no router or phone hotspot: switch it on, join `rover-jetson` on the phone, open `http://10.42.0.1:8080/?key=<key>`. Run `ssh -t orin "bash ~/rover_bridge/setup_ap.sh"` once (it asks for the Jetson's sudo password). It creates the network (WPA2, 2.4 GHz, a random password kept in `~/.rover_ap_password`, the Jetson at `10.42.0.1`) and gives it the highest priority, so it comes up at every boot, together with the rover software (started by cron). Other modes:

| Command | Effect |
|---------|--------|
| `bash setup_ap.sh` | own network on at every boot (default) |
| `bash setup_ap.sh fallback` | own network only when no saved network is in range |
| `bash setup_ap.sh online` | own network off; join the best saved network now (for apt, pip, updates) |
| `bash setup_ap.sh remove` | delete the own network |

Add `now` to switch immediately. While the Jetson hosts the network it has no internet. A phone keeps its own mobile data for internet. Waveshare's `ugv_jetson` does the same job with AccessPopup, which checks every 2 minutes and only starts its hotspot when no known network is in range. `bash check_network.sh` shows the current state.

### Phone sensor driver

`rover_autostart.sh` also starts the SensorStream driver (`USE_PHONE=1`, built in `~/sensorstream_ws`) so it is ready after every boot. It listens on port 5678; join the phone to `rover-jetson`, open the SensorStream app and connect to `10.42.0.1:5678` (the driver also prints a QR code in `~/rover_logs/sensorstream.log`). Topics: `color_image/compressed`, `depth_image`, `imu/data`, `gps/fix`. Measured over the Jetson's own network: colour and depth about 30 fps, IMU about 49 Hz, 5 ms round trip. The listener has no authentication, so it relies on the Wi-Fi password. The phone's timestamps are not ROS time (the Jetson's clock has no source to set it), so consumers must restamp on arrival. The driver has its own restart loop, so a driver crash does not stop the drive software.

## Dead zone

Below a certain command the wheels do not turn at all. `deadband` (default 60 of 255) lifts every non-zero wheel command over that zone, so the stick's low end moves the rover and the feed-forward is no longer too small. Find your value with `calibrate_deadband.py` and set `DEADBAND` in `rover_autostart.sh`.

## Calibrating `max_speed`

The rover has no encoders, so the node assumes speed is proportional to the command. On the floor, run `rover_link.py drive 100 100 --seconds 3`, measure the distance `d` in metres, then `max_speed = (d / 3) * 255 / 100`.

## Safety

- The firmware stops the motors if `V` commands stop for 500 ms, and the node also stops them when `/cmd_vel` goes quiet or the node exits.
- The default 100/255 cap keeps first tests slow. Raise `speed_limit` once the rover is on the floor.
- Wheels off the ground for any new test.

## Notes

- The Jetson's header UART corrupts replies of 16 bytes or more (every full 16-byte block comes back as zeros). `OK`, `ERR`, `PONG` and `FAILSAFE` are short enough; the `ID`, `?` and `C ?` replies are not readable here.
- `rover_link.py raw "T 1 80"` cannot test a motor: closing the port sends a stop that cancels the 1.5 s pulse. Use `motor_test.py`, which keeps the port open.
- The first byte after opening the port is lost, so `RoverLink` sends a newline and flushes first.

## Status

Bench-tested with the wheels lifted: `rover_link.py` drive, and the node driving forward and turning left from `/cmd_vel`, stopping on its own when the messages stop. The phone remote was driven from a phone on the floor: the rover moves only weakly and does not turn, and motor torque is the limit, not the software. `max_speed` is still uncalibrated. The IMU is verified (100 Hz, X forward, Y left, Z up, left turn reads positive) and the turn controller passed its simulation and a lifted-wheel run (spin saturates, then `auto` falls back to an arc); it has not yet been tried on the floor.
