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
| `remote.html` | the phone page: touch joystick, speed slider, STOP button |
| `run_remote.sh` | starts the bridge and the phone remote together |

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
- The speed slider (10-100%) scales the stick. Defaults top out at 0.2 m/s and 1.5 rad/s, which the bridge turns into about 100/255 motor power.

| Remote parameter | Default | Meaning |
|------------------|---------|---------|
| `port` | 8080 | web server port |
| `max_linear` | 0.2 | m/s at full stick and 100% slider |
| `max_angular` | 1.5 | rad/s at full sideways stick and 100% slider |

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
| `send_rate` | 10 | Hz, must stay well inside the firmware failsafe (500 ms) |
| `cmd_timeout` | 0.5 | seconds without `/cmd_vel` before the rover is stopped |

`linear.x` and `angular.z` follow the ROS convention: positive angular is a left turn (counter-clockwise).

## Calibrating `max_speed`

The rover has no encoders, so the node assumes speed is proportional to the command. On the floor, run `rover_link.py drive 100 100 --seconds 3`, measure the distance `d` in metres, then `max_speed = (d / 3) * 255 / 100`.

## Safety

- The firmware stops the motors if `V` commands stop for 500 ms, and the node also stops them when `/cmd_vel` goes quiet or the node exits.
- The default 100/255 cap keeps first tests slow. Raise `speed_limit` once the rover is on the floor.
- Wheels off the ground for any new test.

## Notes

- The Jetson's header UART corrupts replies of 16 bytes or more (every full 16-byte block comes back as zeros). `OK`, `ERR`, `PONG` and `FAILSAFE` are short enough; the `ID`, `?` and `C ?` replies are not readable here.
- The first byte after opening the port is lost, so `RoverLink` sends a newline and flushes first.

## Status

Bench-tested with the wheels lifted: `rover_link.py` drive, and the node driving forward and turning left from `/cmd_vel`, stopping on its own when the messages stop. The phone remote was driven from a phone on the floor: the rover moves only weakly and does not turn, and motor torque is the limit, not the software. `max_speed` is still uncalibrated.
