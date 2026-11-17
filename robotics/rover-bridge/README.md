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

Bench-tested with the wheels lifted: `rover_link.py` drive, and the node driving forward and turning left from `/cmd_vel`, stopping on its own when the messages stop. `max_speed` is still uncalibrated and nothing has been run on the floor.
