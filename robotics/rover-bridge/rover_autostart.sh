#!/bin/bash
# Keeps the rover software running: IMU node, bridge and phone remote, restarted if any of them stops.
# Started at boot by cron (see install_autostart.sh). Keep the rover still for the first 3 s after power-on.
# Settings live here; change them and reboot (or kill this script and run it again).
SPEED_LIMIT=255      # cap sent to the firmware, out of 255
MAX_SPEED=0.25       # m/s at full command, placeholder until calibrated
DEADBAND=60.0          # motor command below which the wheels do not move (calibrate_deadband.py)
MAX_LINEAR=0.25      # m/s at full stick
MAX_ANGULAR=2.5      # rad/s at full stick
USE_IMU=auto         # auto: start the IMU node only if the MPU-6050 answers; 1: always; 0: never (turns open loop)

cd "$(dirname "$0")" || exit 1
exec 9>/tmp/rover_autostart.lock
flock -n 9 || exit 0                       # already running

source /opt/ros/humble/setup.bash
LOG="$HOME/rover_logs"
mkdir -p "$LOG"

imu_present() {
    case "$USE_IMU" in
        1) return 0 ;;
        0) return 1 ;;
        *) python3 -c "from mpu6050 import MPU6050; MPU6050(7)" > /dev/null 2>&1 ;;
    esac
}

while true; do
    for f in imu bridge remote; do          # keep the logs from growing for ever
        [ -f "$LOG/$f.log" ] && tail -n 2000 "$LOG/$f.log" > "$LOG/$f.tmp" && mv "$LOG/$f.tmp" "$LOG/$f.log"
    done

    IMU=""
    if imu_present; then
        python3 rover_imu.py >> "$LOG/imu.log" 2>&1 &
        IMU=$!
        sleep 3
    else
        echo "$(date +%T) no IMU on the I2C bus, turning open loop" >> "$LOG/imu.log"
    fi

    python3 rover_bridge.py --ros-args -p speed_limit:=$SPEED_LIMIT -p max_speed:=$MAX_SPEED \
        -p deadband:=$DEADBAND >> "$LOG/bridge.log" 2>&1 &
    BRIDGE=$!

    python3 rover_remote.py --ros-args -p max_linear:=$MAX_LINEAR -p max_angular:=$MAX_ANGULAR \
        >> "$LOG/remote.log" 2>&1 &
    REMOTE=$!

    wait -n $IMU $BRIDGE $REMOTE            # returns as soon as any one stops
    kill -INT $BRIDGE $REMOTE $IMU 2>/dev/null
    wait 2>/dev/null
    sleep 2
done
