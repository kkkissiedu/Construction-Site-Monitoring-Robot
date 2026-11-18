"""
mpu6050.py — read an MPU-6050 (GY-521) over Linux I2C without extra libraries, with a small CLI.

Usage:
    python3 mpu6050.py probe                 # print live accel (g) and gyro (deg/s) for a few seconds
    python3 mpu6050.py axes                  # 45 s tilt test that finds the board axes
"""

# %% Imports
import argparse
import fcntl
import os
import struct
import time


# %% Config
BUS           = 7                # header pins 3 (SDA) and 5 (SCL) on the Orin NX dev kit
ADDRESS       = 0x68
I2C_SLAVE     = 0x0703
ACCEL_SCALE   = 8192.0           # LSB per g at +-4 g
GYRO_SCALE    = 65.5             # LSB per deg/s at +-500 deg/s
RETRIES       = 3
SAMPLE_HZ     = 50.0

PWR_MGMT_1    = 0x6B
SMPLRT_DIV    = 0x19
CONFIG        = 0x1A
GYRO_CONFIG   = 0x1B
ACCEL_CONFIG  = 0x1C
ACCEL_XOUT_H  = 0x3B
WHO_AM_I      = 0x75

AXES_SECONDS  = 45.0             # recording length of the tilt test
HOLD_STD      = 0.03             # g, a hold is steady when every axis varies less than this
HOLD_SECONDS  = 2.0              # shortest steady stretch that counts as a pose
SAME_POSE     = 0.07             # g, two readings closer than this are the same pose


# %% Sensor
class MPU6050:

    def __init__(self, bus: int = BUS, address: int = ADDRESS) -> None:
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, address)

        self.who_am_i = self.read_reg(WHO_AM_I, 1)[0]
        self.write_reg(PWR_MGMT_1,   0x01)      # wake up, clock from the gyro PLL
        time.sleep(0.1)
        self.write_reg(SMPLRT_DIV,   4)         # 1 kHz / 5 = 200 Hz internal rate
        self.write_reg(CONFIG,       3)         # low-pass filter about 44 Hz
        self.write_reg(GYRO_CONFIG,  0x08)      # +-500 deg/s
        self.write_reg(ACCEL_CONFIG, 0x08)      # +-4 g
        time.sleep(0.1)

    def write_reg(self, reg: int, value: int) -> None:
        for attempt in range(RETRIES):
            try:
                os.write(self.fd, bytes([reg, value]))
                return
            except OSError:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(0.002)

    def read_reg(self, reg: int, count: int) -> bytes:
        for attempt in range(RETRIES):
            try:
                os.write(self.fd, bytes([reg]))
                return os.read(self.fd, count)
            except OSError:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(0.002)

    def read(self) -> tuple:
        """Return ((ax, ay, az) in g, (gx, gy, gz) in deg/s) in the board's own axes."""
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", self.read_reg(ACCEL_XOUT_H, 14))
        return ((ax / ACCEL_SCALE, ay / ACCEL_SCALE, az / ACCEL_SCALE),
                (gx / GYRO_SCALE,  gy / GYRO_SCALE,  gz / GYRO_SCALE))

    def close(self) -> None:
        os.close(self.fd)


# %% Helpers
def mean(rows: list) -> tuple:
    return tuple(sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0])))


def spread(rows: list) -> float:
    centre = mean(rows)
    return max(max(abs(r[i] - centre[i]) for r in rows) for i in range(len(centre)))


def distance(a: tuple, b: tuple) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def steady_holds(accel: list) -> list:
    """Return [(mean accel, seconds)] for each steady stretch, in time order."""
    window = int(SAMPLE_HZ)                        # one second
    step   = window // 2
    holds  = []                                    # [sum vector, count, mean]

    for start in range(0, len(accel) - window, step):
        chunk = accel[start:start + window]
        if spread(chunk) > HOLD_STD:
            holds.append(None)                     # a gap while moving
            continue
        centre = mean(chunk)
        last = next((h for h in reversed(holds[-1:]) if h is not None), None)
        if last is not None and distance(last["mean"], centre) < SAME_POSE:
            last["chunks"].append(chunk)
            last["mean"] = mean([r for c in last["chunks"] for r in c])
        else:
            holds.append({"chunks": [chunk], "mean": centre})

    result = []
    for h in holds:
        if h is not None:
            seconds = len(h["chunks"]) * step / SAMPLE_HZ + (window - step) / SAMPLE_HZ
            if seconds >= HOLD_SECONDS:
                result.append((h["mean"], seconds))
    return result


def classify(holds: list):
    """Split steady holds into level (the pose held longest in total), then nose up, then right side down.

    Returns (level mean, nose change, right change) or None if the two tilts cannot be told from level.
    """
    anchor = max(holds, key=lambda h: sum(g[1] for g in holds if distance(g[0], h[0]) < SAME_POSE))[0]
    level  = [h for h in holds if distance(h[0], anchor) < SAME_POSE]
    others = [h for h in holds if distance(h[0], anchor) >= SAME_POSE]
    if len(others) < 2:
        return None

    level_mean = mean([h[0] for h in level])
    nose  = tuple(others[0][0][i] - level_mean[i] for i in range(3))
    right = tuple(others[1][0][i] - level_mean[i] for i in range(3))
    return level_mean, nose, right


def axis_name(index: int) -> str:
    return "XYZ"[index]


def probe(imu: MPU6050, seconds: float) -> None:
    print(f"WHO_AM_I = 0x{imu.who_am_i:02X} (0x68 is an MPU-6050)")
    end = time.time() + seconds
    while time.time() < end:
        (ax, ay, az), (gx, gy, gz) = imu.read()
        print(f"accel g  x {ax:+.2f}  y {ay:+.2f}  z {az:+.2f}   gyro deg/s  x {gx:+6.1f}  y {gy:+6.1f}  z {gz:+6.1f}")
        time.sleep(0.2)


def axes(imu: MPU6050) -> None:
    print(f"Recording {AXES_SECONDS:.0f} s. Level, nose UP, level, right side DOWN, level, about 6 s each, 15-20 degrees.", flush=True)
    accel, gyro = [], []
    end = time.time() + AXES_SECONDS
    while time.time() < end:
        a, g = imu.read()
        accel.append(a)
        gyro.append(g)
        time.sleep(1.0 / SAMPLE_HZ)

    holds = steady_holds(accel)
    print(f"steady holds found: {len(holds)}")
    for i, (m, seconds) in enumerate(holds):
        print(f"  {i + 1}: x {m[0]:+.2f}  y {m[1]:+.2f}  z {m[2]:+.2f} g  for {seconds:.1f} s")

    if len(holds) < 3:
        print("Too few steady holds. Hold each pose still for about 6 s and run it again.")
        return

    result = classify(holds)
    if result is None:
        print("Could not tell the two tilts apart from level: tilt a bit more (15-20 degrees) and hold still.")
        return

    level_mean, nose, right = result
    bias = mean(gyro[: int(SAMPLE_HZ * 2)])

    print(f"\nlevel accel (g): x {level_mean[0]:+.2f}  y {level_mean[1]:+.2f}  z {level_mean[2]:+.2f}")
    print(f"gyro at the start (deg/s): x {bias[0]:+.2f}  y {bias[1]:+.2f}  z {bias[2]:+.2f}")
    print(f"change with nose up:         x {nose[0]:+.2f}  y {nose[1]:+.2f}  z {nose[2]:+.2f}")
    print(f"change with right side down: x {right[0]:+.2f}  y {right[1]:+.2f}  z {right[2]:+.2f}")

    up  = max(range(3), key=lambda i: abs(level_mean[i]))
    fwd = max(range(3), key=lambda i: abs(nose[i]))
    rgt = max(range(3), key=lambda i: abs(right[i]))

    print("\nBoard axes, as the accelerometer sees them:")
    print(f"  {axis_name(up)} points {'UP' if level_mean[up] > 0 else 'DOWN'} (it reads {level_mean[up]:+.2f} g at rest)")
    print(f"  {axis_name(fwd)} points {'FORWARD' if nose[fwd] > 0 else 'BACKWARD'} (nose up changed it by {nose[fwd]:+.2f} g)")
    print(f"  {axis_name(rgt)} points {'LEFT' if right[rgt] > 0 else 'RIGHT'} (right side down changed it by {right[rgt]:+.2f} g)")
    if len({up, fwd, rgt}) < 3:
        print("  The same axis was picked twice: the tilts were too small or mixed, run it again.")


# %% Entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="MPU-6050 live readout and axis finder")
    parser.add_argument("command", choices=["probe", "axes"])
    parser.add_argument("--bus", type=int, default=BUS)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    imu = MPU6050(args.bus)
    try:
        if args.command == "probe":
            probe(imu, args.seconds)
        else:
            axes(imu)
    finally:
        imu.close()


if __name__ == "__main__":
    main()
