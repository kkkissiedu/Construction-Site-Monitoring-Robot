"""
calibrate_deadband.py — find the smallest motor command that makes the rover move.

Usage:
    python3 calibrate_deadband.py            # rover on the floor, clear space, stop the autostart first
Watch the wheels and press Ctrl+C the moment the rover starts to roll; the last number printed is the
deadband. The ramp stops by itself at 150 and the rover is always stopped on exit.
"""

# %% Imports
import time

from rover_link import RoverLink


# %% Config
PORT      = "/dev/ttyTHS1"
START     = 20
STOP      = 150
STEP      = 2
HOLD_S    = 0.8       # s per step
SEND_S    = 0.1       # s between commands, well inside the firmware failsafe (500 ms)


# %% Main function
def main() -> None:
    with RoverLink(PORT) as link:
        link.configure({"limit": 255})
        try:
            for level in range(START, STOP + 1, STEP):
                print(f"command {level}", flush=True)
                end = time.monotonic() + HOLD_S
                while time.monotonic() < end:
                    link.drive(level, level)
                    time.sleep(SEND_S)
            print("reached the end of the ramp without you stopping it")
        except KeyboardInterrupt:
            print("stopped")
        finally:
            link.stop()


# %% Entry point
if __name__ == "__main__":
    main()
