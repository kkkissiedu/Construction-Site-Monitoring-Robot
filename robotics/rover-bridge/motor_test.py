"""
motor_test.py — pulse each motor in turn so you can check wheel position and direction.

Usage:
    python3 motor_test.py                 # wheels lifted, rover software stopped
Order: front-left forward, front-left reverse, back-left forward, back-left reverse,
front-right forward, back-right forward. Each pulse lasts 1.5 s, one every 5 s.
"""

# %% Imports
import time

from rover_link import RoverLink


# %% Config
LEAD_S  = 15            # s before the first pulse
GAP_S   = 5             # s between pulses
SPEED   = 120
STEPS   = [(1, SPEED), (1, -SPEED), (2, SPEED), (2, -SPEED), (3, SPEED), (4, SPEED)]
NAMES   = {1: "front-left", 2: "back-left", 3: "front-right", 4: "back-right"}


# %% Main function
def main() -> None:
    # the port stays open for the whole run: closing it sends a stop that cancels a pulse
    link = RoverLink()
    try:
        link.configure({"limit": 255})
        time.sleep(LEAD_S)
        for motor, speed in STEPS:
            reply = link.send(f"T {motor} {speed}")
            print(f"{time.strftime('%H:%M:%S')} {NAMES[motor]:11s} {'forward' if speed > 0 else 'reverse'} -> {reply}", flush=True)
            time.sleep(GAP_S)
    finally:
        link.close()


# %% Entry point
if __name__ == "__main__":
    main()
